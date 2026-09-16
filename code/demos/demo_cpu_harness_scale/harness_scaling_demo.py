#!/usr/bin/env python3
"""Standalone, offline demo of the async-RL rollout scaling law:

        throughput  =  min( inference , sandbox , harness )

It runs ONE task family end-to-end (DeepScaleR math, expressed coding-shape: the agent writes
`solution.py` that sets a module-level `answer`, and a numeric-assert checker `run_checks.py`
scores it — RLVR, no LLM judge) through a pool of **W warm in-process harness workers** driven by
asyncio. The three replicated tiers of the rollout are each modelled as a capacity + a service
time, so you can watch which one binds:

  * harness   — W warm workers pulling episodes off a shared asyncio.Queue (THE knob we sweep).
  * inference — a shared semaphore of CINF concurrent generations, each taking ~TINF seconds
                (models slime's SGLang router admitting CINF in-flight `/generate` calls).
  * sandbox   — a shared semaphore of CSBX concurrent checker runs, each taking ~TSBX seconds
                (models K AgentCore sandbox Runtimes; the REAL LocalSandbox subprocess runs too,
                so its cost rides inside this tier — this is a genuine end-to-end episode).

The point: with a single/small harness the *harness* is the ceiling (throughput ≈ W / T_episode);
raising W lifts that ceiling until it hits min(inference, sandbox) and throughput plateaus. Nothing
here is fabricated — throughput is measured wall-clock, and the per-tier ceilings are computed from
the service times MEASURED during the run (mean time each semaphore was held). The plateau height
is set by whichever of inference/sandbox is scarcer; raise CINF (or CSBX) to lift it, which is the
whole argument for scaling the three tiers *jointly*.

Offline: no AWS, no GPU, no network. Runs in a few seconds. See README.md for the architecture and
load-balance diagram, and ../../agentcore_slime_demo/harness_pool.py for the real slime integration
(a warm worker pool behind the `--custom-generate-function-path` hook) this demo abstracts.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path

from agentcore_slime_demo.harness import run_episode
from agentcore_slime_demo.policy import MockPolicy
from agentcore_slime_demo.sandbox import LocalSandbox
from agentcore_slime_demo.task import task_from_dict

# DeepScaleR rows carry the reward as a numeric assert on the bare name `answer`, e.g.
#   assert abs(answer - 26) <= 0.0001, f'answer={answer} expected 26'
# We parse the gold out so a deterministic MockPolicy can solve the episode in one turn (the
# reference answer is stripped from real-RL prompts; here it only drives the offline mock).
_GOLD = re.compile(r"abs\(answer\s*-\s*([-\d.eE+]+)\)")

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------------------------
# Capacity wrappers — model the inference engine and the sandbox as a shared concurrency limit
# plus a per-call service time, and MEASURE the service time actually incurred.
# --------------------------------------------------------------------------------------------
class Meter:
    """Accumulates held-time + count for each tier so we can report measured ceilings."""

    def __init__(self) -> None:
        self.inf_time = 0.0
        self.inf_n = 0
        self.sbx_time = 0.0
        self.sbx_n = 0

    @property
    def mean_inf(self) -> float:
        return self.inf_time / self.inf_n if self.inf_n else 0.0

    @property
    def mean_sbx(self) -> float:
        return self.sbx_time / self.sbx_n if self.sbx_n else 0.0


class CappedPolicy:
    """Wrap a policy behind a shared `sem` (CINF in-flight) + an injected `latency` (~TINF)."""

    def __init__(self, inner, sem: asyncio.Semaphore, latency: float, meter: Meter) -> None:
        self._inner = inner
        self._sem = sem
        self._latency = latency
        self._meter = meter

    async def complete(self, messages, sampling_params):
        async with self._sem:
            t = time.perf_counter()
            await asyncio.sleep(self._latency)
            out = await self._inner.complete(messages, sampling_params)
            dt = time.perf_counter() - t
        self._meter.inf_time += dt
        self._meter.inf_n += 1
        return out


class CappedSandbox:
    """Wrap a real LocalSandbox: gate `exec` (the checker run — the representative sandbox op)
    behind a shared `sem` (CSBX concurrent) + injected `latency` (~TSBX). write_file/read_file
    pass straight through (cheap local ops); the real subprocess cost of `exec` rides inside the
    measured sandbox service time, so this stays a genuine end-to-end episode."""

    def __init__(self, inner: LocalSandbox, sem: asyncio.Semaphore, latency: float, meter: Meter) -> None:
        self._inner = inner
        self._sem = sem
        self._latency = latency
        self._meter = meter

    async def __aenter__(self) -> "CappedSandbox":
        await self._inner.__aenter__()
        return self

    async def __aexit__(self, *exc) -> None:
        await self._inner.__aexit__(*exc)

    async def write_file(self, sandbox_path, content, *, user: str = "root") -> None:
        await self._inner.write_file(sandbox_path, content, user=user)

    async def read_file(self, *args, **kwargs):
        return await self._inner.read_file(*args, **kwargs)

    async def exec(self, command, timeout: int = 60):
        async with self._sem:
            t = time.perf_counter()
            await asyncio.sleep(self._latency)
            out = await self._inner.exec(command, timeout=timeout)
            dt = time.perf_counter() - t
        self._meter.sbx_time += dt
        self._meter.sbx_n += 1
        return out


# --------------------------------------------------------------------------------------------
# Task loading — DeepScaleR rows -> coding-shape math tasks the mock can solve in one turn.
# --------------------------------------------------------------------------------------------
def load_tasks(path: Path, n: int):
    """Return n (task, gold_str) pairs, cycling the file if it has fewer than n rows."""
    pairs = []
    lines = [l for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")]
    i = 0
    while len(pairs) < n and lines:
        d = json.loads(lines[i % len(lines)])
        i += 1
        meta = d["metadata"]["task"]
        checks = meta.get("spec", {}).get("checks", [])
        m = _GOLD.search(checks[0]) if checks else None
        if not m:
            continue
        task = task_from_dict(meta)            # -> MathTask (family="math")
        pairs.append((task, m.group(1)))
        if i > len(lines) * 3 and not pairs:   # guard against an all-unparseable file
            break
    return pairs


# --------------------------------------------------------------------------------------------
# The W-worker pool over one task at a time.
# --------------------------------------------------------------------------------------------
async def _worker(queue, inf_sem, sbx_sem, lat_inf, lat_sbx, meter, solved):
    while True:
        item = await queue.get()
        try:
            if item is None:
                return
            task, gold = item
            policy = CappedPolicy(MockPolicy([f"answer = {gold}"]), inf_sem, lat_inf, meter)
            async with CappedSandbox(LocalSandbox(workdir="/workspace"), sbx_sem, lat_sbx, meter) as sb:
                traj = await run_episode(policy, sb, task, {"max_new_tokens": 64, "temperature": 0.0})
            solved.append(bool(traj.solved))
        finally:
            queue.task_done()


async def run_one(w: int, tasks, cfg) -> dict:
    """Drive `tasks` through `w` warm workers; return measured throughput + tier ceilings."""
    inf_sem = asyncio.Semaphore(cfg["cinf"])
    sbx_sem = asyncio.Semaphore(cfg["csbx"])
    queue: asyncio.Queue = asyncio.Queue()
    for t in tasks:
        queue.put_nowait(t)
    for _ in range(w):
        queue.put_nowait(None)   # one shutdown sentinel per worker

    meter = Meter()
    solved: list[bool] = []
    t0 = time.perf_counter()
    workers = [
        asyncio.create_task(
            _worker(queue, inf_sem, sbx_sem, cfg["tinf"], cfg["tsbx"], meter, solved)
        )
        for _ in range(w)
    ]
    await asyncio.gather(*workers)
    elapsed = time.perf_counter() - t0

    n = len(tasks)
    thr = n / elapsed if elapsed else 0.0
    t_ep = meter.mean_inf + meter.mean_sbx           # 1-turn episode critical path
    cap_harness = w / t_ep if t_ep else float("inf")
    cap_inf = cfg["cinf"] / meter.mean_inf if meter.mean_inf else float("inf")
    cap_sbx = cfg["csbx"] / meter.mean_sbx if meter.mean_sbx else float("inf")
    ceilings = {"inference": cap_inf, "sandbox": cap_sbx, "harness": cap_harness}
    bind = min(ceilings, key=ceilings.get)
    return {
        "W": w,
        "elapsed": elapsed,
        "throughput": thr,
        "solved": sum(solved),
        "n": n,
        "cap_inf": cap_inf,
        "cap_sbx": cap_sbx,
        "cap_harness": cap_harness,
        "predicted": min(ceilings.values()),
        "bind": bind,
    }


async def main_async(args) -> int:
    tasks = load_tasks(Path(args.data), args.episodes)
    if not tasks:
        print(f"no parseable numeric tasks in {args.data}", flush=True)
        return 1
    cfg = {"cinf": args.cinf, "csbx": args.csbx, "tinf": args.tinf, "tsbx": args.tsbx}
    ws = [int(x) for x in args.workers.split(",") if x.strip()]

    print(
        f"harness-scaling demo | task=DeepScaleR(math, coding-shape) episodes={len(tasks)}\n"
        f"inference: CINF={cfg['cinf']} @ ~{cfg['tinf']*1000:.0f}ms   "
        f"sandbox: CSBX={cfg['csbx']} @ ~{cfg['tsbx']*1000:.0f}ms + real subprocess\n"
        f"sweeping harness workers W over {ws}\n"
    )
    hdr = f"{'W':>3} {'ep/s':>8} {'solved':>8} {'bind':>10}   ceilings ep/s [inf | sbx | harness]"
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for w in ws:
        r = await run_one(w, tasks, cfg)
        rows.append(r)
        print(
            f"{r['W']:>3} {r['throughput']:>8.1f} {r['solved']:>4}/{r['n']:<3} "
            f"{r['bind']:>10}   {r['cap_inf']:>6.0f} | {r['cap_sbx']:>5.0f} | {r['cap_harness']:>7.1f}"
        )

    # Honest read-out: where the knee is, and that the plateau is set by the scarcer of the two
    # non-harness tiers (raise its capacity to lift the plateau — the joint-scaling argument).
    plateau = max(r["throughput"] for r in rows)
    non_harness = min(rows[-1]["cap_inf"], rows[-1]["cap_sbx"])
    binder = "inference" if rows[-1]["cap_inf"] <= rows[-1]["cap_sbx"] else "sandbox"
    print(
        f"\nthroughput rises ~linearly with W while the harness binds, then plateaus at "
        f"~{plateau:.0f} ep/s once {binder} binds (~{non_harness:.0f} ep/s).\n"
        f"=> throughput = min(inference, sandbox, harness). Raising W alone cannot pass the "
        f"{binder} ceiling; lift it by scaling {binder} too (raise "
        f"{'CINF' if binder == 'inference' else 'CSBX'})."
    )
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.json}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(HERE / "data" / "deepscaler_train.jsonl"))
    p.add_argument("--episodes", type=int, default=int(os.environ.get("EPISODES", "48")))
    p.add_argument("--workers", default=os.environ.get("WORKERS", "1,2,4,6,8,12"),
                   help="comma-separated W values to sweep")
    p.add_argument("--cinf", type=int, default=int(os.environ.get("CINF", "4")),
                   help="inference concurrency ceiling (SGLang in-flight generations)")
    p.add_argument("--csbx", type=int, default=int(os.environ.get("CSBX", "8")),
                   help="sandbox concurrency ceiling (concurrent checker runs)")
    p.add_argument("--tinf", type=float, default=float(os.environ.get("TINF", "0.10")),
                   help="modelled per-generation service time (seconds)")
    p.add_argument("--tsbx", type=float, default=float(os.environ.get("TSBX", "0.05")),
                   help="modelled per-checker service time (seconds); real subprocess adds to it")
    p.add_argument("--json", default=os.environ.get("JSON_OUT", ""),
                   help="optional path to write the per-W results as JSON")
    return p


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async(build_parser().parse_args())))
