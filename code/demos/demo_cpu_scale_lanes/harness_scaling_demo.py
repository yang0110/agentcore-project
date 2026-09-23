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

REAL-remote mode (opt-in): set REAL_SANDBOX=1 (or SANDBOX=<real backend>, e.g. agentcore-runtime)
to make the SANDBOX tier's service time a genuine measured round-trip against a live remote
AgentCore sandbox instead of the simulated `--tsbx` sleep. In that mode CappedSandbox wraps a real
`make_sandbox(...)` backend and drops the injected sleep, so the Meter records the true remote
latency; the `--csbx` semaphore still caps concurrent ops, and inference stays mocked (`--tinf`).
Each episode is then a real network round-trip, so keep --episodes/EPISODES modest. The default
(REAL_SANDBOX unset / SANDBOX=local) remains the fully offline analytical sweep, unchanged.
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
from agentcore_slime_demo.sandbox import LocalSandbox, make_sandbox
from agentcore_slime_demo.task import task_from_dict

# DeepScaleR rows carry the reward as a numeric assert on the bare name `answer`, e.g.
#   assert abs(answer - 26) <= 0.0001, f'answer={answer} expected 26'
# We parse the gold out so a deterministic MockPolicy can solve the episode in one turn (the
# reference answer is stripped from real-RL prompts; here it only drives the offline mock).
_GOLD = re.compile(r"abs\(answer\s*-\s*([-\d.eE+]+)\)")

HERE = Path(__file__).resolve().parent


def _real_sandbox_selected() -> bool:
    """True in REAL-remote mode: measure the sandbox tier against a live remote AgentCore sandbox
    instead of the simulated --tsbx sleep. Opt in with REAL_SANDBOX=1, or by setting SANDBOX to any
    real backend (anything other than 'local'). Default (both unset / SANDBOX=local) is offline."""
    if os.environ.get("REAL_SANDBOX", "").strip().lower() in ("1", "true", "yes", "on"):
        return True
    sb = os.environ.get("SANDBOX", "").strip().lower()
    return bool(sb) and sb != "local"


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
    """Wrap a sandbox: gate `exec` (the checker run — the representative sandbox op) behind a
    shared `sem` (CSBX concurrent) + injected `latency` (~TSBX). write_file/read_file pass straight
    through; the real subprocess cost of `exec` rides inside the measured sandbox service time, so
    this stays a genuine end-to-end episode.

    Two modes for the inner sandbox:
      * OFFLINE (default, `real=False`) — inner is a LocalSandbox subprocess; `exec` sleeps
        `latency` (~TSBX) to model the remote sandbox's service time before the local checker runs.
      * REAL   (`real=True`)            — inner is a live remote backend from make_sandbox(); the
        injected sleep is DROPPED so the metered window is the genuine remote round-trip (no
        double-counting). The `sem` (CSBX) ceiling still applies; the Meter records real latency."""

    def __init__(self, inner, sem: asyncio.Semaphore, latency: float, meter: Meter,
                 real: bool = False) -> None:
        self._inner = inner
        self._sem = sem
        self._latency = latency
        self._meter = meter
        self._real = real

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
            if not self._real:               # OFFLINE: inject the modelled ~TSBX service time.
                await asyncio.sleep(self._latency)   # REAL: measure the genuine round-trip instead.
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
async def _worker(queue, inf_sem, sbx_sem, lat_inf, lat_sbx, meter, solved, cfg):
    # REAL mode opens ONE live remote sandbox per worker and reuses it across every episode that
    # worker pulls, closing it in a finally. A real session is a full start round-trip (far too
    # heavy to reopen per episode), so per-worker is the clean choice; make_sandbox() hands each
    # worker the next Runtime ARN round-robin, so W workers fan out across the configured ARNs.
    # The --csbx semaphore still caps concurrent exec ops across all workers. OFFLINE mode is
    # unchanged: a fresh cheap LocalSandbox per episode with the injected ~TSBX sleep.
    worker_sbx = None
    if cfg["real_sbx"]:
        inner = make_sandbox(cfg["sbx_kind"], workdir="/workspace")
        worker_sbx = CappedSandbox(inner, sbx_sem, lat_sbx, meter, real=True)
        await worker_sbx.__aenter__()
    try:
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return
                task, gold = item
                policy = CappedPolicy(MockPolicy([f"answer = {gold}"]), inf_sem, lat_inf, meter)
                sampling = {"max_new_tokens": 64, "temperature": 0.0}
                if worker_sbx is not None:                    # REAL: reuse this worker's remote sandbox
                    traj = await run_episode(policy, worker_sbx, task, sampling)
                    solved.append(bool(traj.solved))
                else:                                         # OFFLINE: fresh LocalSandbox per episode
                    async with CappedSandbox(LocalSandbox(workdir="/workspace"), sbx_sem, lat_sbx, meter) as sb:
                        traj = await run_episode(policy, sb, task, sampling)
                    solved.append(bool(traj.solved))
            finally:
                queue.task_done()
    finally:
        if worker_sbx is not None:
            await worker_sbx.__aexit__(None, None, None)


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
            _worker(queue, inf_sem, sbx_sem, cfg["tinf"], cfg["tsbx"], meter, solved, cfg)
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


async def run_balanced(tasks, cfg, ns: list[int], args) -> int:
    """Balanced joint sweep: for each N, scale inference=sandbox=harness=N in lockstep (4+4+4).

    Contrast with the harness-only sweep: there CINF/CSBX are fixed and only W rises, so throughput
    plateaus once it hits min(inference, sandbox). Here all three rise together, so no single tier
    is the ceiling and throughput scales ~linearly with N until the host itself (cores / the GIL /
    the real remote sandbox's own limits) saturates."""
    sbx_line = (
        f"REAL remote={cfg['sbx_kind']} (measured round-trip, --tsbx ignored)"
        if cfg["real_sbx"] else f"~{cfg['tsbx']*1000:.0f}ms + real subprocess"
    )
    print(
        f"harness-scaling demo | BALANCED joint sweep | task=DeepScaleR(math) episodes={len(tasks)}\n"
        f"each N scales inference=sandbox=harness=N together (the 4+4+4 case)\n"
        f"inference @ ~{cfg['tinf']*1000:.0f}ms   sandbox {sbx_line}\n"
        f"sweeping tiers N over {ns}\n"
    )
    hdr = f"{'N':>3} {'I+S+H':>7} {'ep/s':>8} {'solved':>8} {'bind':>10}   ceilings ep/s [inf | sbx | harness]"
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for n in ns:
        cfg_n = dict(cfg, cinf=n, csbx=n)          # inference = sandbox = harness = N
        r = await run_one(n, tasks, cfg_n)
        r["N"] = n
        rows.append(r)
        print(
            f"{n:>3} {f'{n}+{n}+{n}':>7} {r['throughput']:>8.1f} {r['solved']:>4}/{r['n']:<3} "
            f"{r['bind']:>10}   {r['cap_inf']:>6.0f} | {r['cap_sbx']:>5.0f} | {r['cap_harness']:>7.1f}"
        )

    if len(rows) >= 2:
        lo, hi = rows[0], rows[-1]
        span = hi["N"] / lo["N"] if lo["N"] else 1.0
        gain = hi["throughput"] / lo["throughput"] if lo["throughput"] else 0.0
        print(
            f"\nscaling the tiers jointly {lo['N']}+{lo['N']}+{lo['N']} -> {hi['N']}+{hi['N']}+{hi['N']} "
            f"({span:.0f}x) lifted throughput ~{gain:.1f}x — near-linear, because no single tier binds "
            f"(each N raises inference, sandbox, AND harness at once).\n"
            f"=> throughput = min(inference, sandbox, harness): the harness-only sweep plateaus at "
            f"min(inf,sbx); the balanced sweep moves all three, so the min moves with N."
        )
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.json}")
    return 0


async def main_async(args) -> int:
    tasks = load_tasks(Path(args.data), args.episodes)
    if not tasks:
        print(f"no parseable numeric tasks in {args.data}", flush=True)
        return 1
    # REAL mode: the sandbox tier's service time is a live remote round-trip (measured), not the
    # simulated --tsbx sleep. Selected by REAL_SANDBOX=1, or by SANDBOX set to any real backend
    # (anything other than 'local'). sbx_kind defaults to 'agentcore-runtime' when only REAL_SANDBOX
    # is set. Default (both unset / SANDBOX=local) stays the fully offline analytical sweep.
    real_sbx = _real_sandbox_selected()
    sbx_kind = os.environ.get("SANDBOX", "agentcore-runtime")
    cfg = {"cinf": args.cinf, "csbx": args.csbx, "tinf": args.tinf, "tsbx": args.tsbx,
           "real_sbx": real_sbx, "sbx_kind": sbx_kind}

    # BALANCED joint-scaling mode: --tiers N,... sets inference=sandbox=harness=N together (4+4+4),
    # the complement to the harness-only sweep. Because every tier scales in lockstep, no single one
    # binds, so throughput should rise ~linearly with N — that is the whole point of scaling the
    # three tiers *jointly* rather than lifting only the harness (which plateaus at min(inf,sbx)).
    if args.tiers.strip():
        return await run_balanced(tasks, cfg, [int(x) for x in args.tiers.split(",") if x.strip()], args)

    ws = [int(x) for x in args.workers.split(",") if x.strip()]

    sbx_line = (
        f"sandbox: CSBX={cfg['csbx']} REAL remote={sbx_kind} (measured round-trip, --tsbx ignored)"
        if real_sbx else
        f"sandbox: CSBX={cfg['csbx']} @ ~{cfg['tsbx']*1000:.0f}ms + real subprocess"
    )
    print(
        f"harness-scaling demo | task=DeepScaleR(math, coding-shape) episodes={len(tasks)}\n"
        f"inference: CINF={cfg['cinf']} @ ~{cfg['tinf']*1000:.0f}ms   {sbx_line}\n"
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
                   help="comma-separated W values to sweep (harness-only sweep; CINF/CSBX fixed)")
    p.add_argument("--tiers", default=os.environ.get("TIERS", ""),
                   help="comma-separated N values for the BALANCED joint sweep: each N sets "
                        "inference=sandbox=harness=N together (the 4+4+4 case). When set, this "
                        "replaces the harness-only --workers sweep.")
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
