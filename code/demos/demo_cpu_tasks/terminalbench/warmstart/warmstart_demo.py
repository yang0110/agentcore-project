#!/usr/bin/env python3
"""demo_cpu_tasks/terminalbench/warmstart — the pre-open + heartbeat probe: what a sandbox launch costs, and two ways to stop paying it: pre-open + heartbeat.

No GPU, no model, no slime. The SAME batch of M episodes (a real task family, the real in-process
`run_episode` harness, MockPolicy + a simulated decode delay, a REAL AgentCore sandbox) is run three ways:

  1. cold       Each episode opens a fresh session when it starts and waits for the boot.
  2. preopen    A look-ahead pool opens batch k+1's sessions while batch k runs (the idea behind
                this demo's tb_lookahead_source + TerminalBenchPool, reduced to ~20 lines).
                Episodes find their sandbox already booted; the boot cost moves off the critical path
                and turns into idle session time instead.
  3. heartbeat  Sessions opened and seeded mid-episode, then left silent for GAP_S (a simulated
                weight-sync barrier) that is longer than the Runtime's idleRuntimeSessionTimeout.
                Arm "on": session_keeper.SessionKeeper pings each session (a no-op `exec true`) every
                IDLE_S * 0.5 s. Arm "off": silence. After the gap we check whether the session, its
                files and its microVM (kernel boot_id) survived, and time the first op.

Backends (BACKENDS=runtime,codeinterpreter):
  * runtime          your custom AgentCore Runtime (sandbox_server_agent.py), AGENTCORE_SANDBOX_RUNTIME_ARN
                     from warmstart/runtime.env — created with a short idle timeout by warmstart/deploy_warmstart.py.
  * local            offline subprocess sandbox — a dry run of the code paths, no AWS.
  * codeinterpreter  the managed Code Interpreter (no image, no deploy). Its session timeout is set per
                     session at start (session_timeout_seconds); the heartbeat arm tests whether traffic
                     extends it.

    bash warmstart/run_warmstart.sh                        # deploy (if needed) + run
    MODES=cold,preopen BACKENDS=runtime,codeinterpreter M=24 python3 warmstart_demo.py

Knobs: MODES, BACKENDS, FAMILY (sql), M (24 episodes), BATCH (6 concurrent), THINK_S (2.0 s simulated
decode per policy turn), GAP_S (150), IDLE_S (the Runtime's idle timeout; WARMSTART_IDLE_S from
warmstart/runtime.env, else 60), HB_SESSIONS (3 per arm), OUT (warmstart/logs/warmstart_<ts>).
Writes <OUT>.csv (one row per episode / session) and <OUT>.summary.txt (the table).
"""
from __future__ import annotations

import asyncio
import csv
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent            # demo_cpu_tasks/terminalbench/warmstart
sys.path.insert(0, str(HERE.parent.parent))             # the demo's vendored agentcore_slime_demo/

from agentcore_slime_demo.harness import run_episode  # noqa: E402
from agentcore_slime_demo.policy import MockPolicy  # noqa: E402
from agentcore_slime_demo.session_keeper import KeptSandbox, SessionKeeper  # noqa: E402
from agentcore_slime_demo.task import task_from_dict  # noqa: E402
from agentcore_slime_demo.task_families import FAMILY_CLASSES  # noqa: E402

MODES = os.environ.get("MODES", "cold,preopen,heartbeat").split(",")
BACKENDS = os.environ.get("BACKENDS", "runtime").split(",")
FAMILY = os.environ.get("FAMILY", "sql")
M = int(os.environ.get("M", "24"))
BATCH = int(os.environ.get("BATCH", "6"))
THINK_S = float(os.environ.get("THINK_S", "2.0"))
IDLE_S = float(os.environ.get("IDLE_S") or os.environ.get("WARMSTART_IDLE_S") or "60")
GAP_S = float(os.environ.get("GAP_S", "150"))
HB_SESSIONS = int(os.environ.get("HB_SESSIONS", "3"))
OUT = os.environ.get("OUT") or str(HERE / "logs" / f"warmstart_{time.strftime('%Y%m%d_%H%M%S')}")

BOOT_ID = "cat /proc/sys/kernel/random/boot_id"


def now() -> float:
    return time.perf_counter()


class ThinkingPolicy(MockPolicy):
    """MockPolicy + a fixed per-turn delay standing in for LLM decode, so an episode takes real time
    (without it, there is no 'current batch' for the pre-open pool to overlap with)."""

    async def complete(self, messages, sampling_params):
        await asyncio.sleep(THINK_S)
        return await super().complete(messages, sampling_params)


def tasks() -> list:
    cls = next(c for c in FAMILY_CLASSES if c.family == FAMILY)
    seeds = cls.seeds()
    # Round-trip through the dataset row format, exactly like the trainer does.
    return [task_from_dict({**s.to_row_kwargs(), "mock_attempts": s.mock_attempts})
            for s in (seeds[i % len(seeds)] for i in range(M))]


def new_sandbox(backend: str):
    if backend == "runtime":
        from agentcore_slime_demo.agentcore_runtime_sandbox import AgentCoreRuntimeSandbox
        if not os.environ.get("AGENTCORE_SANDBOX_RUNTIME_ARN"):
            raise SystemExit("BACKENDS=runtime needs AGENTCORE_SANDBOX_RUNTIME_ARN (bash 4_deploy_runtime.sh)")
        return AgentCoreRuntimeSandbox()
    if backend == "codeinterpreter":
        from agentcore_slime_demo.sandbox import AgentCoreSandbox
        # Session timeout = IDLE_S, so the heartbeat arms on both backends defend the same window.
        return AgentCoreSandbox(session_timeout_seconds=int(IDLE_S))
    if backend == "local":                     # offline dry run of the code paths (nothing is reclaimed)
        from agentcore_slime_demo.sandbox import LocalSandbox
        return LocalSandbox()
    raise SystemExit(f"unknown backend {backend!r} (runtime | codeinterpreter | local)")


async def opened(backend: str) -> tuple[object, float, float]:
    """Open (= boot) one session. Returns (sandbox, open_ms, t_ready)."""
    sb = new_sandbox(backend)
    t0 = now()
    if backend == "codeinterpreter":
        # AgentCoreSandbox.__aenter__ builds its CodeInterpreter client synchronously on the event loop
        # (~150 ms each), which would stall the RUNNING batch while the next one pre-opens. Enter it on
        # a worker thread instead; the opened sandbox is then used from this loop as usual.
        await asyncio.to_thread(lambda: asyncio.run(sb.__aenter__()))
    else:
        await sb.__aenter__()
    return sb, (now() - t0) * 1000, now()


async def close(sb) -> None:
    try:
        await sb.__aexit__(None, None, None)
    except Exception as exc:  # noqa: BLE001 — teardown is best-effort
        print(f"  close failed: {exc!r}", flush=True)


async def episode(task, sb) -> tuple[float, float]:
    t0 = now()
    traj = await run_episode(ThinkingPolicy(task.mock_attempts), sb, task)
    return traj.reward, (now() - t0) * 1000


# ------------------------------------------------------------------------------ 1. cold
async def run_cold(backend: str, ts: list) -> list[dict]:
    rows = []

    async def one(i, task, batch_start):
        sb, open_ms, t_ready = await opened(backend)
        try:
            reward, ep_ms = await episode(task, sb)
        finally:
            await close(sb)
        rows.append(dict(mode="cold", backend=backend, episode=i, batch=i // BATCH, open_ms=open_ms,
                         open_wait_ms=(t_ready - batch_start) * 1000, idle_held_ms=0.0,
                         episode_ms=ep_ms, reward=reward, boots=1))

    for b in range(0, M, BATCH):
        t = now()
        await asyncio.gather(*(one(i, ts[i], t) for i in range(b, min(b + BATCH, M))))
    return rows


# ------------------------------------------------------------------------------ 2. preopen
async def run_preopen(backend: str, ts: list) -> list[dict]:
    """Look-ahead = 1 batch: when batch k starts, batch k+1's sessions start booting."""
    rows = []
    batches = [list(range(b, min(b + BATCH, M))) for b in range(0, M, BATCH)]
    pending: dict[int, asyncio.Task] = {}

    def preopen(k):
        if k < len(batches):
            for i in batches[k]:
                pending[i] = asyncio.create_task(opened(backend))

    async def one(i, task, batch_start):
        sb, open_ms, t_ready = await pending.pop(i)
        t_start = now()
        try:
            reward, ep_ms = await episode(task, sb)
        finally:
            await close(sb)
        rows.append(dict(mode="preopen", backend=backend, episode=i, batch=i // BATCH, open_ms=open_ms,
                         open_wait_ms=max(0.0, t_start - batch_start) * 1000,
                         idle_held_ms=max(0.0, batch_start - t_ready) * 1000,
                         episode_ms=ep_ms, reward=reward, boots=1))

    preopen(0)                                   # batch 0 has nothing to hide behind: it pays the boot
    for k, idx in enumerate(batches):
        t = now()
        preopen(k + 1)                           # look ahead while batch k runs
        await asyncio.gather(*(one(i, ts[i], t) for i in idx))
    return rows


# ------------------------------------------------------------------------------ 3. heartbeat
async def run_heartbeat(backend: str, ts: list) -> list[dict]:
    rows = []
    keeper = SessionKeeper(ping_fraction=0.5, tick=2.0)
    await keeper.start()

    async def one(arm: str, j: int):
        sb, open_ms, _ = await opened(backend)
        if arm == "on":
            sb = KeptSandbox(sb, keeper, idle_timeout=IDLE_S)
            # already open: register directly instead of re-entering the inner sandbox
            keeper.register(sb._inner.sandbox_id, sb._heartbeat, IDLE_S)
            sb.sandbox_id = sb._inner.sandbox_id
        marker = f"state-{arm}-{j}"
        await sb.write_file("state.txt", marker)          # the seeded, half-done workspace
        boot0 = (await sb.exec(BOOT_ID))[1].strip()
        t_idle = now()
        await barrier.wait()                              # every session idles for the same GAP_S
        t0 = now()
        kept, err = False, ""
        try:
            code, out, _ = await sb.exec(f"cat state.txt; echo; {BOOT_ID}")
            lines = out.split()
            kept = code == 0 and marker in lines and boot0 in lines
            if not kept:
                err = "new microVM, workspace gone" if boot0 not in lines else "files gone"
        except Exception as exc:  # noqa: BLE001 — a reclaimed Code Interpreter session raises
            err = ("session expired (not active)" if "not active" in str(exc)
                   else f"{type(exc).__name__}: {str(exc)[:60]}")
        resume_ms = (now() - t0) * 1000
        pings = keeper._sessions[sb.sandbox_id].pings if arm == "on" else 0
        if arm == "on":
            keeper.unregister(sb.sandbox_id)
            sb = sb._inner                                # back to the raw sandbox for teardown
        boots = 0
        if not kept:
            # Recovery = what an episode must do after losing its sandbox: a fresh session + re-seed.
            await close(sb)
            t1 = now()
            sb, _, _ = await opened(backend)
            await sb.write_file("state.txt", marker)
            resume_ms += (now() - t1) * 1000
            boots = 1
        rows.append(dict(mode=f"heartbeat_{arm}", backend=backend, episode=j, batch=0, open_ms=open_ms,
                         open_wait_ms=resume_ms, idle_held_ms=(t0 - t_idle) * 1000, episode_ms=0.0,
                         reward=float(kept), boots=boots, pings=pings, note=err))
        await close(sb)

    class _Barrier:
        """All sessions of both arms start their idle gap together, then GAP_S later resume together."""
        def __init__(self, n):
            self.n, self.ev = n, asyncio.Event()

        async def wait(self):
            self.n -= 1
            if self.n == 0:
                print(f"  [{backend}] all {2 * HB_SESSIONS} sessions seeded — idle for {GAP_S:.0f}s "
                      f"(idle timeout {IDLE_S:.0f}s; 'on' arm heartbeats every {IDLE_S * 0.5:.0f}s)", flush=True)
                keeper.sync_begin()                       # the weight-sync window opens
                await asyncio.sleep(GAP_S)
                keeper.sync_end()
                self.ev.set()
            await self.ev.wait()

    barrier = _Barrier(2 * HB_SESSIONS)
    await asyncio.gather(*(one(arm, j) for arm in ("on", "off") for j in range(HB_SESSIONS)))
    await keeper.aclose()
    return rows


# ------------------------------------------------------------------------------ report
def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else 0.0


def summarize(rows: list[dict], makespan: dict) -> str:
    out = []
    out.append(f"warmstart probe  family={FAMILY} M={M} BATCH={BATCH} THINK_S={THINK_S} "
               f"IDLE_S={IDLE_S:.0f} GAP_S={GAP_S:.0f}")
    out.append("")
    out.append(f"{'backend':16} {'mode':10} {'n':>3} {'open_ms p50':>11} {'wait_ms p50':>11} {'wait_ms p95':>11} "
               f"{'boots':>5} {'boots waited':>12} {'idle held s':>11} {'makespan s':>10} {'reward':>6}")
    for (bk, mode), ms in makespan.items():
        r = [x for x in rows if x["backend"] == bk and x["mode"] == mode]
        if not r or mode.startswith("heartbeat"):
            continue
        w = [x["open_wait_ms"] for x in r]
        out.append(f"{bk:16} {mode:10} {len(r):>3} {statistics.median(x['open_ms'] for x in r):>11.0f} "
                   f"{statistics.median(w):>11.0f} {pct(w, .95):>11.0f} {sum(x['boots'] for x in r):>5} "
                   f"{sum(1 for x in w if x > 250):>12} {sum(x['idle_held_ms'] for x in r) / 1000:>11.1f} "
                   f"{ms:>10.1f} {statistics.mean(x['reward'] for x in r):>6.2f}")
    hb = [x for x in rows if x["mode"].startswith("heartbeat")]
    if hb:
        out.append("")
        out.append(f"heartbeat: sessions idle {GAP_S:.0f}s > idle timeout {IDLE_S:.0f}s")
        out.append(f"{'backend':16} {'arm':13} {'n':>3} {'survived':>8} {'pings':>5} {'resume_ms p50':>13} "
                   f"{'re-boots':>8}  note")
        for bk in BACKENDS:
            for arm in ("heartbeat_on", "heartbeat_off"):
                r = [x for x in hb if x["backend"] == bk and x["mode"] == arm]
                if not r:
                    continue
                notes = sorted({x.get("note", "") for x in r} - {""})
                out.append(f"{bk:16} {arm:13} {len(r):>3} {sum(int(x['reward']) for x in r):>8} "
                           f"{sum(x.get('pings', 0) for x in r):>5} "
                           f"{statistics.median(x['open_wait_ms'] for x in r):>13.0f} "
                           f"{sum(x['boots'] for x in r):>8}  {'; '.join(notes)[:90]}")
    out.append("")
    out.append("wait = time an episode waited for its sandbox before its first op (cold: the boot; preopen:"
               " ~0 unless the look-ahead was late). boots waited = episodes whose wait > 250 ms.")
    out.append("idle held = session-seconds a pre-opened / kept-warm session existed without doing work "
               "(the price of hiding the boot).")
    return "\n".join(out)


async def main() -> int:
    ts = tasks()
    rows: list[dict] = []
    makespan: dict = {}
    t_all = now()
    for bk in BACKENDS:
        for mode in MODES:
            print(f"\n=== backend={bk} mode={mode}", flush=True)
            t = now()
            fn = {"cold": run_cold, "preopen": run_preopen, "heartbeat": run_heartbeat}[mode]
            r = await fn(bk, ts)
            rows += r
            if mode == "heartbeat":
                makespan[(bk, "heartbeat_on")] = makespan[(bk, "heartbeat_off")] = now() - t
            else:
                makespan[(bk, mode)] = now() - t
            print(f"=== backend={bk} mode={mode} done in {now() - t:.1f}s", flush=True)
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    fields = ["mode", "backend", "episode", "batch", "open_ms", "open_wait_ms", "idle_held_ms",
              "episode_ms", "reward", "boots", "pings", "note"]
    with open(f"{OUT}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["backend"], x["mode"], x["episode"])):
            w.writerow({k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()})
    text = summarize(rows, makespan) + f"\n\ntotal wall {now() - t_all:.0f}s -> {OUT}.csv"
    Path(f"{OUT}.summary.txt").write_text(text + "\n")
    print("\n" + text, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
