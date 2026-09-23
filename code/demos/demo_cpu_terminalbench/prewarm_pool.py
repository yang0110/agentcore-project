#!/usr/bin/env python3
"""Prewarm the whole Terminal-Bench Runtime pool up front — and MEASURE per-start seconds.

Run this MANUALLY before a training run. It opens one session against every Runtime in the pool
(`tasks/arn_by_image.json`, written by deploy_terminalbench.py). Opening a session is exactly one
`invoke_agent_runtime` "ping": AgentCore allocates a microVM on the fleet, PULLS the task's image
from ECR, boots it, and starts the /invocations server. So this pass forces the (otherwise
lazy, per-first-episode) image pull + cold boot to happen NOW, concurrently, off the training
critical path — and records how long each one took.

HONEST SCOPE (see docs/fault-recovery.md — AWS runtime-sessions):
  * A microVM is warm only while ITS `runtimeSessionId` is alive; on session completion the microVM
    is terminated and memory sanitized. AWS does NOT document a per-account cross-session image
    cache, so prewarm-and-release is BEST-EFFORT: it pays the pull now, but a later fresh-session
    episode may still cold-start. The GUARANTEED warm lever is session affinity — reuse one
    `runtimeSessionId` per task and heartbeat it (KeptSandbox / SessionKeeper). This script is the
    up-front pull + a real latency probe, not a guarantee.
  * With KEEP_WARM=1 the sessions are left ALIVE (not stopped). They still expire on the ~15-min
    idle timeout unless something keeps pinging them, and each holds a concurrent-session quota
    slot while alive — use it only if you launch training within the idle window.

Usage (from the demo dir, with the pool map already deployed + AWS creds in the env):
    export AGENTCORE_REGION=us-west-2
    python3 prewarm_pool.py                    # warm every Runtime in tasks/arn_by_image.json
    PREWARM_CONC=16 python3 prewarm_pool.py     # 16 opens in flight (default 12)
    TB_TASKS=analyze-access-logs,acl-permissions-inheritance python3 prewarm_pool.py  # subset
    TB_LIMIT=10 python3 prewarm_pool.py         # first 10 only (cheap check)
    KEEP_WARM=1 python3 prewarm_pool.py         # leave sessions alive (session affinity)

Writes tasks/prewarm_times.csv (task, arn_tail, start_s, status) and prints a min/p50/p95/max
summary. Purely reads the pool map + invokes AgentCore; builds/pushes nothing.
"""
from __future__ import annotations

import asyncio
import csv
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # vendored agentcore_slime_demo (this demo's own copy)

from agentcore_slime_demo.agentcore_runtime_sandbox import AgentCoreRuntimeSandbox  # noqa: E402

TASKS_DIR = HERE / "tasks"
POOL_MAP = TASKS_DIR / "arn_by_image.json"
OUT_CSV = TASKS_DIR / "prewarm_times.csv"
WORKDIR = os.environ.get("TB_WORKDIR", "/app")
CONC = int(os.environ.get("PREWARM_CONC", "12"))
KEEP_WARM = os.environ.get("KEEP_WARM", "0") not in ("0", "false", "no", "")


def _load_pool() -> dict[str, str]:
    if not POOL_MAP.exists():
        sys.exit(f"ERROR: {POOL_MAP} not found — run deploy/deploy_terminalbench.py first.")
    pool = json.loads(POOL_MAP.read_text())
    only = {t for t in os.environ.get("TB_TASKS", "").split(",") if t}
    if only:
        pool = {k: v for k, v in pool.items() if k in only}
    limit = int(os.environ.get("TB_LIMIT", "0"))
    if limit:
        pool = dict(list(pool.items())[:limit])
    if not pool:
        sys.exit("ERROR: no Runtimes selected (check the pool map / TB_TASKS / TB_LIMIT).")
    return pool


async def _warm_one(task: str, arn: str, region: str, sem: asyncio.Semaphore) -> dict:
    """Open one session (ping = pull + boot), timing the WHOLE cold start; then release it.

    We do NOT reuse the demo's per-episode session id — the point is to time a genuine cold open.
    On KEEP_WARM we leave the session alive by disabling the sandbox's stop-on-exit teardown.
    """
    async with sem:
        # AGENTCORE_STOP_SESSION controls whether __aexit__ stops the session. Set per-call so the
        # keep-warm choice is explicit and doesn't leak to other code in this process.
        os.environ["AGENTCORE_STOP_SESSION"] = "0" if KEEP_WARM else "1"
        sb = AgentCoreRuntimeSandbox(runtime_arn=arn, region=region, workdir=WORKDIR)
        t0 = time.perf_counter()
        status = "ok"
        try:
            await sb.__aenter__()               # the cold start: allocate + pull + boot + ping
            start_s = time.perf_counter() - t0
        except Exception as e:  # noqa: BLE001 — record the failure, keep warming the rest
            start_s = time.perf_counter() - t0
            status = f"{type(e).__name__}: {e}"[:160]
            print(f"[prewarm] task={task} FAILED after {start_s:.1f}s: {status}", flush=True)
            return {"task": task, "arn_tail": arn.rsplit('/', 1)[-1], "start_s": round(start_s, 2),
                    "status": status}
        finally:
            if not KEEP_WARM:
                try:
                    await sb.__aexit__(None, None, None)   # stop_runtime_session
                except Exception:  # noqa: BLE001
                    pass
        print(f"[prewarm] task={task} start_s={start_s:.2f} status=ok"
              f"{' (kept warm)' if KEEP_WARM else ''}", flush=True)
        return {"task": task, "arn_tail": arn.rsplit('/', 1)[-1], "start_s": round(start_s, 2),
                "status": "ok"}


async def _main() -> int:
    region = os.environ.get("AGENTCORE_REGION", "us-west-2")
    pool = _load_pool()
    print(f"prewarming {len(pool)} Runtimes (conc={CONC}, keep_warm={KEEP_WARM}, region={region})",
          flush=True)
    sem = asyncio.Semaphore(CONC)
    t0 = time.perf_counter()
    rows = await asyncio.gather(*[_warm_one(t, a, region, sem) for t, a in pool.items()])
    wall = time.perf_counter() - t0

    TASKS_DIR.mkdir(exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "arn_tail", "start_s", "status"])
        w.writeheader()
        w.writerows(rows)

    oks = sorted(r["start_s"] for r in rows if r["status"] == "ok")
    fails = [r for r in rows if r["status"] != "ok"]

    def pct(p: float) -> float:
        if not oks:
            return 0.0
        return oks[min(len(oks) - 1, int(p * len(oks)))]

    print("\n=== prewarm summary ===")
    print(f"  warmed OK : {len(oks)}/{len(rows)}   failed: {len(fails)}")
    if oks:
        print(f"  start_s   : min={oks[0]:.2f}  p50={pct(0.5):.2f}  p95={pct(0.95):.2f}  "
              f"max={oks[-1]:.2f}  sum={sum(oks):.1f}")
    print(f"  wall clock: {wall:.1f}s (conc={CONC})")
    print(f"  per-Runtime seconds -> {OUT_CSV}")
    if fails:
        print(f"  {len(fails)} failed (see the CSV status column): "
              f"{[r['task'] for r in fails][:10]}")
    return 0 if oks else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
