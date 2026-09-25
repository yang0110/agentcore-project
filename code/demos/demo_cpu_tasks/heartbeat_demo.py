#!/usr/bin/env python3
"""demo_cpu_tasks HEARTBEAT=1 (was demo_cpu_heartbeat) — quiesce / resume the sandbox across the weight-sync barrier (standalone).

Runs the exact arc from docs/demos-program.md "quiesce / resume the sandbox", end-to-end, in a few
seconds, with NO AWS and NO GPU:

  ACT 1  naive "stop invoking"  — during weight sync we simply stop touching the sandbox. AgentCore
                                  reclaims the idle session; on resume it is GONE. (anti-pattern)
  ACT 2  heartbeat quiesce      — a SessionKeeper (outside the harness) gates new invokes and
                                  heartbeats the session under the idle timeout. It survives a pause
                                  far longer than the timeout; resume is instant.
  ACT 3  recover-if-reclaimed   — even heartbeated, a GPU instance recycle can kill the session
                                  (no snapshot; docs/gpu-snapshot.md). The keeper is necessary, not
                                  sufficient: we detect the death and recover by re-seeding, so the
                                  episode continues instead of crashing.

The heartbeat lives in SessionKeeper + KeptSandbox (agentcore_slime_demo/session_keeper.py), NOT in
the harness — the whole point of the design. By default the idle timeout is modelled locally by
AgentCoreIdleSim (agentcore_idle_sim.py) so the demo runs offline.

    python3 heartbeat_demo.py                 # defaults: idle=4s, pause=10s, tick=0.5s
    IDLE=6 PAUSE=20 python3 heartbeat_demo.py  # a longer pause; still survives on the heartbeat

Run against real AgentCore
--------------------------
Set REAL_AGENTCORE=1 (or SANDBOX to anything other than "local") to run the SAME arc against a real
remote CPU sandbox (Code Interpreter managed microVM). No file needs to be deleted or edited — the
switch is env-only. The inner sandbox is built with make_sandbox(SANDBOX) and wrapped in a real
KeptSandbox whose heartbeat is a real inner.exec("true") round-trip; the idle timer is AWS's own.
Because Code Interpreter's idle timeout is 900s, IDLE defaults to 900 in real mode and a short PAUSE
survives ACT 1 too — seeing a genuine reclamation requires PAUSE > idle_timeout. Keep PAUSE small so
the smoke is fast:

    REAL_AGENTCORE=1 SANDBOX=agentcore AGENTCORE_REGION=us-west-2 PAUSE=30 python3 heartbeat_demo.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from agentcore_slime_demo.recovery import is_recoverable
from agentcore_slime_demo.sandbox import make_sandbox
from agentcore_slime_demo.session_keeper import KeptSandbox, SessionKeeper

# --- mode switch: REAL_AGENTCORE=1 (or SANDBOX != "local") runs against a real remote CPU sandbox;
# otherwise the default is the fully-offline AgentCoreIdleSim simulation. ------------------------
_SANDBOX = os.environ.get("SANDBOX", "").strip()
REAL = os.environ.get("REAL_AGENTCORE", "").strip() in ("1", "true", "yes") \
    or (_SANDBOX not in ("", "local"))

# --- knobs (offline: small so it runs in seconds; real: idle defaults to the 900s Code-Interpreter
# timeout so a short PAUSE survives — see the module docstring). -----------------------------------
IDLE = float(os.environ.get("IDLE", "900" if REAL else "4"))  # AgentCore idle-timeout (s)
PAUSE = float(os.environ.get("PAUSE", "10"))     # how long the weight sync stalls generation (s)
TICK = float(os.environ.get("TICK", "0.5"))      # keeper + idle-sim poll granularity (s)

if not REAL:
    # Offline-only: the local idle-timeout simulator and its reclaimable sandbox. In real mode the
    # timer is AWS's, so we neither import nor use these (make_sandbox builds the real sandbox).
    from agentcore_idle_sim import AgentCoreIdleSim, ReclaimableLocalSandbox

_PROBLEM = "Write solution.py so that `python3 run_checks.py` exits 0.\n"
_SOLUTION = "print('agentcore says 42')\n"
_CHECKS = "import subprocess,sys\n" \
          "out=subprocess.run([sys.executable,'solution.py'],capture_output=True,text=True).stdout\n" \
          "sys.exit(0 if '42' in out else 1)\n"


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


async def seed(sb) -> None:
    """Seed a tiny episode workspace (the state a weight-sync pause must preserve)."""
    await sb.write_file("PROBLEM.md", _PROBLEM)
    await sb.write_file("solution.py", _SOLUTION)
    await sb.write_file("run_checks.py", _CHECKS)


async def run_check(sb) -> int:
    code, _out, _err = await sb.exec("python3 run_checks.py", timeout=30)
    return code


async def weight_sync(seconds: float, *, keeper: SessionKeeper | None = None) -> None:
    """Model the trainer->inference weight sync: generation stalls for `seconds`. If a keeper is
    given, wrap the stall in its sync barrier so it gates invokes + heartbeats meanwhile."""
    print(f"  [trainer] weight sync START — generation stalls for {seconds:.0f}s "
          f"(idle timeout is {IDLE:.0f}s)")
    if keeper is not None:
        keeper.sync_begin()
    try:
        await asyncio.sleep(seconds)
    finally:
        if keeper is not None:
            keeper.sync_end()
    print("  [trainer] weight sync END — generation resumes")


# ------------------------------------------------------------------------------------------------
async def act1_naive() -> None:
    banner("ACT 1 — naive 'stop invoking' (anti-pattern): the idle session is reclaimed")
    async with AgentCoreIdleSim(idle_timeout=IDLE, tick=TICK) as sim:
        sb = ReclaimableLocalSandbox(workdir="/workspace")
        sim.watch(sb)
        async with sb:
            await seed(sb)
            print(f"  episode running: check exit={await run_check(sb)} (0 = solved) on {sb.sandbox_id}")
            # No keeper, no heartbeat: we just stop invoking for the whole sync window.
            await weight_sync(PAUSE)
            print(f"  resume: idle for {sb.idle_seconds():.1f}s during the pause")
            try:
                await run_check(sb)
                print("  UNEXPECTED: session survived (raise IDLE below PAUSE to see the loss)")
            except Exception as exc:  # noqa: BLE001
                print(f"  >> LOST THE SESSION: {exc}")
                print("  >> the harness would crash here — nothing kept the session warm.")


async def act2_heartbeat() -> None:
    banner("ACT 2 — heartbeat quiesce: the keeper holds the session warm across the pause")
    async with AgentCoreIdleSim(idle_timeout=IDLE, tick=TICK) as sim, \
            SessionKeeper(ping_fraction=0.5, tick=TICK) as keeper:
        inner = ReclaimableLocalSandbox(workdir="/workspace")
        sim.watch(inner)
        sb = KeptSandbox(inner, keeper, idle_timeout=IDLE)   # <-- composes keeper with the sandbox
        async with sb:
            await seed(sb)
            print(f"  episode running: check exit={await run_check(sb)} on {sb.sandbox_id}")

            # Prove the GATE: an invoke issued DURING the sync must block until sync_end.
            order: list[str] = []

            async def gated_invoke():
                await sb.exec("echo held-invoke", timeout=30)
                order.append("gated-invoke-ran")

            keeper.sync_begin()
            print(f"  [trainer] weight sync START — gate closed, keeper heartbeats every ~{IDLE*0.5:.0f}s")
            task = asyncio.create_task(gated_invoke())
            await asyncio.sleep(PAUSE)
            order.append("sync-ended")
            keeper.sync_end()
            print("  [trainer] weight sync END")
            await task

            print(f"  gate order: {order}  (invoke waited for sync-end, as intended)")
            print(f"  heartbeats fired during pause: {keeper.total_pings}")
            print(f"  resume: idle only {inner.idle_seconds():.1f}s (heartbeat kept it under {IDLE:.0f}s)")
            print(f"  >> SESSION SURVIVED: check exit={await run_check(sb)} — no reclamation, instant resume")


async def act3_recover() -> None:
    banner("ACT 3 — recover-if-reclaimed: heartbeat is necessary, not sufficient (instance recycle)")
    async with AgentCoreIdleSim(idle_timeout=IDLE, tick=TICK) as sim, \
            SessionKeeper(ping_fraction=0.5, tick=TICK) as keeper:
        inner = ReclaimableLocalSandbox(workdir="/workspace")
        sim.watch(inner)
        sb = KeptSandbox(inner, keeper, idle_timeout=IDLE)
        async with sb:
            await seed(sb)
            print(f"  episode running: check exit={await run_check(sb)} on {sb.sandbox_id}")
            keeper.sync_begin()
            print("  [trainer] weight sync START — keeper heartbeating...")
            await asyncio.sleep(PAUSE / 2)
            # A GPU instance recycle kills the session even though the heartbeat is alive.
            sim.force_recycle(inner, reason="GPU instance recycle mid-sync")
            await asyncio.sleep(PAUSE / 2)
            keeper.sync_end()
            print("  [trainer] weight sync END")
            try:
                await run_check(sb)
            except Exception as exc:  # noqa: BLE001
                print(f"  resume FAILED: {exc}")
                print(f"  recoverable? {is_recoverable(exc)} -> recover by re-seeding on a fresh session")
                # recovery.py's contract: restart on a fresh session + re-write the workspace.
                fresh = ReclaimableLocalSandbox(workdir="/workspace")
                sim.watch(fresh)
                async with fresh as fsb:
                    await seed(fsb)   # re-seed: no snapshot to restore, so re-write the workspace
                    print(f"  >> RECOVERED: re-seeded {fsb.sandbox_id}, check exit={await run_check(fsb)} "
                          f"— episode continues instead of crashing the run")


# ================================================================================================
# REAL mode — the same arc against a real remote AgentCore CPU sandbox (Code Interpreter microVM).
# No AgentCoreIdleSim / ReclaimableLocalSandbox here: the idle timer is AWS's and the heartbeat is a
# real inner.exec("true") round-trip. The inner sandbox comes from make_sandbox(SANDBOX).
# ================================================================================================
def _make_inner():
    """Build a real remote sandbox from the SANDBOX kind (default 'agentcore' = Code Interpreter)."""
    return make_sandbox(_SANDBOX or "agentcore")


async def act1_naive_real() -> None:
    banner("ACT 1 — naive 'stop invoking' (anti-pattern), REAL sandbox: no keeper across the pause")
    async with _make_inner() as sb:
        await seed(sb)
        print(f"  episode running: check exit={await run_check(sb)} (0 = solved) on {sb.sandbox_id}")
        # No keeper, no heartbeat: we just stop invoking for the whole sync window.
        await weight_sync(PAUSE)
        try:
            code = await run_check(sb)
            print(f"  session survived the pause: check exit={code}")
            print(f"  NOTE: real idle timeout is {IDLE:.0f}s, so a {PAUSE:.0f}s PAUSE survives — "
                  f"seeing a real reclamation requires PAUSE > idle_timeout.")
        except Exception as exc:  # noqa: BLE001
            print(f"  >> LOST THE SESSION: {exc}")
            print("  >> the harness would crash here — nothing kept the session warm.")


async def act2_heartbeat_real() -> None:
    banner("ACT 2 — heartbeat quiesce, REAL sandbox: the keeper holds the session warm (primary demo)")
    async with SessionKeeper(ping_fraction=0.5, tick=TICK) as keeper:
        sb = KeptSandbox(_make_inner(), keeper, idle_timeout=IDLE)  # composes keeper + real sandbox
        async with sb:
            await seed(sb)
            print(f"  episode running: check exit={await run_check(sb)} on {sb.sandbox_id}")

            # Prove the GATE: an invoke issued DURING the sync must block until sync_end.
            order: list[str] = []

            async def gated_invoke():
                await sb.exec("echo held-invoke", timeout=30)
                order.append("gated-invoke-ran")

            keeper.sync_begin()
            print(f"  [trainer] weight sync START — gate closed, keeper heartbeats every ~{IDLE*0.5:.0f}s")
            task = asyncio.create_task(gated_invoke())
            await asyncio.sleep(PAUSE)
            order.append("sync-ended")
            keeper.sync_end()
            print("  [trainer] weight sync END")
            await task

            print(f"  gate order: {order}  (invoke waited for sync-end, as intended)")
            print(f"  heartbeats fired during pause: {keeper.total_pings}")
            print(f"  >> SESSION SURVIVED: check exit={await run_check(sb)} — no reclamation, instant resume")


async def act3_recover_real() -> None:
    banner("ACT 3 — recover-if-reclaimed, REAL sandbox: re-seed on a genuine session death")
    async with SessionKeeper(ping_fraction=0.5, tick=TICK) as keeper:
        sb = KeptSandbox(_make_inner(), keeper, idle_timeout=IDLE)
        async with sb:
            await seed(sb)
            print(f"  episode running: check exit={await run_check(sb)} on {sb.sandbox_id}")
            keeper.sync_begin()
            print("  [trainer] weight sync START — keeper heartbeating...")
            await asyncio.sleep(PAUSE)
            keeper.sync_end()
            print("  [trainer] weight sync END")
            # No force_recycle in real mode — only recover if the real session actually died.
            try:
                code = await run_check(sb)
                print(f"  no reclamation occurred at PAUSE={PAUSE:.0f}s (check exit={code}); "
                      "recovery is exercised only on a genuine session-death (PAUSE > idle_timeout,")
                print("  or a real instance recycle). We do NOT fabricate a failure.")
            except Exception as exc:  # noqa: BLE001
                print(f"  resume FAILED: {exc}")
                print(f"  recoverable? {is_recoverable(exc)} -> recover by re-seeding on a fresh session")
                # recovery.py's contract: restart on a fresh session + re-write the workspace.
                async with _make_inner() as fsb:
                    await seed(fsb)   # re-seed: no snapshot to restore, so re-write the workspace
                    print(f"  >> RECOVERED: re-seeded {fsb.sandbox_id}, check exit={await run_check(fsb)} "
                          f"— episode continues instead of crashing the run")


async def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    if REAL:
        print(f"heartbeat_demo — IDLE={IDLE}s PAUSE={PAUSE}s TICK={TICK}s "
              f"(REAL AgentCore sandbox: {_SANDBOX or 'agentcore'})")
        await act1_naive_real()
        await act2_heartbeat_real()
        await act3_recover_real()
    else:
        print(f"heartbeat_demo — IDLE={IDLE}s PAUSE={PAUSE}s TICK={TICK}s (offline LocalSandbox)")
        await act1_naive()
        await act2_heartbeat()
        await act3_recover()
    banner("takeaway: heartbeat (SessionKeeper, outside the harness) survives the weight-sync\n"
           "pause; recovery (re-seed) backstops a real loss. No snapshot needed — the pause is\n"
           "seconds, never past max-lifetime. See docs/gpu-snapshot.md for why EC2 snapshots don't apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
