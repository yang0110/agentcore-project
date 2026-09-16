#!/usr/bin/env python3
"""demo_gpu_snapshot — application-level snapshot / resume across a GPU instance recycle (standalone).

There is NO real GPU-instance snapshot on AgentCore (you don't own the managed EC2; an EBS snapshot
is disk-only, not the paused VRAM/CUDA context; a recycled GPU instance comes back blank — see
docs/gpu-snapshot.md). The ONLY feasible "snapshot" is application-level: serialize the sandbox
WORKSPACE to a store you own, and restore it onto a fresh session after a recycle. This demo shows
exactly that, end-to-end, in a second, with NO AWS and NO GPU.

The task accumulates progress across steps (progress.json + work.log). Midway, a GPU instance
recycle destroys the session. Two runs, side by side:

  RUN A — no snapshot      : the recycle loses all progress -> redo every step from scratch (wasted).
  RUN B — app-level snapshot: snapshot the workspace before the recycle, restore on a fresh session
                              -> resume from where it stopped (no wasted work).

Snapshot/restore is `agentcore_slime_demo/snapshot_store.py` (LocalSnapshotStore here; S3SnapshotStore
in production — no mount, AZ-agnostic; docs/gpu-snapshot.md "S3 vs FSx"). The recycle is modelled by
`agentcore_idle_sim.force_recycle` so the demo runs offline; against real AgentCore the store is S3
and the recycle is a real instance recycle.

    python3 snapshot_demo.py            # TARGET=4 steps, recycle after step 2
    TARGET=6 RECYCLE_AFTER=3 python3 snapshot_demo.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile

from agentcore_slime_demo.snapshot_store import (
    LocalSnapshotStore,
    restore_workspace,
    snapshot_workspace,
)

from agentcore_idle_sim import AgentCoreIdleSim, ReclaimableLocalSandbox

TARGET = int(os.environ.get("TARGET", "4"))            # steps needed to finish the task
RECYCLE_AFTER = int(os.environ.get("RECYCLE_AFTER", "2"))  # instance recycle after this many steps
SNAPSHOT_PATHS = ["progress.json", "work.log"]         # the workspace state worth preserving


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


async def progress(sb) -> int:
    try:
        return json.loads(await sb.read_file("progress.json")).get("steps_done", 0)
    except Exception:  # noqa: BLE001 — fresh workspace has no progress file yet
        return 0


async def do_step(sb) -> int:
    """One unit of agent work: bump the counter and append to the log (cumulative state)."""
    n = await progress(sb) + 1
    await sb.write_file("progress.json", json.dumps({"steps_done": n}))
    try:
        log = await sb.read_file("work.log")
    except Exception:  # noqa: BLE001
        log = ""
    await sb.write_file("work.log", log + f"step {n} done\n")
    return n


def fresh_sandbox(sim: AgentCoreIdleSim) -> ReclaimableLocalSandbox:
    sb = ReclaimableLocalSandbox(workdir="/workspace")
    sim.watch(sb)
    return sb


async def run(*, use_snapshot: bool) -> int:
    """Drive the task to completion across ONE mid-run recycle. Returns total steps EXECUTED
    (TARGET is the useful amount; anything above TARGET is wasted rework)."""
    # idle_timeout huge: we only trigger the recycle explicitly, not via idle here.
    sim = AgentCoreIdleSim(idle_timeout=1e9)
    store = LocalSnapshotStore(root=tempfile.mkdtemp(prefix="snap-"))
    key = "episode-42"
    executed = 0

    sb = fresh_sandbox(sim)
    async with sb:
        # --- work until the recycle point --------------------------------------------------
        while await progress(sb) < RECYCLE_AFTER:
            executed += 1
            n = await do_step(sb)
            print(f"  step {n} on {sb.sandbox_id}")
        if use_snapshot:
            manifest = await snapshot_workspace(sb, SNAPSHOT_PATHS, store, key)
            print(f"  snapshot: saved {list(manifest)} at step {await progress(sb)} -> store")
        # --- the GPU instance recycle destroys this session --------------------------------
        sim.force_recycle(sb, reason="GPU instance recycle")
        print(f"  !! instance recycled — {sb.sandbox_id} is gone (no VM snapshot)")

    # --- fresh session after the recycle ---------------------------------------------------
    sb2 = fresh_sandbox(sim)
    async with sb2:
        if use_snapshot:
            restored = await restore_workspace(sb2, store, key)
            print(f"  restore: wrote {restored} onto fresh {sb2.sandbox_id} "
                  f"-> resuming at step {await progress(sb2)}")
        else:
            print(f"  no snapshot: fresh {sb2.sandbox_id} starts at step {await progress(sb2)} "
                  f"(all prior work lost)")
        while await progress(sb2) < TARGET:
            executed += 1
            n = await do_step(sb2)
            print(f"  step {n} on {sb2.sandbox_id}")
        print(f"  DONE: {await progress(sb2)}/{TARGET} steps; executed {executed} total")
    return executed


async def main() -> int:
    print(f"demo_gpu_snapshot — TARGET={TARGET} steps, recycle after {RECYCLE_AFTER} "
          f"(offline LocalSandbox; store=LocalSnapshotStore)")

    banner("RUN A — no snapshot: recycle loses progress, redo from scratch")
    a = await run(use_snapshot=False)

    banner("RUN B — application-level snapshot: restore on a fresh session, resume")
    b = await run(use_snapshot=True)

    wasted = a - b
    banner(f"result: no-snapshot executed {a} steps; app-snapshot executed {b} steps "
           f"({wasted} rework step(s) saved).\n"
           "This is application-level state (S3 in prod) — NOT a VM/EC2 snapshot, which AgentCore\n"
           "does not offer. See docs/gpu-snapshot.md. For a seconds-long weight-sync pause, prefer\n"
           "the heartbeat (demo_cpu_quiesce); snapshot the workspace only to survive a real recycle.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
