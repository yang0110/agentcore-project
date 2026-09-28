"""Application-level sandbox-workspace snapshot glue for the REAL RL trainer.

`snapshot_store.py` provides the primitives (`snapshot_workspace` / `restore_workspace` over a
`LocalSnapshotStore` or `S3SnapshotStore`); `snapshot_demo.py` drives them in an offline animation.
THIS module wires the same primitives into the live `generate.generate` rollout so a real slime
training run preserves each task's sandbox workspace across the session churn of the run.

Why this is the "GPU instance snapshot": AgentCore exposes NO VM/EC2 snapshot (you don't own the
managed instance; an EBS snapshot is disk-only, not the paused VRAM/CUDA context; a recycled GPU
instance comes back blank — see docs/gpu-snapshot.md). The only feasible snapshot is
application-level: serialize the WORKSPACE to a store you own and restore it onto a fresh session.

Where it fires relative to weight sync: in async/decoupled slime every episode opens its OWN fresh
sandbox session (a fresh `runtimeSessionId`) — so from the workspace's point of view **every episode
boundary is already a recycle**. Snapshotting at episode EXIT (the quiescent point, after the
multi-turn loop finished and before the trainer's weight-sync barrier) and restoring at the next
same-key episode's ENTER is exactly "survive a recycle": the store, not the session, is the durable
source of truth. `RECYCLE_AFTER>0` additionally logs an explicit simulated instance recycle so the
resume is legible in the run log.

Gating: `SNAPSHOT_EVERY=0` disables everything → `generate` behaves EXACTLY like the plain demo_gpu_farm run.
Env knobs (all optional; defaults chosen so a smoke is safe):
  * SNAPSHOT_EVERY   (int, default 0; demo_gpu_farm run.sh: SNAPSHOT_EVERY=1 turns it on)  — snapshot cadence in episodes; 0 = off.
  * SNAPSHOT_DIR     (path)            — LocalSnapshotStore root (ignored if SNAPSHOT_S3_BUCKET set).
  * SNAPSHOT_S3_BUCKET (str)           — use S3SnapshotStore instead (production; no mount, AZ-agnostic).
  * SNAPSHOT_PATHS   (csv, default "progress.json,work.log,solution.py")
                                        — workspace files to preserve; missing files are skipped.
  * RECYCLE_AFTER    (int, default 0)  — after N episodes, log a simulated instance recycle.

Best-effort by construction: a snapshot/restore failure must never fail a training rollout (idle
reclaim / re-seed is the fallback), so every op here is wrapped and swallowed.
"""

from __future__ import annotations

import itertools
import os
import sys
import threading

from .snapshot_store import (
    LocalSnapshotStore,
    S3SnapshotStore,
    restore_workspace,
    snapshot_workspace,
)

# Process-wide episode counter (one rollout worker process; asyncio is single-threaded per loop, but
# guard anyway so a threaded eval path can't race the count).
_counter = itertools.count(1)
_lock = threading.Lock()
_recycle_logged = False


def enabled() -> bool:
    """Snapshotting is on unless SNAPSHOT_EVERY == 0."""
    return _every() > 0


def _every() -> int:
    try:
        return int(os.environ.get("SNAPSHOT_EVERY", "0"))
    except ValueError:
        return 1


def _paths() -> list[str]:
    raw = os.environ.get("SNAPSHOT_PATHS", "progress.json,work.log,solution.py")
    return [p.strip() for p in raw.split(",") if p.strip()]


def _make_store():
    """The store we OWN the snapshot in — S3 in production (no mount, AZ-agnostic), else a host dir."""
    if os.environ.get("SNAPSHOT_S3_BUCKET"):
        return S3SnapshotStore()
    return LocalSnapshotStore(root=os.environ.get("SNAPSHOT_DIR") or "/tmp/agentcore_snapshots")


def _log(msg: str) -> None:
    sys.stdout.write(f"[app-snapshot] {msg}\n")
    sys.stdout.flush()


async def maybe_restore(sb, key: str) -> list[str]:
    """At episode ENTER: restore this task's workspace from the store onto the fresh session.

    This is the resume path — a fresh session starts blank, so anything a prior episode of the same
    task saved is written back here. Returns the restored relative paths (empty if none / disabled)."""
    if not enabled():
        return []
    try:
        restored = await restore_workspace(sb, _make_store(), key)
        if restored:
            _log(f"restore key={key} -> {restored} onto {getattr(sb, 'sandbox_id', '?')}")
        # DEMO-SPECIFIC metric ([demo] channel): files restored on this resume. A restore>0 IS the
        # rework saved — those files would otherwise be recomputed from scratch after a recycle.
        if restored:
            print(f"[demo] snapshot_restored_files={len(restored)} key={key}", flush=True)
        return restored
    except Exception as exc:  # noqa: BLE001 — best-effort; a fresh workspace / store miss is fine
        _log(f"restore key={key} skipped ({type(exc).__name__})")
        return []


async def maybe_snapshot(sb, key: str) -> list[str]:
    """At episode EXIT (quiescent, before the weight-sync barrier): persist the workspace.

    Fires every SNAPSHOT_EVERY-th episode. Also emits the simulated-recycle marker once, after
    RECYCLE_AFTER episodes, so the next same-key restore in the log reads as a genuine resume."""
    if not enabled():
        return []
    with _lock:
        n = next(_counter)
    global _recycle_logged
    recycle_after = int(os.environ.get("RECYCLE_AFTER", "0") or "0")
    if recycle_after > 0 and n >= recycle_after and not _recycle_logged:
        _recycle_logged = True
        _log(f"!! simulated GPU-instance recycle after {recycle_after} episodes — live sessions are "
             f"gone; subsequent episodes must restore each task's workspace from the store")
    if n % _every() != 0:
        return []
    try:
        manifest = await snapshot_workspace(sb, _paths(), _make_store(), key)
        saved = list(manifest)
        if saved:
            _log(f"snapshot key={key} <- {saved} (episode {n})")
            # DEMO-SPECIFIC metric ([demo] channel): files persisted at this quiescent episode-exit.
            print(f"[demo] snapshot_saved_files={len(saved)} episode={n} key={key}", flush=True)
        return saved
    except Exception as exc:  # noqa: BLE001 — teardown-adjacent; never fail a rollout on snapshot
        _log(f"snapshot key={key} skipped ({type(exc).__name__})")
        return []
