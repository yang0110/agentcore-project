# demo_gpu_snapshot — application-level snapshot / resume across a GPU instance recycle

A **standalone, self-contained** demo of the only feasible "GPU instance snapshot" on AgentCore:
an **application-level** one. It serializes the sandbox workspace to a store you own and restores it
onto a fresh session after a recycle, so accumulated work survives. Runs in a second, offline, no
AWS and no GPU.

> **There is no VM/EC2 snapshot here — because AgentCore doesn't offer one.** You don't own the
> managed EC2, an EBS snapshot is disk-only (not the paused VRAM/CUDA context), and a recycled GPU
> instance comes back blank. See [`docs/gpu-snapshot.md`](../../../docs/gpu-snapshot.md) for the full
> reasoning (the three walls, and why process/GPU checkpointing via cuda-checkpoint+CRIU is tier-(b)
> heavy/fragile and unnecessary for RL). This demo is **tier (a)**: serialize workspace state.

## What it shows
The task accumulates progress across steps (`progress.json` + `work.log`). Midway, a GPU instance
recycle destroys the session. Two runs, side by side:

- **RUN A — no snapshot.** The recycle loses all progress; the fresh session restarts from step 0 and
  redoes everything → **wasted rework**.
- **RUN B — application-level snapshot.** Snapshot the workspace before the recycle, restore onto the
  fresh session → **resume from where it stopped**, no wasted work.

`python3 snapshot_demo.py` prints both and the rework saved (e.g. "6 vs 4 steps → 2 saved").

## The mechanism (reusable)
`agentcore_slime_demo/snapshot_store.py`:
- **`snapshot_workspace(sb, paths, store, key)`** — read a listed set of files off any `Sandbox` and
  persist them as a manifest `{path: text}`.
- **`restore_workspace(sb, store, key)`** — write a saved manifest back into a fresh sandbox.
- **`LocalSnapshotStore`** — JSON files under a host dir (offline/CI).
- **`S3SnapshotStore`** — boto3 put/get; the recommended production store: **no mount, AZ-agnostic,
  VPC-optional**. FSx/EFS would need a mount AgentCore won't hand you (see `docs/gpu-snapshot.md`
  "S3 vs FSx").

Backend-agnostic: the same calls work for `LocalSandbox`, `AgentCoreSandbox` (Code Interpreter), or
the GPU Runtime sandbox. The recycle is modelled by `agentcore_idle_sim.force_recycle` for the
offline run.

## Run it
```bash
python3 snapshot_demo.py                       # TARGET=4 steps, recycle after step 2
TARGET=6 RECYCLE_AFTER=3 python3 snapshot_demo.py
bash run.sh                                    # same, via the convenience runner
```

## Run against real AgentCore
- Swap `ReclaimableLocalSandbox` for an `AgentCoreSandbox` / `AgentCoreRuntimeSandbox`, and
  `LocalSnapshotStore` for `S3SnapshotStore(bucket=..., prefix=...)` (set `SNAPSHOT_S3_BUCKET`).
- Snapshot on the ops you choose (episode boundary, or a `{"op":"quiesce"}` you add to
  `sandbox_server_agent.py`); restore on the first invoke after a recycle.
- The recycle is then a real AgentCore instance recycle / session loss; recovery (`recovery.py`)
  detects it and the restore re-seeds the workspace.

## When to use this vs the heartbeat
- **Weight-sync pause (seconds):** DON'T snapshot — keep the session warm with the heartbeat
  ([`demo_cpu_quiesce`](../demo_cpu_quiesce/)). Snapshot/restore costs more than the pause.
- **Genuine session loss / instance recycle:** snapshot the workspace so recovery resumes with
  progress instead of from scratch — this demo.

## Folder layout (self-contained, layout B)
- `snapshot_demo.py` — the standalone A-vs-B driver ("the run")
- `agentcore_idle_sim.py` — offline instance-recycle / idle simulator (shared with demo_cpu_quiesce)
- `agentcore_slime_demo/` — vendored shared library (adds `snapshot_store.py`, `session_keeper.py`)
- `deploy/Dockerfile.sandbox.gpu` — the GPU sandbox image (for the real-AgentCore variant)
- `run.sh` — convenience runner

References (not copied): `docs/gpu-snapshot.md`, `docs/demos-program.md`, and the shared
`agentcore_slime_demo/recovery.py` contract.
