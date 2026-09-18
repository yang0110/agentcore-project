# demo_gpu_snapshot — REAL KernelBench GPU trainer with application-level snapshot / resume

A **self-contained** slime GPU trainer (same decoupled/async KernelBench shape as `demo_gpu_single`:
policy writes `solution.py`, a torch/CUDA checker runs it on a **remote g6 sandbox Runtime #2**,
multi-turn repair in-process) that additionally **snapshots each task's sandbox workspace to a store
you own and restores it onto a fresh session** — the only feasible "GPU instance snapshot" on
AgentCore. It does a real 10-step run (`NR=11` ⇒ 10 train + 1 eval, single final checkpoint).

> **There is no VM/EC2 snapshot here — because AgentCore doesn't offer one.** You don't own the
> managed EC2; an EBS snapshot is disk-only (not the paused VRAM/CUDA context); a recycled GPU
> instance comes back blank. On the microVM substrate there is no persistent disk at all. See
> [`docs/gpu-snapshot.md`](../../../docs/gpu-snapshot.md) for the full reasoning (the three walls, and
> why process/GPU checkpointing via cuda-checkpoint+CRIU is tier-(b) heavy/fragile and unnecessary for
> RL). This demo is **tier (a)**: serialize *application* workspace state.

> ⚠️ **Hardware status.** The snapshot glue is offline-validated (import + `ast` + the offline
> `snapshot_demo.py` animation). The **end-to-end REAL trainer path** (live g6 sandbox + store
> round-trip during a training run) is **UNVERIFIED on hardware**.

## How the snapshot rides the real run

In async/decoupled slime **every episode opens its own fresh sandbox session** (a fresh
`runtimeSessionId`), so from the workspace's point of view **every episode boundary is already a
recycle**. The trainer turns that into snapshot/resume with two hooks in the live rollout
(`agentcore_slime_demo/generate.py`, gated by `SNAPSHOT_EVERY`):

1. **Restore at episode ENTER.** `maybe_restore(sb, "ep-<task_id>")` writes any previously-saved
   workspace for this task back onto the fresh session — the **resume** path. The store, not the
   session, is the durable source of truth.
2. **Snapshot at episode EXIT.** `maybe_snapshot(sb, "ep-<task_id>")` reads the workspace files back
   out to the store. Exit is the **quiescent point** — the multi-turn loop has finished and it is
   right before the trainer's **weight-sync barrier**, so the snapshot is consistent (no torn write).

The primitives are the existing `agentcore_slime_demo/snapshot_store.py`
(`snapshot_workspace`/`restore_workspace` over a `LocalSnapshotStore` or, in production, an
`S3SnapshotStore` — no mount, AZ-agnostic); the trainer glue is `agentcore_slime_demo/app_snapshot.py`.
`RECYCLE_AFTER>0` logs an explicit **simulated instance recycle** after N episodes so the subsequent
restore reads as a genuine resume in the run log. Everything is **best-effort**: a snapshot/restore
failure never fails a rollout (re-seed / idle reclaim is the fallback).

`SNAPSHOT_EVERY=0` disables both hooks ⇒ the run is **identical to `demo_gpu_single`**.

## Reproduce (inside the `agentcore_p4` container)

```bash
# data ships in data/ (copied from demo_gpu_single: 100 level_1 train + 20 level_2 eval); no prepare step.
export G6_ARN='arn:aws:bedrock-agentcore:us-west-2:ACCT:runtime/acs_sandbox_gpu-XXXX'
docker exec -e NR=11 -e EVAL_INTERVAL=10 -e EVAL_NS=2 \
    -e AGENTCORE_SANDBOX_RUNTIME_ARN="$G6_ARN" \
    -e SNAPSHOT_EVERY=1 -e RECYCLE_AFTER=6 agentcore_p4 \
    bash -lc 'cd /work && bash code/demos/demo_gpu_snapshot/run.sh 2>&1 | tee /work/code/demos/_smoke_logs/demo_gpu_snapshot.log'
docker exec agentcore_p4 python3 /work/code/demos/demo_gpu_snapshot/metrics.py \
    /work/code/demos/_smoke_logs/demo_gpu_snapshot.log \
    --csv /work/code/demos/_smoke_logs/demo_gpu_snapshot.csv \
    --eval-csv /work/code/demos/_smoke_logs/demo_gpu_snapshot.eval.csv \
    --plot /work/code/demos/_smoke_logs/demo_gpu_snapshot.png
```

Production store: set `SNAPSHOT_S3_BUCKET=<bucket>` to swap `LocalSnapshotStore` for `S3SnapshotStore`
(no mount, works from any AZ). `AGENTCORE_SANDBOX_RUNTIME_ARN` is mandatory (hard error if unset).

## Knobs (env; defaults in `run.sh`)

| Knob | Default | Meaning |
|---|---|---|
| `SNAPSHOT_EVERY` | `1` | snapshot cadence in episodes; **`0` disables** ⇒ behaves as `demo_gpu_single` |
| `SNAPSHOT_DIR` | `$DEMO_DIR/snapshots` | `LocalSnapshotStore` root (ignored if `SNAPSHOT_S3_BUCKET` set) |
| `SNAPSHOT_S3_BUCKET` | *(unset)* | use `S3SnapshotStore` (production; no mount, AZ-agnostic) |
| `SNAPSHOT_PATHS` | `progress.json,work.log,solution.py` | workspace files to preserve (missing skipped) |
| `RECYCLE_AFTER` | `0` | `>0` ⇒ log a simulated GPU-instance recycle after N episodes |
| `NR` / `RB` / `NS` | 11 / 4 / 8 | 10 train + 1 eval; GRPO group; keep `NR×RB ≤ 100` |
| `EVAL_INTERVAL` / `EVAL_NS` | 10 / 2 | held-out eval cadence / samples |
| `SANDBOX` | `agentcore-runtime` | remote g6 sandbox Runtime #2 (the KernelBench GPU checker) |

## What to watch

The `[app-snapshot]` log lines: `snapshot key=ep-<task> <- [...]` at exits and `restore key=ep-<task>
-> [...]` at the entries after a recycle — a restore that carries the pre-recycle files onto a fresh
`sandbox_id` is the resume working. Plus the usual trainer signals: `rollout/raw_reward`,
`eval/kernelbench`, and the `perf/sandbox_ms_*` vs `perf/sglang_gen_ms_*` split.

## Offline animation (no AWS, no GPU)

`snapshot_demo.py` plays the naive→snapshot arc end-to-end in a second: two runs (no-snapshot vs
app-snapshot) across one mid-run recycle, printing the rework steps saved. It is the legible
illustration of the exact mechanism the trainer wires in.

```bash
python3 snapshot_demo.py                    # OFFLINE sim: TARGET=4 steps, recycle after step 2
REAL_SNAPSHOT=1 LOCAL_GPU=1 SANDBOX=agentcore-runtime \
    AGENTCORE_LOCAL_ENDPOINT=http://127.0.0.1:8080 python3 snapshot_demo.py   # against a real session
```

## Folder layout (self-contained)

- `agentcore_slime_demo/` — vendored library (in-process harness + remote GPU sandbox client), plus
  `snapshot_store.py` (the store primitives) and `app_snapshot.py` (the live-rollout glue).
- `run.sh` — the real KernelBench GPU trainer with snapshot/resume · `metrics.py` — log→CSV/PNG.
- `snapshot_demo.py` + `agentcore_idle_sim.py` — the offline animation of the same mechanism.
- `data/` — `kernelbench_{train,eval}.jsonl` (100 level_1 train + 20 level_2 eval).
- `snapshots/` — default `LocalSnapshotStore` root for the workspace snapshots.
