#!/bin/bash
# ============================================================================
# CONVERGENCE RUN 18 — demo_gpu_snapshot  (KernelBench, REMOTE g6, app-level snapshot/resume)
# ============================================================================
# TASK / DATASET : KernelBench (same as 12/13/14): write a custom CUDA ModelNew, the
#                  checker COMPILES + RUNS it on GPU and scores torch.allclose vs the
#                  reference. Train = level_1 (100 single-op problems); held-out eval =
#                  disjoint level_2 fusions.
# SUBSTRATE      : a REMOTE g6 CUDA sandbox Runtime (SANDBOX=agentcore-runtime). REMOTE-ONLY
#                  — there is no on-box LOCAL_GPU path for this demo (unlike 19).
# PROPERTY (LIVE): APP-LEVEL SNAPSHOT / RESUME of the sandbox workspace at each episode
#                  boundary. With SNAPSHOT_EVERY=1 the demo tars the workspace files
#                  (SNAPSHOT_PATHS: progress.json, work.log, solution.py) into a
#                  LocalSnapshotStore (SNAPSHOT_DIR) after each episode and restores them
#                  into the next session — the application-level stand-in for a real
#                  GPU-instance disk snapshot (which AgentCore does not expose; see
#                  docs/gpu-snapshot.md). SNAPSHOT_EVERY=0 => plain demo_gpu_single.
#                  Over ~100 steps the snapshot/restore round-trip runs every episode.
# BOTTLENECK     : nvcc compile + kernel run in the remote g6 (seconds/episode); the
#                  tar/restore is small and off the critical path.
# CONVERGENCE    : eval/kernelbench mean reward (compile-pass AND allclose) rises then
#                  plateaus; watch compile-pass rate climb first.
# REWARD         : RLVR (compiles & correct = 1.0 else 0.0); GRPO. SYNC save (this
#                  sandbox image's seccomp blocks async-ckpt pidfd_getfd).
# PREREQS        : run INSIDE agentcore_p4. REMOTE-ONLY — you MUST export
#                  AGENTCORE_SANDBOX_RUNTIME_ARN pointing at a live g6 CUDA sandbox Runtime.
# RUNTIME/CKPT   : ~100 steps (NR=101 default); ckpt ~56 GB (disk caveat).
# STATUS         : end-to-end on a live g6 is UNVERIFIED on hardware (offline-validated only).
# ----------------------------------------------------------------------------
# Overridable, e.g.:  NR=201 SNAPSHOT_EVERY=1 bash 18_gpu_snapshot.sh
#   SNAPSHOT_EVERY=0  -> disable snapshot/resume (== demo_gpu_single on KernelBench)
#   SNAPSHOT_S3_BUCKET=<bucket>  -> back the snapshot store with S3 instead of local disk
# ============================================================================
set -euo pipefail

CONV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO="$(cd "$CONV_DIR/.." && pwd)/demo_gpu_snapshot"

if [ -z "${AGENTCORE_SANDBOX_RUNTIME_ARN:-}" ]; then
  echo "ERROR: demo_gpu_snapshot is REMOTE-ONLY — export AGENTCORE_SANDBOX_RUNTIME_ARN" >&2
  echo "       to a live g6 CUDA sandbox Runtime ARN. There is no on-box LOCAL_GPU path" >&2
  echo "       for this demo; for an on-box GPU run use 14_gpu_farm_whole.sh or 19_gpu_customruntime.sh." >&2
  exit 1
fi

: "${NR:=101}"           # real train steps = NR-1 = 100 (GPU runs are expensive)
: "${RB:=8}"; : "${NS:=8}"; : "${TEMP:=1.0}"
: "${EVAL_INTERVAL:=20}"; : "${EVAL_NS:=4}"; : "${SAVE_INTERVAL:=100}"
: "${CONC:=4}"           # GPU demos are stock 1-step-async (no MODE knob)
: "${SNAPSHOT_EVERY:=1}" # snapshot/resume the workspace every episode (the demo's point)
export NR RB NS TEMP EVAL_INTERVAL EVAL_NS SAVE_INTERVAL CONC SNAPSHOT_EVERY

LOG="$DEMO/train.convergence_$(date +%Y%m%d_%H%M%S).log"
echo ">>> converging demo_gpu_snapshot (KernelBench, remote g6, SNAPSHOT_EVERY=$SNAPSHOT_EVERY)  NR=$NR RB=$RB NS=$NS steps=$((NR-1))  log=$LOG"
bash "$DEMO/run.sh" 2>&1 | tee "$LOG"
