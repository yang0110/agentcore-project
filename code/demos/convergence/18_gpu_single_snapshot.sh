#!/bin/bash
# ============================================================================
# CONVERGENCE RUN 18 — demo_gpu_single SNAPSHOT_EVERY=1  (KernelBench, app-level snapshot/resume; was demo_gpu_snapshot)
# ============================================================================
# TASK / DATASET : KernelBench (same as 12/13/14): write a custom CUDA ModelNew, the
#                  checker COMPILES + RUNS it on GPU and scores torch.allclose vs the
#                  reference. Train = level_1 (100 single-op problems); held-out eval =
#                  disjoint level_2 fusions.
# SUBSTRATE      : the warm REMOTE g6 custom GPU Runtime (as 12), or LOCAL_GPU=1 on-box farm.
# PROPERTY (LIVE): APP-LEVEL SNAPSHOT / RESUME of the sandbox workspace at each episode
#                  boundary. With SNAPSHOT_EVERY=1 the demo tars the workspace files
#                  (SNAPSHOT_PATHS: progress.json, work.log, solution.py) into a
#                  LocalSnapshotStore (SNAPSHOT_DIR) after each episode and restores them
#                  into the next session — the application-level stand-in for a real
#                  GPU-instance disk snapshot (which AgentCore does not expose; see
#                  docs/gpu-snapshot.md). SNAPSHOT_EVERY=0 => plain 12_gpu_single.sh.
#                  Over ~100 steps the snapshot/restore round-trip runs every episode.
# BOTTLENECK     : nvcc compile + kernel run in the remote g6 (seconds/episode); the
#                  tar/restore is small and off the critical path.
# CONVERGENCE    : eval/kernelbench mean reward (compile-pass AND allclose) rises then
#                  plateaus; watch compile-pass rate climb first.
# REWARD         : RLVR (compiles & correct = 1.0 else 0.0); GRPO.
# PREREQS        : run INSIDE agentcore_p4. Export AGENTCORE_SANDBOX_RUNTIME_ARN, or LOCAL_GPU=1.
# RUNTIME/CKPT   : ~100 steps (NR=101 default); ckpt ~56 GB (disk caveat).
# STATUS         : end-to-end on a live g6 is UNVERIFIED on hardware (offline-validated only).
# ----------------------------------------------------------------------------
# Overridable, e.g.:  NR=201 SNAPSHOT_EVERY=1 bash 18_gpu_single_snapshot.sh
#   SNAPSHOT_EVERY=0  -> disable snapshot/resume (== demo_gpu_single on KernelBench)
#   SNAPSHOT_S3_BUCKET=<bucket>  -> back the snapshot store with S3 instead of local disk
# ============================================================================
set -euo pipefail

CONV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO="$(cd "$CONV_DIR/.." && pwd)/demo_gpu_single"

if [ "${LOCAL_GPU:-0}" = "1" ]; then
  ENDPOINT="${AGENTCORE_LOCAL_ENDPOINT:-http://127.0.0.1:8080}"
  if ! curl -fsS "${ENDPOINT}/ping" >/dev/null 2>&1; then
    echo "ERROR: LOCAL_GPU=1 but the local GPU farm is not reachable at ${ENDPOINT}/ping." >&2
    echo "       Start it FIRST on the HOST: bash code/deploy/run_local_gpu_runtime.sh whole" >&2
    exit 1
  fi
elif [ -z "${AGENTCORE_SANDBOX_RUNTIME_ARN:-}" ]; then
  echo "ERROR: demo_gpu_single needs a GPU sandbox. Either export AGENTCORE_SANDBOX_RUNTIME_ARN to a" >&2
  echo "       live g6 CUDA sandbox Runtime (bash code/demos/demo_gpu_single/4_deploy_runtime.sh), or" >&2
  echo "       set LOCAL_GPU=1 to serve it on-box (whole-GPU Ray farm on GPUs 6,7)." >&2
  exit 1
fi

: "${NR:=101}"           # real train steps = NR-1 = 100 (GPU runs are expensive)
: "${RB:=8}"; : "${NS:=8}"; : "${TEMP:=1.0}"
: "${EVAL_INTERVAL:=20}"; : "${EVAL_NS:=4}"; : "${SAVE_INTERVAL:=100}"
: "${CONC:=4}"           # GPU demos are stock 1-step-async (no MODE knob)
: "${SNAPSHOT_EVERY:=1}" # snapshot/resume the workspace every episode (the demo's point)
export NR RB NS TEMP EVAL_INTERVAL EVAL_NS SAVE_INTERVAL CONC SNAPSHOT_EVERY
export LOCAL_GPU="${LOCAL_GPU:-0}"

LOG="$DEMO/train.convergence_snapshot_$(date +%Y%m%d_%H%M%S).log"
echo ">>> converging demo_gpu_single (KernelBench, remote g6, SNAPSHOT_EVERY=$SNAPSHOT_EVERY)  NR=$NR RB=$RB NS=$NS steps=$((NR-1))  log=$LOG"
bash "$DEMO/run.sh" 2>&1 | tee "$LOG"
