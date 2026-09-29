#!/bin/bash
# ============================================================================
# CONVERGENCE RUN 18 — demo_gpu_farm SNAPSHOT_EVERY=1  (KernelBench, app-level snapshot/resume; was demo_gpu_snapshot)
# ============================================================================
# TASK / DATASET : KernelBench (same as 12/13/14): write a custom CUDA ModelNew, the
#                  checker COMPILES + RUNS it on GPU and scores torch.allclose vs the
#                  reference. Train = level_1 (100 single-op problems); held-out eval =
#                  disjoint level_2 fusions.
# SUBSTRATE      : the warm REMOTE g6 custom GPU Runtime (as 12; default), or LOCAL_GPU=1 on-box farm.
# PROPERTY (LIVE): APP-LEVEL SNAPSHOT / RESUME of the sandbox workspace at each episode
#                  boundary. With SNAPSHOT_EVERY=1 the demo tars the workspace files
#                  (SNAPSHOT_PATHS: progress.json, work.log, solution.py) into a
#                  LocalSnapshotStore (SNAPSHOT_DIR) after each episode and restores them
#                  into the next session — the application-level stand-in for a real
#                  GPU-instance disk snapshot (which AgentCore does not expose; see
#                  docs/gpu-snapshot.md). SNAPSHOT_EVERY=0 => plain 12_gpu_farm_remote.sh.
#                  Over ~100 steps the snapshot/restore round-trip runs every episode.
# BOTTLENECK     : nvcc compile + kernel run in the remote g6 (seconds/episode); the
#                  tar/restore is small and off the critical path.
# CONVERGENCE    : eval/kernelbench mean reward (compile-pass AND allclose) rises then
#                  plateaus; watch compile-pass rate climb first.
# REWARD         : RLVR (compiles & correct = 1.0 else 0.0); GRPO.
# PREREQS        : run INSIDE agentcore_p4. Export AGENTCORE_SANDBOX_RUNTIME_ARN, or LOCAL_GPU=1 (farm up).
# RUNTIME/CKPT   : ~100 steps (NR=101 default); ckpt ~56 GB (disk caveat).
# STATUS         : end-to-end on a live g6 is UNVERIFIED on hardware (offline-validated only).
# ----------------------------------------------------------------------------
# Overridable, e.g.:  NR=201 SNAPSHOT_EVERY=1 bash 18_gpu_farm_snapshot.sh
#   SNAPSHOT_EVERY=0  -> disable snapshot/resume (== 12_gpu_farm_remote.sh)
#   SNAPSHOT_S3_BUCKET=<bucket>  -> back the snapshot store with S3 instead of local disk
# ============================================================================
set -euo pipefail

CONV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO="$(cd "$CONV_DIR/.." && pwd)/demo_gpu_farm"
export LOCAL_GPU="${LOCAL_GPU:-0}" FARM="${FARM:-fractional}"

if [ "$LOCAL_GPU" = "1" ]; then
  ENDPOINT="${AGENTCORE_LOCAL_ENDPOINT:-http://127.0.0.1:8080}"
  if ! curl -fsS "${ENDPOINT}/ping" >/dev/null 2>&1; then
    echo "ERROR: LOCAL_GPU=1 but the local GPU farm is not reachable at ${ENDPOINT}/ping." >&2
    echo "       Start it FIRST on the HOST: bash code/deploy/run_local_gpu_runtime.sh $FARM" >&2
    exit 1
  fi
elif [ -z "${AGENTCORE_SANDBOX_RUNTIME_ARN:-}" ]; then
  echo "ERROR: 18 needs a GPU sandbox. Either export AGENTCORE_SANDBOX_RUNTIME_ARN to a live g6 CUDA" >&2
  echo "       sandbox Runtime (LOCAL_GPU=0 FARM=$FARM bash code/demos/demo_gpu_farm/4_deploy_runtime.sh)," >&2
  echo "       or set LOCAL_GPU=1 to use the on-box farm." >&2
  exit 1
fi

: "${NR:=101}"           # real train steps = NR-1 = 100 (GPU runs are expensive)
: "${RB:=8}"; : "${NS:=8}"; : "${TEMP:=1.0}"
: "${EVAL_INTERVAL:=20}"; : "${EVAL_NS:=4}"; : "${SAVE_INTERVAL:=100}"
: "${CONC:=}"            # empty => run.sh uses the farm's slot count 
: "${SNAPSHOT_EVERY:=1}" # snapshot/resume the workspace every episode (the demo's point)
export NR RB NS TEMP EVAL_INTERVAL EVAL_NS SAVE_INTERVAL SNAPSHOT_EVERY
[ -n "$CONC" ] && export CONC

LOG="$DEMO/train.convergence_snapshot_$(date +%Y%m%d_%H%M%S).log"
echo ">>> converging demo_gpu_farm LOCAL_GPU=$LOCAL_GPU FARM=$FARM (KernelBench, SNAPSHOT_EVERY=$SNAPSHOT_EVERY)  NR=$NR RB=$RB NS=$NS steps=$((NR-1))  log=$LOG"
bash "$DEMO/run.sh" 2>&1 | tee "$LOG"
