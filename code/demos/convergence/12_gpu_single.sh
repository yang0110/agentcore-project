#!/bin/bash
# ============================================================================
# CONVERGENCE RUN 12 — demo_gpu_single  (KernelBench, REMOTE g6, one GPU/session)
# ============================================================================
# TASK / DATASET : KernelBench (ScalingIntelligence/KernelBench). The policy writes
#                  a custom CUDA `ModelNew`; the checker COMPILES and RUNS it on GPU
#                  and scores torch.allclose vs the reference. Train = level_1 (100
#                  single-op problems); held-out eval = disjoint level_2 fusions.
# SUBSTRATE      : ONE warm REMOTE g6 custom GPU Runtime shared by all episodes
#                  (SANDBOX=agentcore-ray-farm, whole GPU per episode), or LOCAL_GPU=1 for
#                  the on-box whole-GPU farm on 6,7 (replaces the former 19_gpu_customruntime.sh).
# BOTTLENECK     : nvcc COMPILE + kernel run in the remote sandbox is the long pole
#                  (seconds/episode), and a single session serializes them. This is
#                  the "sandbox IS the bottleneck" demo — contrast with the farm (13/14).
# CONVERGENCE    : eval/kernelbench mean reward (compile-pass AND allclose) rises then
#                  plateaus. Early reward is often ~0 until the policy learns to emit
#                  a COMPILABLE kernel; watch compile-pass rate climb first.
# REWARD         : RLVR (compiles & correct = 1.0 else 0.0); GRPO group-relative.
#                  Save: async on the remote g6, SYNC under LOCAL_GPU=1 (farm seccomp blocks pidfd_getfd).
# PREREQS        : run INSIDE agentcore_p4. Export AGENTCORE_SANDBOX_RUNTIME_ARN (a live g6
#                  CUDA sandbox Runtime), OR LOCAL_GPU=1 with the host farm started.
# RUNTIME/CKPT   : ~100 steps (GPU is slow/costly => NR=101 default); ckpt ~56 GB.
# ----------------------------------------------------------------------------
# Overridable, e.g.:  NR=201 CONC=8 bash 12_gpu_single.sh
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
export NR RB NS TEMP EVAL_INTERVAL EVAL_NS SAVE_INTERVAL CONC
export LOCAL_GPU="${LOCAL_GPU:-0}"

LOG="$DEMO/train.convergence_$(date +%Y%m%d_%H%M%S).log"
echo ">>> converging demo_gpu_single (KernelBench, $([ "$LOCAL_GPU" = 1 ] && echo on-box farm 6,7 || echo remote g6))  NR=$NR RB=$RB NS=$NS steps=$((NR-1))  log=$LOG"
bash "$DEMO/run.sh" 2>&1 | tee "$LOG"
