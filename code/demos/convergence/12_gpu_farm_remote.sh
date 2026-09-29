#!/bin/bash
# ============================================================================
# CONVERGENCE RUN 12 — demo_gpu_farm LOCAL_GPU=0  (KernelBench, REMOTE g6; was demo_gpu_single)
# ============================================================================
# TASK / DATASET : KernelBench (ScalingIntelligence/KernelBench). The policy writes
#                  a custom CUDA `ModelNew`; the checker COMPILES and RUNS it on GPU
#                  and scores torch.allclose vs the reference. Train = level_1 (100
#                  single-op problems); held-out eval = disjoint level_2 fusions.
# SUBSTRATE      : ONE warm REMOTE g6 custom GPU Runtime shared by all episodes
#                  (SANDBOX=agentcore-ray-farm, per-episode sub_session; FARM=fractional = the
#                  L4 split 0.5+0.5, 2 slots; FARM=whole = the whole L4, 1 slot). The on-box
#                  farm is 13/14 (demo_gpu_farm LOCAL_GPU=1).
# BOTTLENECK     : nvcc COMPILE + kernel run in the remote sandbox is the long pole
#                  (seconds/episode), and one g6 serializes them (1-2 slots). This is
#                  the "sandbox IS the bottleneck" demo — contrast with the farm (13/14).
# CONVERGENCE    : eval/kernelbench mean reward (compile-pass AND allclose) rises then
#                  plateaus. Early reward is often ~0 until the policy learns to emit
#                  a COMPILABLE kernel; watch compile-pass rate climb first.
# REWARD         : RLVR (compiles & correct = 1.0 else 0.0); GRPO group-relative.
#                  Save: async + persistent ckpt worker (remote path).
# PREREQS        : run INSIDE agentcore_p4. Export AGENTCORE_SANDBOX_RUNTIME_ARN (a live g6 CUDA
#                  sandbox Runtime: LOCAL_GPU=0 FARM=$FARM bash demo_gpu_farm/4_deploy_runtime.sh).
# RUNTIME/CKPT   : ~100 steps (GPU is slow/costly => NR=101 default); ckpt ~56 GB.
# ----------------------------------------------------------------------------
# Overridable, e.g.:  NR=201 FARM=whole bash 12_gpu_farm_remote.sh
# ============================================================================
set -euo pipefail

CONV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO="$(cd "$CONV_DIR/.." && pwd)/demo_gpu_farm"
export LOCAL_GPU=0 FARM="${FARM:-fractional}"

if [ -z "${AGENTCORE_SANDBOX_RUNTIME_ARN:-}" ]; then
  echo "ERROR: 12 needs the remote g6 GPU sandbox. Export AGENTCORE_SANDBOX_RUNTIME_ARN to a live g6" >&2
  echo "       CUDA sandbox Runtime (LOCAL_GPU=0 FARM=$FARM bash code/demos/demo_gpu_farm/4_deploy_runtime.sh)." >&2
  echo "       For the on-box farm use 13_gpu_farm_fractional.sh / 14_gpu_farm_whole.sh." >&2
  exit 1
fi

: "${NR:=101}"           # real train steps = NR-1 = 100 (GPU runs are expensive)
: "${RB:=8}"; : "${NS:=8}"; : "${TEMP:=1.0}"
: "${EVAL_INTERVAL:=20}"; : "${EVAL_NS:=4}"; : "${SAVE_INTERVAL:=100}"
: "${CONC:=}"            # empty => run.sh uses the farm's slot count 
export NR RB NS TEMP EVAL_INTERVAL EVAL_NS SAVE_INTERVAL
[ -n "$CONC" ] && export CONC

LOG="$DEMO/train.convergence_$(date +%Y%m%d_%H%M%S).log"
echo ">>> converging demo_gpu_farm LOCAL_GPU=0 FARM=$FARM (KernelBench, remote g6)  NR=$NR RB=$RB NS=$NS steps=$((NR-1))  log=$LOG"
bash "$DEMO/run.sh" 2>&1 | tee "$LOG"
