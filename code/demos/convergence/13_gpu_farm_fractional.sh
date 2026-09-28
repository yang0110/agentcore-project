#!/bin/bash
# ============================================================================
# CONVERGENCE RUN 13 — demo_gpu_farm FARM=fractional  (KernelBench, LOCAL farm, ONE GPU 0.5+0.5)
# ============================================================================
# TASK / DATASET : KernelBench (same as 12/14): write CUDA ModelNew, checker compiles +
#                  runs on GPU + allclose. Train = level_1 (100); eval = level_2 (20).
# SUBSTRATE      : the SAME Ray farm as 14 (demo_gpu_farm FARM=whole), served
#                  LOCALLY on THIS box's reserved GPU 6 (SANDBOX=agentcore-ray-farm,
#                  endpoint http://127.0.0.1:8080), with B4_FARM_GPUS=1, B4_GPU_FRACTION=0.5
#                  => 2 FRACTIONAL sub-session slots SHARING one GPU (0.5 + 0.5). slime
#                  trains on GPUs 0-5 (4 train + 2 infer); GPU 7 idle. One shared
#                  runtimeSessionId + per-episode sub_session (1 session = 1 instance).
# BOTTLENECK     : fractional sharing on a non-MIG L40S can WEDGE under sustained nvcc
#                  load; each actor caps its VRAM in software (B4_PER_SANDBOX_MEM_FRACTION
#                  =0.5, baked on the host container). This is the intended GPU-SHARING
#                  stress case (contrast: 14_gpu_farm_whole = whole GPU per session).
# CONVERGENCE    : eval/kernelbench mean reward rises then plateaus (compile-pass first).
# REWARD         : RLVR (compiles & correct = 1.0 else 0.0); GRPO. SYNC save.
# PREREQS (TWO-STEP LAUNCH):
#   1) ON THE HOST (needs docker + nvidia runtime, NOT inside agentcore_p4):
#        bash code/deploy/run_local_gpu_runtime.sh fractional   # reserves GPU 6, serves :8080
#   2) INSIDE agentcore_p4:
#        bash code/demos/convergence/13_gpu_farm_fractional.sh
#   This script PREFLIGHTS the local farm's /ping and refuses to start if it is down.
# TEARDOWN       : bash code/deploy/run_local_gpu_runtime.sh stop   # frees GPU 6
# RUNTIME/CKPT   : ~100 steps (NR=101 default); ckpt ~56 GB.
# ----------------------------------------------------------------------------
# Overridable, e.g.:  NR=201 bash 13_gpu_farm_fractional.sh
# (B4_FARM_GPUS/B4_GPU_FRACTION/CONC below MUST match the host container's farm knobs.)
# ============================================================================
set -euo pipefail

CONV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO="$(cd "$CONV_DIR/.." && pwd)/demo_gpu_farm"
export FARM=fractional

ENDPOINT="${AGENTCORE_LOCAL_ENDPOINT:-http://127.0.0.1:8080}"
if ! curl -fsS "${ENDPOINT}/ping" >/dev/null 2>&1; then
  echo "ERROR: local GPU farm not reachable at ${ENDPOINT}/ping." >&2
  echo "       Start it FIRST on the HOST (not inside agentcore_p4):" >&2
  echo "         bash code/deploy/run_local_gpu_runtime.sh fractional" >&2
  echo "       (first boot pip-installs the CUDA torch stack — can take ~2-3 min)." >&2
  exit 1
fi

: "${NR:=101}"           # real train steps = NR-1 = 100
: "${RB:=8}"; : "${NS:=8}"; : "${TEMP:=1.0}"
: "${EVAL_INTERVAL:=20}"; : "${EVAL_NS:=4}"; : "${SAVE_INTERVAL:=100}"
# Fractional single-GPU farm: 1 GPU / 0.5 => 2 slots => CONC=2. These MUST match the host
# container's baked knobs (run_local_gpu_runtime.sh fractional: FARM_GPUS=1, FRACTION=0.5).
: "${B4_FARM_GPUS:=1}"; : "${B4_GPU_FRACTION:=0.5}"; : "${CONC:=2}"
export NR RB NS TEMP EVAL_INTERVAL EVAL_NS SAVE_INTERVAL CONC B4_FARM_GPUS B4_GPU_FRACTION

LOG="$DEMO/train.convergence_$(date +%Y%m%d_%H%M%S).log"
echo ">>> converging demo_gpu_farm FARM=fractional (KernelBench, local fractional farm, GPU 6 0.5+0.5)  NR=$NR RB=$RB NS=$NS steps=$((NR-1)) CONC=$CONC  log=$LOG"
bash "$DEMO/run.sh" 2>&1 | tee "$LOG"
