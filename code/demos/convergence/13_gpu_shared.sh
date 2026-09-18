#!/bin/bash
# ============================================================================
# CONVERGENCE RUN 13 — demo_gpu_shared  (KernelBench, REMOTE g6, Ray FRACTIONAL farm)
# ============================================================================
# TASK / DATASET : KernelBench (same as 12): write CUDA ModelNew, checker compiles +
#                  runs on GPU + allclose. Train = level_1 (100); eval = level_2 (20).
# SUBSTRATE      : ONE shared REMOTE g6 AgentCore session (B4_SHARED_RUNTIME_SESSION)
#                  running an in-instance Ray farm that packs MANY episodes onto ONE
#                  g6 GPU via FRACTIONAL actors (SANDBOX=agentcore-ray-farm,
#                  B4_GPU_FRACTION<1). AgentCore reality: 1 session = 1 instance, so
#                  this is one session + per-episode sub_session slots, NOT N sessions.
# BOTTLENECK     : The non-MIG L4/L40S GPU can WEDGE under sustained concurrent nvcc
#                  load — runs then see "Runtime health check failed / 500 from runtime".
#                  MITIGATION (baked as the convergence default here): raise
#                  B4_GPU_FRACTION toward 0.5 so fewer episodes compile at once. The
#                  demo run.sh recomputes the client open-gate (B4_FARM_MAX_CONCURRENT
#                  = floor(B4_FARM_GPUS / B4_GPU_FRACTION)) from this value.
# CONVERGENCE    : eval/kernelbench mean reward rises then plateaus. If the run stalls
#                  with health-check errors, raise B4_GPU_FRACTION further (0.5 -> 1.0
#                  degenerates to whole-GPU, i.e. behaves like 14_gpu_local remotely).
# REWARD         : RLVR (compiles & correct = 1.0 else 0.0); GRPO. SYNC save.
# PREREQS        : run INSIDE agentcore_p4. REMOTE-ONLY — export
#                  AGENTCORE_SANDBOX_RUNTIME_ARN to a live g6 sandbox Runtime that has
#                  the Ray farm baked in. For an on-box GPU run use 14_gpu_local.sh.
# RUNTIME/CKPT   : ~100 steps (NR=101 default); ckpt ~56 GB.
# ----------------------------------------------------------------------------
# Overridable, e.g.:  B4_GPU_FRACTION=0.34 NR=201 bash 13_gpu_shared.sh
# ============================================================================
set -euo pipefail

CONV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO="$(cd "$CONV_DIR/.." && pwd)/demo_gpu_shared"

if [ -z "${AGENTCORE_SANDBOX_RUNTIME_ARN:-}" ]; then
  echo "ERROR: demo_gpu_shared is REMOTE-ONLY — export AGENTCORE_SANDBOX_RUNTIME_ARN" >&2
  echo "       to a live g6 GPU-shared sandbox Runtime ARN (Ray farm baked in), or use" >&2
  echo "       14_gpu_local.sh for an on-box whole-GPU farm on GPUs 6,7." >&2
  exit 1
fi

: "${NR:=101}"           # real train steps = NR-1 = 100
: "${RB:=8}"; : "${NS:=8}"; : "${TEMP:=1.0}"
: "${EVAL_INTERVAL:=20}"; : "${EVAL_NS:=4}"; : "${SAVE_INTERVAL:=100}"
: "${CONC:=4}"           # GPU demos are stock 1-step-async (no MODE knob)
# Convergence default: half-GPU fractional actors to dodge the non-MIG nvcc wedge.
# The demo run.sh derives the client open-gate from this (and B4_FARM_GPUS).
: "${B4_GPU_FRACTION:=0.5}"
export NR RB NS TEMP EVAL_INTERVAL EVAL_NS SAVE_INTERVAL CONC B4_GPU_FRACTION

LOG="$DEMO/train.convergence_$(date +%Y%m%d_%H%M%S).log"
echo ">>> converging demo_gpu_shared (KernelBench, remote g6 fractional farm)  NR=$NR RB=$RB NS=$NS steps=$((NR-1)) FRACTION=$B4_GPU_FRACTION  log=$LOG"
bash "$DEMO/run.sh" 2>&1 | tee "$LOG"
