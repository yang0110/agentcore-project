#!/bin/bash
# ============================================================================
# CONVERGENCE RUN 12 — demo_gpu_single  (KernelBench, REMOTE g6, one GPU/session)
# ============================================================================
# TASK / DATASET : KernelBench (ScalingIntelligence/KernelBench). The policy writes
#                  a custom CUDA `ModelNew`; the checker COMPILES and RUNS it on GPU
#                  and scores torch.allclose vs the reference. Train = level_1 (100
#                  single-op problems); held-out eval = disjoint level_2 fusions.
# SUBSTRATE      : ONE dedicated REMOTE g6 CUDA sandbox Runtime per AgentCore session
#                  (SANDBOX=agentcore-runtime). 1 session = 1 whole g6 instance, serial.
# BOTTLENECK     : nvcc COMPILE + kernel run in the remote sandbox is the long pole
#                  (seconds/episode), and a single session serializes them. This is
#                  the "sandbox IS the bottleneck" demo — contrast with the farm (13/14).
# CONVERGENCE    : eval/kernelbench mean reward (compile-pass AND allclose) rises then
#                  plateaus. Early reward is often ~0 until the policy learns to emit
#                  a COMPILABLE kernel; watch compile-pass rate climb first.
# REWARD         : RLVR (compiles & correct = 1.0 else 0.0); GRPO group-relative.
#                  Save is SYNC (this sandbox image's seccomp blocks async-ckpt pidfd).
# PREREQS        : run INSIDE agentcore_p4. REMOTE-ONLY — you MUST export
#                  AGENTCORE_SANDBOX_RUNTIME_ARN pointing at a live g6 CUDA sandbox
#                  Runtime. No on-box path here; for a local GPU use 14_gpu_farm_whole.sh.
# RUNTIME/CKPT   : ~100 steps (GPU is slow/costly => NR=101 default); ckpt ~56 GB.
# ----------------------------------------------------------------------------
# Overridable, e.g.:  NR=201 CONC=8 bash 12_gpu_single.sh
# ============================================================================
set -euo pipefail

CONV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO="$(cd "$CONV_DIR/.." && pwd)/demo_gpu_single"

if [ -z "${AGENTCORE_SANDBOX_RUNTIME_ARN:-}" ]; then
  echo "ERROR: demo_gpu_single is REMOTE-ONLY — export AGENTCORE_SANDBOX_RUNTIME_ARN" >&2
  echo "       to a live g6 CUDA sandbox Runtime ARN, or use 14_gpu_farm_whole.sh for an" >&2
  echo "       on-box GPU run (Ray whole-GPU farm on GPUs 6,7)." >&2
  exit 1
fi

: "${NR:=101}"           # real train steps = NR-1 = 100 (GPU runs are expensive)
: "${RB:=8}"; : "${NS:=8}"; : "${TEMP:=1.0}"
: "${EVAL_INTERVAL:=20}"; : "${EVAL_NS:=4}"; : "${SAVE_INTERVAL:=100}"
: "${CONC:=4}"           # GPU demos are stock 1-step-async (no MODE knob)
export NR RB NS TEMP EVAL_INTERVAL EVAL_NS SAVE_INTERVAL CONC

LOG="$DEMO/train.convergence_$(date +%Y%m%d_%H%M%S).log"
echo ">>> converging demo_gpu_single (KernelBench, remote g6)  NR=$NR RB=$RB NS=$NS steps=$((NR-1))  log=$LOG"
bash "$DEMO/run.sh" 2>&1 | tee "$LOG"
