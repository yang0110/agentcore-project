#!/bin/bash
# ============================================================================
# CONVERGENCE RUN 19 — demo_gpu_customruntime  (KernelBench, YOUR custom GPU Runtime)
# ============================================================================
# TASK / DATASET : KernelBench (same as 12/13/14/18): write a custom CUDA ModelNew,
#                  the checker COMPILES + RUNS it on GPU and scores torch.allclose vs
#                  the reference. Train = level_1 (100); held-out eval = level_2 (20).
# SUBSTRATE      : the sandbox is YOUR OWN custom GPU-backed AgentCore Runtime on a g6
#                  Instance (SANDBOX=agentcore-runtime). The POINT is that CUDA is only
#                  available in a custom GPU Runtime — the managed Code Interpreter is
#                  CPU-only (torch.cuda.is_available() is False), so this training
#                  CANNOT run there. This is the custom-GPU-Runtime capability made into
#                  a real RL run (gpu_torch_smoke.py remains the seconds-long probe).
# TWO WAYS TO RUN:
#   * REMOTE (default): export AGENTCORE_SANDBOX_RUNTIME_ARN -> your g6 CUDA Runtime.
#   * ON-BOX (LOCAL_GPU=1): serve the SAME GPU sandbox image LOCALLY as a whole-GPU Ray
#     farm on THIS box's reserved GPUs 6,7 (endpoint :8080), slime on GPUs 0-5. Start the
#     farm on the HOST first:  bash code/deploy/run_local_gpu_runtime.sh local
# BOTTLENECK     : nvcc compile + kernel run in the (remote or local) GPU sandbox.
# CONVERGENCE    : eval/kernelbench mean reward (compile-pass AND allclose) rises then
#                  plateaus; watch compile-pass rate climb first.
# REWARD         : RLVR (compiles & correct = 1.0 else 0.0); GRPO. SYNC save under
#                  LOCAL_GPU (the local sandbox image's seccomp blocks async-ckpt pidfd).
# PREREQS        : run INSIDE agentcore_p4. Either a live g6 ARN (remote) OR LOCAL_GPU=1
#                  with the local farm up (on-box). RUNTIME/CKPT: ~100 steps; ckpt ~56 GB.
# STATUS         : the LOCAL_GPU path is validated on THIS 8xL40S box; the REMOTE g6
#                  path is UNVERIFIED on hardware (offline-validated only).
# ----------------------------------------------------------------------------
# Overridable, e.g.:
#   REMOTE:  AGENTCORE_SANDBOX_RUNTIME_ARN=<g6> NR=201 bash 19_gpu_customruntime.sh
#   ON-BOX:  LOCAL_GPU=1 NR=201 bash 19_gpu_customruntime.sh   (farm must be up on 6,7)
# ============================================================================
set -euo pipefail

CONV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO="$(cd "$CONV_DIR/.." && pwd)/demo_gpu_customruntime"

: "${LOCAL_GPU:=0}"
if [ "$LOCAL_GPU" = "1" ]; then
  ENDPOINT="${AGENTCORE_LOCAL_ENDPOINT:-http://127.0.0.1:8080}"
  if ! curl -fsS "${ENDPOINT}/ping" >/dev/null 2>&1; then
    echo "ERROR: LOCAL_GPU=1 but the local GPU farm is not reachable at ${ENDPOINT}/ping." >&2
    echo "       Start it FIRST on the HOST (not inside agentcore_p4):" >&2
    echo "         bash code/deploy/run_local_gpu_runtime.sh local" >&2
    exit 1
  fi
elif [ -z "${AGENTCORE_SANDBOX_RUNTIME_ARN:-}" ]; then
  echo "ERROR: demo_gpu_customruntime needs a GPU sandbox. Either:" >&2
  echo "   * export AGENTCORE_SANDBOX_RUNTIME_ARN to a live g6 CUDA sandbox Runtime, or" >&2
  echo "   * set LOCAL_GPU=1 to serve the GPU sandbox on-box (Ray farm on GPUs 6,7)." >&2
  exit 1
fi

: "${NR:=101}"           # real train steps = NR-1 = 100
: "${RB:=8}"; : "${NS:=8}"; : "${TEMP:=1.0}"
: "${EVAL_INTERVAL:=20}"; : "${EVAL_NS:=4}"; : "${SAVE_INTERVAL:=100}"
export NR RB NS TEMP EVAL_INTERVAL EVAL_NS SAVE_INTERVAL LOCAL_GPU
# LOCAL_GPU=1 sets its own CONC / GPU split / B4_* farm knobs inside the demo run.sh.

LOG="$DEMO/train.convergence_$(date +%Y%m%d_%H%M%S).log"
_where=$([ "$LOCAL_GPU" = "1" ] && echo "on-box farm 6,7" || echo "remote g6")
echo ">>> converging demo_gpu_customruntime (KernelBench, $_where)  NR=$NR RB=$RB NS=$NS steps=$((NR-1))  log=$LOG"
bash "$DEMO/run.sh" 2>&1 | tee "$LOG"
