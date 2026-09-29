#!/bin/bash
# ============================================================================
# CONVERGENCE RUN 05 — demo_cpu_tasks  (DATASET=deepscaler)
# ============================================================================
# TASK / DATASET : DeepScaleR competition-math problems (long chain-of-thought,
#                  hardest of the math families). Boxed-answer checker vs gold.
# SUBSTRATE      : our own VPC CPU sandbox Runtime (SANDBOX=agentcore-runtime, steps 2-4): one
#                  fresh microVM session per episode, loop in the session. Answer-match checker => CPU sandbox.
# BOTTLENECK     : SGLang decode (long CoT dominates wall-clock); checker instant.
#                  This is the run where a longer --rollout-max-response-len would
#                  matter most, but we keep the demo default for comparability.
# CONVERGENCE    : eval/deepscaler mean reward rises slowly then plateaus. Expect
#                  the flattest curve of the math set; consider NR=401 to see it.
# REWARD         : RLVR binary; GRPO group-relative.
# PREREQS        : run INSIDE agentcore_p4; AWS creds for the remote sandbox.
# RUNTIME/CKPT   : ~200 steps (longest wall-clock/step); final ckpt ~56 GB.
# ----------------------------------------------------------------------------
# Overridable, e.g.:  NR=401 bash 05_cpu_deepscaler.sh
# ============================================================================
set -euo pipefail

CONV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO="$(cd "$CONV_DIR/.." && pwd)/demo_cpu_tasks"

: "${NR:=201}"; : "${RB:=8}"; : "${NS:=8}"; : "${TEMP:=1.0}"
: "${EVAL_INTERVAL:=20}"; : "${EVAL_NS:=4}"; : "${SAVE_INTERVAL:=100}"
: "${CONC:=4}"
: "${DATASET:=deepscaler}"   # which dataset of demo_cpu_tasks to train
export DATASET NR RB NS TEMP EVAL_INTERVAL EVAL_NS SAVE_INTERVAL CONC

LOG="$DEMO/train.convergence_${DATASET}_$(date +%Y%m%d_%H%M%S).log"
echo ">>> converging demo_cpu_tasks DATASET=$DATASET (DeepScaleR)  NR=$NR RB=$RB NS=$NS steps=$((NR-1))  log=$LOG"
bash "$DEMO/run.sh" 2>&1 | tee "$LOG"
