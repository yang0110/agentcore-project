#!/bin/bash
# ============================================================================
# CONVERGENCE RUN 04 — demo_cpu_tasks  (DATASET=math500 / MATH-500)
# ============================================================================
# TASK / DATASET : MATH-500 (the 500-problem MATH eval subset). Boxed-answer
#                  reasoning; checker extracts \boxed{} and compares to gold.
#                  Harder / more diverse than GSM8K — expect a slower climb.
# SUBSTRATE      : our own VPC CPU sandbox Runtime (SANDBOX=agentcore-runtime, steps 2-4): one
#                  fresh microVM session per episode, loop in the session. Answer-match checker => CPU sandbox.
# BOTTLENECK     : Inference + trainer bound (long CoT); checker is instant.
# CONVERGENCE    : eval/math500 mean reward rises then plateaus. Because the
#                  problems are hard, watch that group reward VARIANCE stays > 0
#                  (all-fail groups give no GRPO gradient) — raise TEMP if it dies.
# REWARD         : RLVR binary; GRPO group-relative.
# PREREQS        : run INSIDE agentcore_p4; AWS creds for the remote sandbox.
# RUNTIME/CKPT   : ~200 steps; final ckpt ~56 GB (disk caveat).
# ----------------------------------------------------------------------------
# Overridable, e.g.:  NR=401 TEMP=1.1 bash 04_cpu_math500.sh
# ============================================================================
set -euo pipefail

CONV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO="$(cd "$CONV_DIR/.." && pwd)/demo_cpu_tasks"

: "${NR:=201}"; : "${RB:=8}"; : "${NS:=8}"; : "${TEMP:=1.0}"
: "${EVAL_INTERVAL:=20}"; : "${EVAL_NS:=4}"; : "${SAVE_INTERVAL:=100}"
: "${MODE:=B}"; : "${CONC:=4}"
: "${DATASET:=math500}"   # which dataset of demo_cpu_tasks to train
export DATASET NR RB NS TEMP EVAL_INTERVAL EVAL_NS SAVE_INTERVAL MODE CONC

LOG="$DEMO/train.convergence_${DATASET}_$(date +%Y%m%d_%H%M%S).log"
echo ">>> converging demo_cpu_tasks DATASET=$DATASET (MATH-500)  NR=$NR RB=$RB NS=$NS steps=$((NR-1)) MODE=$MODE  log=$LOG"
bash "$DEMO/run.sh" 2>&1 | tee "$LOG"
