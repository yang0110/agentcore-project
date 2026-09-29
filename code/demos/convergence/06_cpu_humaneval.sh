#!/bin/bash
# ============================================================================
# CONVERGENCE RUN 06 — demo_cpu_tasks  (DATASET=humaneval)
# ============================================================================
# TASK / DATASET : HumanEval Python programming. The policy completes a function
#                  from its docstring; the checker EXECUTES the hidden unit tests
#                  in the sandbox (multi-turn repair, same harness as MBPP).
# SUBSTRATE      : our own VPC CPU sandbox Runtime (SANDBOX=agentcore-runtime, steps 2-4): one
#                  fresh microVM session per episode, loop in the session. Real code EXECUTION in the sandbox.
# BOTTLENECK     : Sandbox exec is real work here (running tests), but still short
#                  Python — inference + trainer remain dominant. Sandbox count K
#                  flat in the ablation.
# CONVERGENCE    : eval/humaneval mean reward (pass@ over EVAL_NS) rises then plateaus.
# REWARD         : RLVR binary (tests pass = 1.0 else 0.0); GRPO group-relative.
# PREREQS        : run INSIDE agentcore_p4; AWS creds for the remote sandbox.
# RUNTIME/CKPT   : ~200 steps; final ckpt ~56 GB (disk caveat).
# ----------------------------------------------------------------------------
# Overridable, e.g.:  CONC=8 bash 06_cpu_humaneval.sh
# ============================================================================
set -euo pipefail

CONV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO="$(cd "$CONV_DIR/.." && pwd)/demo_cpu_tasks"

: "${NR:=201}"; : "${RB:=8}"; : "${NS:=8}"; : "${TEMP:=1.0}"
: "${EVAL_INTERVAL:=20}"; : "${EVAL_NS:=4}"; : "${SAVE_INTERVAL:=100}"
: "${CONC:=4}"
: "${DATASET:=humaneval}"   # which dataset of demo_cpu_tasks to train
export DATASET NR RB NS TEMP EVAL_INTERVAL EVAL_NS SAVE_INTERVAL CONC

LOG="$DEMO/train.convergence_${DATASET}_$(date +%Y%m%d_%H%M%S).log"
echo ">>> converging demo_cpu_tasks DATASET=$DATASET (HumanEval)  NR=$NR RB=$RB NS=$NS steps=$((NR-1))  log=$LOG"
bash "$DEMO/run.sh" 2>&1 | tee "$LOG"
