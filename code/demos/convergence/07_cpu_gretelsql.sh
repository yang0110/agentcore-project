#!/bin/bash
# ============================================================================
# CONVERGENCE RUN 07 — demo_cpu_tasks  (DATASET=gretelsql)
# ============================================================================
# TASK / DATASET : Gretel synthetic text-to-SQL. Given a schema + NL question,
#                  the policy writes a SQL query; the checker runs it and compares
#                  the result set (execution-based, not string match).
# SUBSTRATE      : our own VPC CPU sandbox Runtime (SANDBOX=agentcore-runtime, steps 2-4): one
#                  fresh microVM session per episode, loop in the session. SQL EXECUTION in the sandbox.
# BOTTLENECK     : Sandbox executes SQL against a small in-memory DB — cheap;
#                  inference + trainer dominate. Reward is denser than pure math
#                  (partial-schema queries often run), so GRPO signal is healthy.
# CONVERGENCE    : eval/gretelsql mean reward (result-set match rate) rises then
#                  plateaus.
# REWARD         : RLVR (query result matches gold = 1.0 else 0.0); GRPO group-relative.
# PREREQS        : run INSIDE agentcore_p4; AWS creds for the remote sandbox.
# RUNTIME/CKPT   : ~200 steps; final ckpt ~56 GB (disk caveat).
# ----------------------------------------------------------------------------
# Overridable, e.g.:  NR=301 bash 07_cpu_gretelsql.sh
# ============================================================================
set -euo pipefail

CONV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO="$(cd "$CONV_DIR/.." && pwd)/demo_cpu_tasks"

: "${NR:=201}"; : "${RB:=8}"; : "${NS:=8}"; : "${TEMP:=1.0}"
: "${EVAL_INTERVAL:=20}"; : "${EVAL_NS:=4}"; : "${SAVE_INTERVAL:=100}"
: "${CONC:=4}"
: "${DATASET:=gretelsql}"   # which dataset of demo_cpu_tasks to train
export DATASET NR RB NS TEMP EVAL_INTERVAL EVAL_NS SAVE_INTERVAL CONC

LOG="$DEMO/train.convergence_${DATASET}_$(date +%Y%m%d_%H%M%S).log"
echo ">>> converging demo_cpu_tasks DATASET=$DATASET (text-to-SQL)  NR=$NR RB=$RB NS=$NS steps=$((NR-1))  log=$LOG"
bash "$DEMO/run.sh" 2>&1 | tee "$LOG"
