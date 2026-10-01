#!/bin/bash
# ============================================================================
# CONVERGENCE RUN 20 — demo_cpu_terminalbench  (Terminal-Bench (one image + microVM Runtime per task, ~218; the task's hidden pytest oracle))
# ============================================================================
# TASK / DATASET : data/terminalbench_{train,eval}.jsonl, filtered to the deployed + oracle-verified pool.
# SUBSTRATE      : image pool of our own VPC Runtimes (SANDBOX=agentcore-runtime), strict row -> image
#                  -> Runtime routing; a FRESH microVM session per episode, pre-opened ahead of time and
#                  heartbeated across the weight-sync stall. The agent loop runs in the session.
# REWARD         : hidden oracle pass = 1.0 else 0.0; GRPO group-relative.
# PREREQS        : run INSIDE agentcore_p4; AWS creds. Deploy the pool first (demo_cpu_terminalbench steps 2-4,
#                  then step 1 to rebuild the data from it).
# RUNTIME/CKPT   : ~200 steps; final ckpt ~56 GB (disk caveat).
# TEARDOWN       : sessions stop per episode; delete the pool with demo_cpu_terminalbench/6_cleanup.sh.
# ----------------------------------------------------------------------------
# Overridable, e.g.:  NR=401 bash 20_cpu_terminalbench.sh
# ============================================================================
set -euo pipefail

CONV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO="$(cd "$CONV_DIR/.." && pwd)/demo_cpu_terminalbench"

: "${NR:=201}"; : "${EVAL_INTERVAL:=20}"; : "${EVAL_NS:=4}"; : "${SAVE_INTERVAL:=100}"
: "${SANDBOX:=agentcore-runtime}"
export NR EVAL_INTERVAL EVAL_NS SAVE_INTERVAL SANDBOX

LOG="$DEMO/train.convergence_terminalbench_$(date +%Y%m%d_%H%M%S).log"
echo ">>> converging demo_cpu_terminalbench SANDBOX=$SANDBOX  NR=$NR steps=$((NR-1))  log=$LOG"
bash "$DEMO/run.sh" 2>&1 | tee "$LOG"
