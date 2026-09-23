#!/bin/bash
# ============================================================================
# CONVERGENCE RUN 09 — demo_cpu_datasets  (DATASET=xlam)
# ============================================================================
# TASK / DATASET : Salesforce xLAM function-calling. Given a user request + a tool
#                  schema, the policy must emit a well-formed tool call with the
#                  right function name and arguments; the checker validates the
#                  call against the gold invocation.
# SUBSTRATE      : REMOTE managed AgentCore sandbox Runtime (SANDBOX=agentcore-runtime),
#                  K=2 CPU sandbox ARNs baked in. JSON-validity + arg-match checker.
# BOTTLENECK     : Trainer + inference bound. The failure mode to watch is the
#                  policy emitting malformed / hallucinated tool calls (cf. the b2
#                  dead-reward finding) — early reward can sit near 0 until it
#                  learns the call FORMAT, then climbs.
# CONVERGENCE    : eval/xlam mean reward (valid + correct tool call) rises then
#                  plateaus. If it is pinned at 0 for many steps, the format isn't
#                  being learned — raise TEMP or inspect a rollout.
# REWARD         : RLVR (tool call name+args match = 1.0 else 0.0); GRPO group-relative.
# PREREQS        : run INSIDE agentcore_p4; AWS creds for the remote sandbox.
# RUNTIME/CKPT   : ~200 steps; final ckpt ~56 GB (disk caveat).
# ----------------------------------------------------------------------------
# Overridable, e.g.:  TEMP=1.1 NR=301 bash 09_cpu_xlam.sh
# ============================================================================
set -euo pipefail

CONV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO="$(cd "$CONV_DIR/.." && pwd)/demo_cpu_datasets"

: "${NR:=201}"; : "${RB:=8}"; : "${NS:=8}"; : "${TEMP:=1.0}"
: "${EVAL_INTERVAL:=20}"; : "${EVAL_NS:=4}"; : "${SAVE_INTERVAL:=100}"
: "${MODE:=B}"; : "${CONC:=4}"
: "${DATASET:=xlam}"   # which dataset of demo_cpu_datasets to train
export DATASET NR RB NS TEMP EVAL_INTERVAL EVAL_NS SAVE_INTERVAL MODE CONC

LOG="$DEMO/train.convergence_${DATASET}_$(date +%Y%m%d_%H%M%S).log"
echo ">>> converging demo_cpu_datasets DATASET=$DATASET (function-calling)  NR=$NR RB=$RB NS=$NS steps=$((NR-1)) MODE=$MODE  log=$LOG"
bash "$DEMO/run.sh" 2>&1 | tee "$LOG"
