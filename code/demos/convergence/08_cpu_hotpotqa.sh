#!/bin/bash
# ============================================================================
# CONVERGENCE RUN 08 — demo_cpu_tasks  (DATASET=hotpotqa)
# ============================================================================
# TASK / DATASET : HotpotQA multi-hop retrieval QA. Given a question + supporting
#                  context, the policy reasons across passages and emits a short
#                  answer; the checker scores it against gold (exact-match / F1).
# SUBSTRATE      : REMOTE managed AgentCore sandbox Runtime (SANDBOX=agentcore-runtime),
#                  K=2 CPU sandbox ARNs baked in. Answer-scoring checker => CPU sandbox.
# BOTTLENECK     : Long retrieved context inflates prompt length => SGLang prefill
#                  is the pressure point; the checker is instant. Trainer + infer bound.
# CONVERGENCE    : eval/hotpotqa mean reward (EM/F1) rises then plateaus. Partial
#                  F1 credit keeps GRPO group variance alive even when EM is rare.
# REWARD         : RLVR (answer-match / F1); GRPO group-relative.
# PREREQS        : run INSIDE agentcore_p4; AWS creds for the remote sandbox.
# RUNTIME/CKPT   : ~200 steps; final ckpt ~56 GB (disk caveat).
# ----------------------------------------------------------------------------
# Overridable, e.g.:  NR=301 bash 08_cpu_hotpotqa.sh
# ============================================================================
set -euo pipefail

CONV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO="$(cd "$CONV_DIR/.." && pwd)/demo_cpu_tasks"

: "${NR:=201}"; : "${RB:=8}"; : "${NS:=8}"; : "${TEMP:=1.0}"
: "${EVAL_INTERVAL:=20}"; : "${EVAL_NS:=4}"; : "${SAVE_INTERVAL:=100}"
: "${MODE:=B}"; : "${CONC:=4}"
: "${DATASET:=hotpotqa}"   # which dataset of demo_cpu_tasks to train
export DATASET NR RB NS TEMP EVAL_INTERVAL EVAL_NS SAVE_INTERVAL MODE CONC

LOG="$DEMO/train.convergence_${DATASET}_$(date +%Y%m%d_%H%M%S).log"
echo ">>> converging demo_cpu_tasks DATASET=$DATASET (multi-hop RAG QA)  NR=$NR RB=$RB NS=$NS steps=$((NR-1)) MODE=$MODE  log=$LOG"
bash "$DEMO/run.sh" 2>&1 | tee "$LOG"
