#!/bin/bash
# ============================================================================
# CONVERGENCE RUN 17 — demo_cpu_tasks DATASET=libs  (pandas + scipy + sklearn, on the DEFAULT Code Interpreter)
# ============================================================================
# (was 17_cpu_tasks_imagepool.sh / demo_cpu_imagepool: one custom image per library. The managed Code
#  Interpreter already ships pandas 2.3 / scipy 1.16 / scikit-learn 1.5 (probed live 2026-09-25), so the
#  images are gone and the same data trains on the default sandbox.)
# TASK / DATASET : mixed-task data-science RL. Each rollout step's batch spans three
#                  families (pandas / scipy / scikit-learn). data/libs_{train,eval}.jsonl
#                  (72 train, 6 eval + 2 per family in libs_<family>_eval.jsonl).
# SUBSTRATE      : the managed AgentCore Code Interpreter (SANDBOX=agentcore, the default):
#                  one isolated microVM session per episode, no image / ECR / Runtime.
# BOTTLENECK     : trainer + SGLang engines; the InvokeCodeInterpreter rate quota (30/s default)
#                  binds first if CONC is raised past ~32.
# CONVERGENCE    : eval/libs mean reward rises then plateaus (per-family evals too).
# REWARD         : hidden checker pass = 1.0 else 0.0; GRPO group-relative (one family per group).
# PREREQS        : run INSIDE agentcore_p4; AWS creds (bedrock-agentcore:*CodeInterpreter*).
#                  No deploy step. Do NOT run it with SANDBOX=agentcore-runtime unless the image
#                  has pandas/scipy/sklearn (the stdlib Dockerfile.sandbox.cpu does not -> reward 0).
# RUNTIME/CKPT   : ~200 steps; final ckpt ~56 GB (disk caveat).
# TEARDOWN       : none (every Code Interpreter session is stopped when its episode ends).
# ----------------------------------------------------------------------------
# Overridable, e.g.:  NR=401 CONC=8 bash 17_cpu_tasks_libs.sh
# ============================================================================
set -euo pipefail

CONV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO="$(cd "$CONV_DIR/.." && pwd)/demo_cpu_tasks"

: "${NR:=201}"; : "${RB:=8}"; : "${NS:=8}"; : "${TEMP:=1.0}"
: "${EVAL_INTERVAL:=20}"; : "${EVAL_NS:=4}"; : "${SAVE_INTERVAL:=100}"
: "${MODE:=B}"; : "${CONC:=4}"
: "${DATASET:=libs}"; : "${SANDBOX:=agentcore}"
export NR RB NS TEMP EVAL_INTERVAL EVAL_NS SAVE_INTERVAL MODE CONC DATASET SANDBOX

LOG="$DEMO/train.convergence_libs_$(date +%Y%m%d_%H%M%S).log"
echo ">>> converging demo_cpu_tasks DATASET=$DATASET SANDBOX=$SANDBOX (pandas/scipy/sklearn on the Code Interpreter)  NR=$NR RB=$RB NS=$NS steps=$((NR-1)) MODE=$MODE  log=$LOG"
bash "$DEMO/run.sh" 2>&1 | tee "$LOG"
