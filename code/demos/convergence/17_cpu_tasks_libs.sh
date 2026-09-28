#!/bin/bash
# ============================================================================
# CONVERGENCE RUN 17 — demo_cpu_tasks DATASET=libs  (pandas + scipy + sklearn, on the libs Runtime)
# ============================================================================
# (was 17_cpu_tasks_imagepool.sh / demo_cpu_imagepool: one custom image per library. Now ONE libs image,
#  deploy/Dockerfile.sandbox.libs, serves all three; image_routing.py maps datasci/scipy/sklearn to it.)
# TASK / DATASET : mixed-task data-science RL. Each rollout step's batch spans three
#                  families (pandas / scipy / scikit-learn). data/libs_{train,eval}.jsonl
#                  (72 train, 6 eval + 2 per family in libs_<family>_eval.jsonl).
# SUBSTRATE      : our own VPC libs sandbox Runtime (SANDBOX=agentcore-runtime): one fresh
#                  microVM session per episode, the agent loop runs in the session.
# BOTTLENECK     : trainer + SGLang engines; the sandbox is not the bottleneck at CONC=4.
# CONVERGENCE    : eval/libs mean reward rises then plateaus (per-family evals too).
# REWARD         : hidden checker pass = 1.0 else 0.0; GRPO group-relative (one family per group).
# PREREQS        : run INSIDE agentcore_p4; AWS creds. Deploy first (demo_cpu_tasks steps 2-4 with
#                  the default IMAGES, which include libs); this script sources its
#                  deploy/runtime.env, whose ..._ARN_BY_IMAGE needs a "libs" entry, else run.sh refuses.
# RUNTIME/CKPT   : ~200 steps; final ckpt ~56 GB (disk caveat).
# TEARDOWN       : sessions stop per episode; delete the Runtimes with demo_cpu_tasks/6_cleanup.sh.
# ----------------------------------------------------------------------------
# Overridable, e.g.:  NR=401 CONC=8 bash 17_cpu_tasks_libs.sh
# ============================================================================
set -euo pipefail

CONV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO="$(cd "$CONV_DIR/.." && pwd)/demo_cpu_tasks"
if [ -f "$DEMO/deploy/runtime.env" ]; then set -a; . "$DEMO/deploy/runtime.env"; set +a; fi

: "${NR:=201}"; : "${RB:=8}"; : "${NS:=8}"; : "${TEMP:=1.0}"
: "${EVAL_INTERVAL:=20}"; : "${EVAL_NS:=4}"; : "${SAVE_INTERVAL:=100}"
: "${MODE:=B}"; : "${CONC:=4}"
: "${DATASET:=libs}"; : "${SANDBOX:=agentcore-runtime}"
export NR RB NS TEMP EVAL_INTERVAL EVAL_NS SAVE_INTERVAL MODE CONC DATASET SANDBOX

LOG="$DEMO/train.convergence_libs_$(date +%Y%m%d_%H%M%S).log"
echo ">>> converging demo_cpu_tasks DATASET=$DATASET SANDBOX=$SANDBOX (pandas/scipy/sklearn on the libs Runtime)  NR=$NR RB=$RB NS=$NS steps=$((NR-1)) MODE=$MODE  log=$LOG"
bash "$DEMO/run.sh" 2>&1 | tee "$LOG"
