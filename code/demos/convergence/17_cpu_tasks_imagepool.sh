#!/bin/bash
# ============================================================================
# CONVERGENCE RUN 17 — demo_cpu_tasks VARIANT=imagepool  (datasci+scipy+sklearn; a POOL of images)
# ============================================================================
# (was demo_cpu_imagepool, merged 2026-09-25. Run 17 used to be demo_cpu_customruntime, which only
#  duplicated 05 on a custom Runtime; that angle is now demo_cpu_tasks/compare_backends.py.)
# TASK / DATASET : mixed-task data-science RL. Each rollout step's batch spans three
#                  families (pandas / scipy / scikit-learn); each family needs ITS OWN
#                  prebuilt image. data/imagepool_{train,eval}.jsonl (72/6 per family).
# SUBSTRATE      : REMOTE serverless microVMs, ONE custom AgentCore Runtime per image
#                  (SANDBOX=agentcore-runtime). Every episode is routed by task.image via
#                  AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE; a mis-routed episode scores 0.
# BOTTLENECK     : trainer + SGLang engines; three Runtimes cold-boot independently.
# CONVERGENCE    : eval/imagepool mean reward rises then plateaus (per-family evals too).
# REWARD         : hidden checker pass = 1.0 else 0.0; GRPO group-relative (one family per group).
# PREREQS        : run INSIDE agentcore_p4; AWS creds; the three Runtimes deployed
#                  (VARIANT=imagepool bash 4_deploy_runtime.sh -> deploy/runtime.imagepool.env,
#                  which exports AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE — source it first).
# RUNTIME/CKPT   : ~200 steps; final ckpt ~56 GB (disk caveat).
# TEARDOWN       : VARIANT=imagepool bash 6_cleanup.sh (deletes the three Runtimes).
# ----------------------------------------------------------------------------
# Overridable, e.g.:  NR=401 CONC=8 bash 17_cpu_tasks_imagepool.sh
# ============================================================================
set -euo pipefail

CONV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO="$(cd "$CONV_DIR/.." && pwd)/demo_cpu_tasks"

# Pick up the deployed ARN map if it is not already in the environment.
if [ -z "${AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE:-}" ] && [ -f "$DEMO/deploy/runtime.imagepool.env" ]; then
  set -a; . "$DEMO/deploy/runtime.imagepool.env"; set +a
fi

: "${NR:=201}"; : "${RB:=8}"; : "${NS:=8}"; : "${TEMP:=1.0}"
: "${EVAL_INTERVAL:=20}"; : "${EVAL_NS:=4}"; : "${SAVE_INTERVAL:=100}"
: "${MODE:=B}"; : "${CONC:=4}"
: "${VARIANT:=imagepool}"
export NR RB NS TEMP EVAL_INTERVAL EVAL_NS SAVE_INTERVAL MODE CONC VARIANT

LOG="$DEMO/train.convergence_imagepool_$(date +%Y%m%d_%H%M%S).log"
echo ">>> converging demo_cpu_tasks VARIANT=$VARIANT (datasci/scipy/sklearn image pool)  NR=$NR RB=$RB NS=$NS steps=$((NR-1)) MODE=$MODE  log=$LOG"
bash "$DEMO/run.sh" 2>&1 | tee "$LOG"
