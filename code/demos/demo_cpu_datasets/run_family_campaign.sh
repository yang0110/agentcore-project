#!/bin/bash
# Run the demo_cpu_datasets task-family expansions ONE BY ONE: for each family, a full real slime training
# job (train_async.py, ~10 train steps + 1 eval) driven by our in-process run_episode harness with
# the sandbox on the REMOTE CPU AgentCore Runtime (SANDBOX=agentcore-runtime). Each run grabs all 8
# GPUs + its own ray head, so they are strictly sequential. Per-family logs land in logs/; a running
# summary is appended to logs/campaign.log. Continues past a failing family (no -e).
set -uo pipefail

export SANDBOX=agentcore-runtime
export AGENTCORE_SANDBOX_RUNTIME_ARN="${AGENTCORE_SANDBOX_RUNTIME_ARN:-arn:aws:bedrock-agentcore:us-west-2:<ACCOUNT_ID>:runtime/acs_sandbox_cpu-1gzBVp675S}"
export AWS_REGION="${AWS_REGION:-us-west-2}"
export AGENTCORE_REGION="${AGENTCORE_REGION:-us-west-2}"

cd /work/code/demos/demo_cpu_datasets
mkdir -p logs
CAMP=logs/campaign.log

# New families first (fresh expansion), then the six originals.
FAMILIES="${FAMILIES:-regex planning config data swe tool_use sql math ctf rag}"

echo "=== campaign START $(date -u) families=[$FAMILIES] arn=$AGENTCORE_SANDBOX_RUNTIME_ARN ===" | tee -a "$CAMP"
for f in $FAMILIES; do
   echo "--- $(date -u) BEGIN $f ---" | tee -a "$CAMP"
   FAMILY="$f" bash run.sh > "logs/family_${f}.log" 2>&1
   rc=$?
   # A quick reward peek from the log tail (best-effort).
   rew=$(grep -oE "eval[^ ]* (solve|reward)[^ ]*[: ]+[0-9.]+" "logs/family_${f}.log" 2>/dev/null | tail -1)
   echo "--- $(date -u) DONE  $f rc=$rc ${rew} ---" | tee -a "$CAMP"
   # Clean ray/sglang between runs so the next family starts from a clean 8-GPU slate.
   pkill -9 sglang 2>/dev/null || true
   ray stop --force  >/dev/null 2>&1 || true
   pkill -9 -f "ray::" 2>/dev/null || true
   sleep 8
done
echo "=== campaign DONE $(date -u) ===" | tee -a "$CAMP"
