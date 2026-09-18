#!/bin/bash
# Build demo_cpu_mixtask's MIXED-task dataset into data/ (offline, stdlib-only — no network, no GPU).
#
# Run INSIDE the persistent container agentcore_p4 (slimerl/slime). From the repo root on the host:
#     docker exec agentcore_p4 bash /work/code/demos/demo_cpu_mixtask/prepare_data.sh
#
# Produces ONE interleaved train/eval pair spanning several task families, so every GRPO rollout
# step trains a batch of prompts that covers multiple tasks. Each row carries its own family +
# hidden RLVR checker in metadata.task, so the in-process run_episode harness dispatches per row and
# the remote AgentCore sandbox session is written with that episode's task-specific files. Fully
# self-contained: imports only this dir's vendored agentcore_slime_demo/. Idempotent (overwrites).
#
# Default mix: math + sql + ctf + datasci (capability-diverse: numeric tolerance, SQL result-set,
# reverse-engineering, and a pandas data-analysis family). `datasci` is the CUSTOM-IMAGE family —
# its sandbox needs pandas (absent from the generic CPU image), so its episodes route to a
# pandas-equipped Runtime via AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE (capability-routing by family;
# see run.sh + agentcore_slime_demo/sandbox.py). Override with FAMILIES / BALANCED / REPEAT.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FAMILIES="${FAMILIES:-math,sql,ctf,datasci}"
REPEAT="${REPEAT:-6}"
BAL=(); [ "${BALANCED:-1}" = "1" ] && BAL=(--balanced)   # default: balanced (every step spans families)

python3 "$DIR/build_mixed_datasets.py" \
   --families "$FAMILIES" \
   --out-dir "$DIR/data" \
   --repeat "$REPEAT" \
   "${BAL[@]}"

echo "done -> $DIR/data : mixed_train.jsonl + mixed_eval.jsonl (families=$FAMILIES)"
