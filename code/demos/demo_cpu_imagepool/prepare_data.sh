#!/bin/bash
# Build demo_cpu_imagepool's MIXED-task dataset into data/ (offline, stdlib-only — no network, no GPU).
#
# Run INSIDE the persistent container agentcore_p4 (slimerl/slime). From the repo root on the host:
#     docker exec agentcore_p4 bash /work/code/demos/demo_cpu_imagepool/prepare_data.sh
#
# Produces ONE interleaved train/eval pair spanning several task families, so every GRPO rollout
# step trains a batch of prompts that covers multiple tasks. Each row carries its own family +
# hidden RLVR checker in metadata.task, so the in-process run_episode harness dispatches per row and
# the remote AgentCore sandbox session is written with that episode's task-specific files. Fully
# self-contained: imports only this dir's vendored agentcore_slime_demo/. Idempotent (overwrites).
#
# Default mix: datasci + scipy + sklearn — the THREE custom-image families, each needing its OWN
# prebuilt sandbox image (pandas / scipy / scikit-learn), NONE of which is in the generic CPU image.
# This is the point of the demo: a POOL of images, one per family, and each episode is routed to
# the image its family needs via AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE (capability-routing by
# family; see run.sh + agentcore_slime_demo/sandbox.py). Override with FAMILIES / BALANCED / REPEAT
# (e.g. add the generic-CPU `math` family to show cpu + pool coexisting: FAMILIES=math,datasci,scipy,sklearn).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FAMILIES="${FAMILIES:-datasci,scipy,sklearn}"
REPEAT="${REPEAT:-6}"
BAL=(); [ "${BALANCED:-1}" = "1" ] && BAL=(--balanced)   # default: balanced (every step spans families)

python3 "$DIR/build_mixed_datasets.py" \
   --families "$FAMILIES" \
   --out-dir "$DIR/data" \
   --repeat "$REPEAT" \
   "${BAL[@]}"

echo "done -> $DIR/data : mixed_train.jsonl + mixed_eval.jsonl (families=$FAMILIES)"
