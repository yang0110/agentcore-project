#!/bin/bash
# Download + prepare the demo_cpu_humaneval dataset (openai/openai_humaneval) into data/. Run INSIDE agentcore_p4:
#     docker exec agentcore_p4 bash /work/code/demos/demo_cpu_humaneval/prepare_data.sh
# RLVR (verifiable reward): the row's own unit-test fn — check(entry_point) — no LLM judge. Idempotent. Builds humaneval (140/24).
# The builder is vendored (agentcore_slime_demo/build_real_datasets.py) so this folder is self-contained.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
B="python3 $DIR/agentcore_slime_demo/build_real_datasets.py"
mkdir -p "$DIR/data"
$B --dataset humaneval --split test --limit 140 --offset 0 --out "$DIR/data/humaneval_train.jsonl"
$B --dataset humaneval --split test --limit 24 --offset 140  --out "$DIR/data/humaneval_eval.jsonl"
echo "done -> $DIR/data : humaneval (140/24)"
