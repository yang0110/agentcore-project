#!/bin/bash
# Download + prepare the demo_cpu_hotpotqa dataset (hotpotqa/hotpot_qa (distractor)) into data/. Run INSIDE agentcore_p4:
#     docker exec agentcore_p4 bash /work/code/demos/demo_cpu_hotpotqa/prepare_data.sh
# RLVR (verifiable reward): SQuAD-normalized exact match (10 passages -> ./docs corpus) — no LLM judge. Idempotent. Builds hotpotqa (300/40).
# The builder is vendored (agentcore_slime_demo/build_real_datasets.py) so this folder is self-contained.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
B="python3 $DIR/agentcore_slime_demo/build_real_datasets.py"
mkdir -p "$DIR/data"
$B --dataset hotpotqa --split train --limit 300 --out "$DIR/data/hotpotqa_train.jsonl"
$B --dataset hotpotqa --split validation --limit 40  --out "$DIR/data/hotpotqa_eval.jsonl"
echo "done -> $DIR/data : hotpotqa (300/40)"
