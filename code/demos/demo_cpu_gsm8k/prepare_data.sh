#!/bin/bash
# Download + prepare the demo_cpu_gsm8k dataset (openai/gsm8k (config main)) into data/. Run INSIDE agentcore_p4:
#     docker exec agentcore_p4 bash /work/code/demos/demo_cpu_gsm8k/prepare_data.sh
# RLVR (verifiable reward): exact integer match — no LLM judge. Idempotent. Builds gsm8k (200/40).
# The builder is vendored (agentcore_slime_demo/build_real_datasets.py) so this folder is self-contained.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
B="python3 $DIR/agentcore_slime_demo/build_real_datasets.py"
mkdir -p "$DIR/data"
$B --dataset gsm8k --split train --limit 200 --out "$DIR/data/gsm8k_train.jsonl"
$B --dataset gsm8k --split test --limit 40  --out "$DIR/data/gsm8k_eval.jsonl"
echo "done -> $DIR/data : gsm8k (200/40)"
