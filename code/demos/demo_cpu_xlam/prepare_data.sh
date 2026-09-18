#!/bin/bash
# Download + prepare the demo_cpu_xlam dataset (minpeter/xlam-function-calling-60k-parsed) into data/. Run INSIDE agentcore_p4:
#     docker exec agentcore_p4 bash /work/code/demos/demo_cpu_xlam/prepare_data.sh
# RLVR (verifiable reward): order-insensitive, arg-normalized gold-call match — no LLM judge. Idempotent. Builds xlam (300/40).
# The builder is vendored (agentcore_slime_demo/build_real_datasets.py) so this folder is self-contained.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
B="python3 $DIR/agentcore_slime_demo/build_real_datasets.py"
mkdir -p "$DIR/data"
$B --dataset xlam --limit 300 --offset 0 --out "$DIR/data/xlam_train.jsonl"
$B --dataset xlam --limit 40 --offset 300  --out "$DIR/data/xlam_eval.jsonl"
echo "done -> $DIR/data : xlam (300/40)"
