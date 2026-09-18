#!/bin/bash
# Download + prepare the demo_cpu_math500 dataset (HuggingFaceH4/MATH-500) into data/. Run INSIDE agentcore_p4:
#     docker exec agentcore_p4 bash /work/code/demos/demo_cpu_math500/prepare_data.sh
# RLVR (verifiable reward): numeric match (tol 1e-4), numeric-answer subset — no LLM judge. Idempotent. Builds math500 (160/40).
# The builder is vendored (agentcore_slime_demo/build_real_datasets.py) so this folder is self-contained.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
B="python3 $DIR/agentcore_slime_demo/build_real_datasets.py"
mkdir -p "$DIR/data"
$B --dataset math500 --split test --limit 160 --offset 0 --out "$DIR/data/math500_train.jsonl"
$B --dataset math500 --split test --limit 40 --offset 160  --out "$DIR/data/math500_eval.jsonl"
echo "done -> $DIR/data : math500 (160/40)"
