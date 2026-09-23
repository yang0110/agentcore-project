#!/bin/bash
# Download + prepare the DeepScaleR dataset (agentica-org/DeepScaleR-Preview-Dataset) into data/. Run INSIDE agentcore_p4:
#     docker exec agentcore_p4 bash /work/code/demos/demo_cpu_harness_scale/prepare_data.sh
# RLVR (verifiable reward): numeric match (tol 1e-4), numeric-answer subset — no LLM judge. Idempotent. Builds deepscaler (400/40).
# The builder is vendored (agentcore_slime_demo/build_real_datasets.py) so this folder is self-contained.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
B="python3 $DIR/agentcore_slime_demo/build_real_datasets.py"
mkdir -p "$DIR/data"
$B --dataset deepscaler --limit 400 --offset 0 --out "$DIR/data/deepscaler_train.jsonl"
$B --dataset deepscaler --limit 40 --offset 400  --out "$DIR/data/deepscaler_eval.jsonl"
echo "done -> $DIR/data : deepscaler (400/40)"
