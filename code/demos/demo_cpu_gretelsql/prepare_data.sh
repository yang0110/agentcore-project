#!/bin/bash
# Download + prepare the demo_cpu_gretelsql dataset (gretelai/synthetic_text_to_sql) into data/. Run INSIDE agentcore_p4:
#     docker exec agentcore_p4 bash /work/code/demos/demo_cpu_gretelsql/prepare_data.sh
# RLVR (verifiable reward): SQL execution vs the gold result set (stdlib sqlite3) — no LLM judge. Idempotent. Builds gretelsql (300/40).
# The builder is vendored (agentcore_slime_demo/build_real_datasets.py) so this folder is self-contained.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
B="python3 $DIR/agentcore_slime_demo/build_real_datasets.py"
mkdir -p "$DIR/data"
$B --dataset sql --split train --limit 300 --out "$DIR/data/gretelsql_train.jsonl"
$B --dataset sql --split test --limit 40  --out "$DIR/data/gretelsql_eval.jsonl"
echo "done -> $DIR/data : gretelsql (300/40)"
