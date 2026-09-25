#!/bin/bash
# Build this demo's KernelBench train/eval data into data/ (run INSIDE agentcore_p4 — has `datasets` +
# the HF cache):
#     docker exec agentcore_p4 bash /work/code/demos/demo_gpu_farm_fractional/prepare_data.sh
#
# The data is SHIPPED (data/kernelbench_{train,eval}.jsonl), so by default this only verifies it.
# REBUILD=1 re-downloads ScalingIntelligence/KernelBench from HF and rewrites both files:
#   train = level_1 (100 single-op problems), eval = level_2[:20] (held-out fusions), max_turns=3.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "${REBUILD:-0}" = "1" ] || [ ! -s "$DIR/data/kernelbench_train.jsonl" ]; then
  python3 "$DIR/build_dataset.py" --dataset kernelbench --level level_1 --limit 100 --max-turns 3 \
      --out "$DIR/data/kernelbench_train.jsonl"
  python3 "$DIR/build_dataset.py" --dataset kernelbench --level level_2 --limit 20 --max-turns 3 \
      --out "$DIR/data/kernelbench_eval.jsonl"
fi
wc -l "$DIR/data/kernelbench_train.jsonl" "$DIR/data/kernelbench_eval.jsonl"
