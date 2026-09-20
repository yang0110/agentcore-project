#!/bin/bash
# Build demo_cpu_heartbeat's CPU training dataset into data/ (deepscaler numeric-answer math, RLVR:
# numeric match tol 1e-4 — no LLM judge). Run INSIDE the persistent container agentcore_p4:
#     docker exec agentcore_p4 bash /work/code/demos/demo_cpu_heartbeat/prepare_data.sh
#
# The builder is vendored (agentcore_slime_demo/build_real_datasets.py) so this folder is
# self-contained. The HF pull needs one-time network; if it is unavailable (fully offline), fall
# back to the deepscaler JSONL already shipped in the sibling demo_cpu_harness_scale/data so the
# trainer still has data. Idempotent (overwrites). Builds deepscaler (400 train / 40 eval).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
B="python3 $DIR/agentcore_slime_demo/build_real_datasets.py"
mkdir -p "$DIR/data"

if $B --dataset deepscaler --limit 400 --offset 0 --out "$DIR/data/deepscaler_train.jsonl" \
   && $B --dataset deepscaler --limit 40 --offset 400 --out "$DIR/data/deepscaler_eval.jsonl"; then
  echo "done (built from HF) -> $DIR/data : deepscaler (400/40)"
else
  echo "HF build failed (offline?) — falling back to sibling demo_cpu_harness_scale/data" >&2
  cp "$DIR/../demo_cpu_harness_scale/data/deepscaler_train.jsonl" "$DIR/data/deepscaler_train.jsonl"
  cp "$DIR/../demo_cpu_harness_scale/data/deepscaler_eval.jsonl"  "$DIR/data/deepscaler_eval.jsonl"
  echo "done (copied fallback) -> $DIR/data : deepscaler (400/40)"
fi
