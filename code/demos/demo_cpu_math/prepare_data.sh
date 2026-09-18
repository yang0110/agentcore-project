#!/bin/bash
# Download + prepare demo_cpu_math's REAL datasets (HuggingFace math corpora) into data/.
#
# Run INSIDE the persistent container agentcore_p4 (has `datasets`; needs one-time network to pull
# the HF rows). From the repo root on the host:
#     docker exec agentcore_p4 bash /work/code/demos/demo_cpu_math/prepare_data.sh
#
# Every dataset is RLVR (verifiable reward): a numeric-answer, exact offline checker that runs in the
# stdlib CPU sandbox — no LLM judge. The builder is vendored (agentcore_slime_demo/build_real_datasets.py),
# so this demo folder is self-contained. Idempotent: re-running overwrites the JSONL files.
#
# NAMING CONTRACT: each file's prefix == run.sh's FAMILY label, so after this you train a task with a
# plain `FAMILY=<name> bash run.sh` (no DATA/EVAL_DATA overrides). Reproduces the exact row counts on
# disk: gsm8k 200/40, math500 160/40, deepscaler 400/40.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="python3 $DIR/agentcore_slime_demo/build_real_datasets.py"
OUT="$DIR/data"
mkdir -p "$OUT"

# GSM8K — grade-school word problems (openai/gsm8k, config "main"); gold is an integer (exact match).
# Disjoint train/test splits.  FAMILY=gsm8k
$BUILD --dataset gsm8k --split train --limit 200 --out "$OUT/gsm8k_train.jsonl"
$BUILD --dataset gsm8k --split test  --limit 40  --out "$OUT/gsm8k_eval.jsonl"

# MATH-500 — competition math (HuggingFaceH4/MATH-500), numeric-answer subset (tol 1e-4). Only a
# `test` split exists, so train/eval are carved by --offset (0..159 train, 160..199 eval).  FAMILY=math500
$BUILD --dataset math500 --split test --limit 160 --offset 0   --out "$OUT/math500_train.jsonl"
$BUILD --dataset math500 --split test --limit 40  --offset 160 --out "$OUT/math500_eval.jsonl"

# DeepScaleR — competition/olympiad math (agentica-org/DeepScaleR-Preview-Dataset), numeric-answer
# subset. Only a `train` split exists, so eval is carved by --offset.  FAMILY=deepscaler
$BUILD --dataset deepscaler --limit 400 --offset 0   --out "$OUT/deepscaler_train.jsonl"
$BUILD --dataset deepscaler --limit 40  --offset 400 --out "$OUT/deepscaler_eval.jsonl"

echo "done -> $OUT : gsm8k (200/40), math500 (160/40), deepscaler (400/40)"
