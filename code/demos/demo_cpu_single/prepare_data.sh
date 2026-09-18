#!/bin/bash
# Download + prepare demo_cpu_single's REAL datasets into data/ (coding + the CPU task zoo).
#
# Run INSIDE the persistent container agentcore_p4 (has `datasets`; needs one-time network to pull
# the HF rows). From the repo root on the host:
#     docker exec agentcore_p4 bash /work/code/demos/demo_cpu_single/prepare_data.sh
#
# Every dataset is RLVR (verifiable reward): a hidden, exact offline checker that runs in the stdlib
# CPU sandbox — no LLM judge. Two vendored builders keep this folder self-contained:
#   * build_dataset.py           -> MBPP (its own loader)
#   * build_real_datasets.py     -> the HuggingFace real-dataset zoo (humaneval/sql/rag/tool_use)
# Idempotent: re-running overwrites the JSONL files.
#
# NAMING CONTRACT: each file's prefix == run.sh's FAMILY label, so after this you train a task with a
# plain `FAMILY=<name> bash run.sh` (no DATA/EVAL_DATA overrides). Reproduces the exact row counts on
# disk: mbpp 500/8, humaneval 140/24, gretelsql 300/40, hotpotqa 300/40, xlam 300/40.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REAL="python3 $DIR/agentcore_slime_demo/build_real_datasets.py"
MBPP="python3 $DIR/build_dataset.py"
OUT="$DIR/data"
mkdir -p "$OUT"

# coding — MBPP (mostly-basic Python problems, hidden asserts). FAMILY=coding (alias mbpp).
$MBPP --dataset mbpp --split "train+test" --limit 500 --out "$OUT/mbpp_train.jsonl"
$MBPP --dataset mbpp --split validation   --limit 8   --out "$OUT/mbpp_eval.jsonl"

# humaneval — openai/openai_humaneval; the row's own `test` fn is the checker (check(entry_point)).
# Only a `test` split (164 rows), so train/eval are carved by --offset (0..139 train, 140..163 eval).
# FAMILY=humaneval
$REAL --dataset humaneval --split test --limit 140 --offset 0   --out "$OUT/humaneval_train.jsonl"
$REAL --dataset humaneval --split test --limit 24  --offset 140 --out "$OUT/humaneval_eval.jsonl"

# sql — gretelai/synthetic_text_to_sql; the row embeds CREATE+INSERT, we materialize the DB at build
# time, run the gold query for the expected result set, and the checker execution-verifies the agent's
# query (stdlib sqlite3). Disjoint train/test splits.  FAMILY=gretelsql
$REAL --dataset sql --split train --limit 300 --out "$OUT/gretelsql_train.jsonl"
$REAL --dataset sql --split test  --limit 40  --out "$OUT/gretelsql_eval.jsonl"

# rag — hotpotqa/hotpot_qa (distractor); the 10 passages are written into ./docs as the corpus and
# the agent must read them. Reward = SQuAD-normalized exact match of the answer. Disjoint
# train/validation splits.  FAMILY=hotpotqa
$REAL --dataset hotpotqa --split train      --limit 300 --out "$OUT/hotpotqa_train.jsonl"
$REAL --dataset hotpotqa --split validation --limit 40  --out "$OUT/hotpotqa_eval.jsonl"

# tool_use — minpeter/xlam-function-calling-60k-parsed (open mirror of Salesforce/xLAM); reward =
# order-insensitive, arg-normalized exact match of the agent's emitted `calls` vs the gold calls.
# Only a `train` split, so eval is carved by --offset.  FAMILY=xlam
$REAL --dataset xlam --limit 300 --offset 0   --out "$OUT/xlam_train.jsonl"
$REAL --dataset xlam --limit 40  --offset 300 --out "$OUT/xlam_eval.jsonl"

# mcp — Team-ACE/ToolACE, MCP-style JSON tool schemas + gold calls. Same verifiable ToolCallTask
# checker as xlam, different (realistic-API) tool source. Only a `train` split → eval by --offset.
# FAMILY=mcp
$REAL --dataset mcp --limit 300 --offset 0   --out "$OUT/mcp_train.jsonl"
$REAL --dataset mcp --limit 40  --offset 300 --out "$OUT/mcp_eval.jsonl"

echo "done -> $OUT : mbpp (500/8), humaneval (140/24), gretelsql (300/40), hotpotqa (300/40), xlam (300/40), mcp (300/40)"
