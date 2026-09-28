#!/bin/bash
# Download + prepare demo_cpu_tasks' datasets into data/ — one folder, every dataset (+ the prebuilt mix).
#
# Run INSIDE the persistent container agentcore_p4 (has `datasets`; needs one-time network to pull
# the HF rows). From the repo root on the host:
#     docker exec agentcore_p4 bash /work/code/demos/demo_cpu_tasks/prepare_data.sh            # all
#     docker exec -e DATASET=gsm8k agentcore_p4 bash /work/code/demos/demo_cpu_tasks/prepare_data.sh
#
# Every dataset is RLVR (verifiable reward): a hidden, exact offline checker that runs in the stdlib
# CPU sandbox — no LLM judge. Two vendored builders keep this folder self-contained:
#   * build_dataset.py           -> MBPP (its own loader)
#   * build_real_datasets.py     -> the HuggingFace real-dataset zoo (math/humaneval/sql/rag/tool_use)
#   * build_mixed_datasets.py    -> DATASET=mixed (offline, stdlib: math+sql+ctf+datasci interleaved;
#                                   FAMILIES / REPEAT / BALANCED override the mix)
# (The synthetic task-family zoo — math swe tool_use sql ctf rag … — comes from build_family_datasets.py.)
# Ad-hoc mixes (DATASET=a,b,c) need no build step: run.sh interleaves the per-dataset files itself.
# Idempotent: re-running overwrites the JSONL files.
#
# NAMING CONTRACT: each file's prefix == run.sh's DATASET label, so after this you train a dataset
# with a plain `DATASET=<name> bash run.sh` (no DATA/EVAL_DATA overrides). Row counts (train/eval):
#   mbpp 500/8   gsm8k 200/40   math500 160/40   deepscaler 400/40   humaneval 140/24
#   gretelsql 300/40   hotpotqa 300/40   xlam 300/40   mcp 300/40   mixed 174/9   libs 72/6
#   terminalbench 135/31 (not in `all`: DATASET=terminalbench, after the per-task Runtimes are deployed)
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REAL="python3 $DIR/agentcore_slime_demo/build_real_datasets.py"
MBPP="python3 $DIR/build_dataset.py"
OUT="$DIR/data"
DATASET="${DATASET:-${1:-all}}"
ALL="mbpp gsm8k math500 deepscaler humaneval gretelsql hotpotqa xlam mcp mixed libs"
mkdir -p "$OUT"

build() {
  case "$1" in
    # coding — MBPP (mostly-basic Python problems, hidden asserts). DATASET=mbpp (alias coding).
    mbpp)
      $MBPP --dataset mbpp --split "train+test" --limit 500 --out "$OUT/mbpp_train.jsonl"
      $MBPP --dataset mbpp --split validation   --limit 8   --out "$OUT/mbpp_eval.jsonl" ;;
    # GSM8K — grade-school word problems (openai/gsm8k, config "main"); gold is an integer (exact
    # match). Disjoint train/test splits.
    gsm8k)
      $REAL --dataset gsm8k --split train --limit 200 --out "$OUT/gsm8k_train.jsonl"
      $REAL --dataset gsm8k --split test  --limit 40  --out "$OUT/gsm8k_eval.jsonl" ;;
    # MATH-500 — competition math (HuggingFaceH4/MATH-500), numeric-answer subset (tol 1e-4). Only a
    # `test` split exists, so train/eval are carved by --offset (0..159 train, 160..199 eval).
    math500)
      $REAL --dataset math500 --split test --limit 160 --offset 0   --out "$OUT/math500_train.jsonl"
      $REAL --dataset math500 --split test --limit 40  --offset 160 --out "$OUT/math500_eval.jsonl" ;;
    # DeepScaleR — competition/olympiad math (agentica-org/DeepScaleR-Preview-Dataset), numeric-answer
    # subset. Only a `train` split exists, so eval is carved by --offset.
    deepscaler)
      $REAL --dataset deepscaler --limit 400 --offset 0   --out "$OUT/deepscaler_train.jsonl"
      $REAL --dataset deepscaler --limit 40  --offset 400 --out "$OUT/deepscaler_eval.jsonl" ;;
    # humaneval — openai/openai_humaneval; the row's own `test` fn is the checker (check(entry_point)).
    # Only a `test` split (164 rows), so train/eval are carved by --offset (0..139 train, 140..163 eval).
    humaneval)
      $REAL --dataset humaneval --split test --limit 140 --offset 0   --out "$OUT/humaneval_train.jsonl"
      $REAL --dataset humaneval --split test --limit 24  --offset 140 --out "$OUT/humaneval_eval.jsonl" ;;
    # sql — gretelai/synthetic_text_to_sql; the row embeds CREATE+INSERT, we materialize the DB at
    # build time, run the gold query for the expected result set, and the checker execution-verifies
    # the agent's query (stdlib sqlite3). Disjoint train/test splits.
    gretelsql)
      $REAL --dataset sql --split train --limit 300 --out "$OUT/gretelsql_train.jsonl"
      $REAL --dataset sql --split test  --limit 40  --out "$OUT/gretelsql_eval.jsonl" ;;
    # rag — hotpotqa/hotpot_qa (distractor); the 10 passages are written into ./docs as the corpus and
    # the agent must read them. Reward = SQuAD-normalized exact match. Disjoint train/validation.
    hotpotqa)
      $REAL --dataset hotpotqa --split train      --limit 300 --out "$OUT/hotpotqa_train.jsonl"
      $REAL --dataset hotpotqa --split validation --limit 40  --out "$OUT/hotpotqa_eval.jsonl" ;;
    # tool_use — minpeter/xlam-function-calling-60k-parsed (open mirror of Salesforce/xLAM); reward =
    # order-insensitive, arg-normalized exact match of the emitted `calls` vs the gold calls. Only a
    # `train` split, so eval is carved by --offset.
    xlam)
      $REAL --dataset xlam --limit 300 --offset 0   --out "$OUT/xlam_train.jsonl"
      $REAL --dataset xlam --limit 40  --offset 300 --out "$OUT/xlam_eval.jsonl" ;;
    # mcp — Team-ACE/ToolACE, MCP-style JSON tool schemas + gold calls. Same verifiable ToolCallTask
    # checker as xlam, different (realistic-API) tool source. Only a `train` split -> eval by --offset.
    mcp)
      $REAL --dataset mcp --limit 300 --offset 0   --out "$OUT/mcp_train.jsonl"
      $REAL --dataset mcp --limit 40  --offset 300 --out "$OUT/mcp_eval.jsonl" ;;
    # mixed — the prebuilt multi-family mix (every GRPO step spans families). datasci needs pandas:
    # those rows route to the "libs" Runtime (deploy/Dockerfile.sandbox.libs). Per-family evals land as
    # mixed_<fam>_eval.jsonl.
    mixed)
      BAL=(); [ "${BALANCED:-1}" = "1" ] && BAL=(--balanced)
      python3 "$DIR/build_mixed_datasets.py" --families "${FAMILIES:-math,sql,ctf,datasci}" \
          --out-dir "$OUT" --repeat "${REPEAT:-6}" "${BAL[@]}" ;;
    # libs — datasci+scipy+sklearn (pandas / scipy / scikit-learn tasks; was the custom-image pool).
    # All three run on the "libs" Runtime (deploy/Dockerfile.sandbox.libs; image_routing.py maps the tags).
    libs)
      BAL=(); [ "${BALANCED:-1}" = "1" ] && BAL=(--balanced)
      python3 "$DIR/build_mixed_datasets.py" --name libs --families "${FAMILIES:-datasci,scipy,sklearn}" \
          --out-dir "$OUT" --repeat "${REPEAT:-6}" "${BAL[@]}" ;;
    # terminalbench — Terminal-Bench (laude-institute/terminal-bench): ONE row per task, each with its OWN
    # image (row image = task name -> that task's Runtime), graded by its hidden pytest oracle. Offline
    # from terminalbench/tasks/manifest.json (fetched once, needs network); rows are kept only for tasks
    # whose Runtime is deployed + oracle-verified, so build it AFTER `VARIANT=terminalbench` steps 2-4.
    # Not in ALL (needs git + the per-task pool). LIMIT=N / REPEAT=k.
    terminalbench)
      [ -f "$DIR/terminalbench/tasks/manifest.json" ] || python3 "$DIR/terminalbench/fetch_terminalbench.py"
      TB=(); [ -n "${LIMIT:-}" ] && TB+=(--limit "$LIMIT"); [ -n "${REPEAT:-}" ] && TB+=(--repeat "$REPEAT")
      python3 "$DIR/terminalbench/build_terminalbench_dataset.py" --out-dir "$OUT" "${TB[@]}" ;;
    *) # synthetic task-family zoo (math swe sql …): committed in data/, built by build_family_datasets.py
       if [ -f "$OUT/$1_train.jsonl" ]; then echo "$1: synthetic zoo dataset, already in $OUT (skip)"; return 0; fi
       echo "unknown DATASET=$1 (one of: all $ALL mixed, or a synthetic zoo name in data/)" >&2; exit 2 ;;
  esac
  echo "done -> $OUT/$1_{train,eval}.jsonl"
}

[ "$DATASET" = "coding" ] && DATASET=mbpp
if [ "$DATASET" = "all" ]; then for ds in $ALL; do build "$ds"; done; else build "$DATASET"; fi
