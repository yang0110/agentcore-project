#!/bin/bash
# Build demo_hybrid_terminalbench's train/eval rows into data/ (INSIDE agentcore_p4; 1_prepare_data.sh execs it).
# ONE row per Terminal-Bench task (laude-institute/terminal-bench), each naming its OWN image; built from
# terminalbench/tasks/manifest.json (fetched once, needs network). Rows are kept only for tasks whose
# route is deployed + oracle-verified (microVM or m5), so re-run this AFTER steps 2-4.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$DIR/data"
mkdir -p "$OUT"
[ -f "$DIR/terminalbench/tasks/manifest.json" ] || python3 "$DIR/terminalbench/fetch_terminalbench.py"
TB=(); [ -n "${LIMIT:-}" ] && TB+=(--limit "$LIMIT"); [ -n "${REPEAT:-}" ] && TB+=(--repeat "$REPEAT")
python3 "$DIR/terminalbench/build_terminalbench_dataset.py" --out-dir "$OUT" "${TB[@]}"
# Drop train tasks whose reward never varied in earlier runs (no GRPO gradient); no logs = keep all.
if [ "${TB_FILTER_HISTORY:-1}" = 1 ]; then
  python3 "$DIR/terminalbench/filter_by_history.py" --train "$OUT/terminalbench_train.jsonl" --logs "$DIR/logs"
fi
