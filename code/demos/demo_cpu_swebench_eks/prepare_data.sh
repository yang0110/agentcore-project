#!/bin/bash
# Build demo_cpu_swebench's train/eval dataset into data/ (offline, stdlib-only — no net/GPU/AWS).
#
#     docker exec agentcore_p4 bash /work/code/demos/demo_cpu_swebench/prepare_data.sh
#
# Turns swebench/manifest.json (from fetch_swebench.py) into ONE self-describing row per SWE-bench
# instance (family "swebench"). Only instances in swebench/arn_by_image.json (once deployed) and, if
# swebench/oracle.csv exists, only usable ones (gold solves, empty patch does not) are kept.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$DIR/fetch_swebench.py" --data-only
echo "done -> $DIR/data : swebench_train.jsonl + swebench_eval.jsonl"
