#!/bin/bash
# Build demo_cpu_terminalbench's train/eval dataset into data/ (offline, stdlib-only — no net/GPU/AWS).
#
# Run INSIDE the persistent container agentcore_p4 (slimerl/slime). From the repo root on the host:
#     docker exec agentcore_p4 bash /work/code/demos/demo_cpu_terminalbench/prepare_data.sh
#
# Turns tasks/manifest.json (from fetch_terminalbench.py) into ONE self-describing row per
# Terminal-Bench task: metadata.task = {family:"terminalbench", task_id, prompt, spec:{image,...}}.
# Each row's spec.image (the task name) routes its episode to that task's OWN Runtime ARN via
# AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE — the large image pool this demo showcases. Reward is the
# task's baked hidden pytest oracle (grade.sh); the sandbox stays task-agnostic. Self-contained:
# imports only this dir's vendored agentcore_slime_demo/. Idempotent (overwrites).
#
# Knobs: LIMIT=N (first N tasks), REPEAT=k (repeat each task k times in train), TB_MAX_TURNS=t.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ARGS=()
[ -n "${LIMIT:-}" ]  && ARGS+=(--limit "$LIMIT")
[ -n "${REPEAT:-}" ] && ARGS+=(--repeat "$REPEAT")

python3 "$DIR/build_terminalbench_dataset.py" --out-dir "$DIR/data" "${ARGS[@]}"

echo "done -> $DIR/data : tb_train.jsonl + tb_eval.jsonl"
