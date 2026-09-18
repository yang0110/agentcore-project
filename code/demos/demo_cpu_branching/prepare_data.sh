#!/bin/bash
# Build demo_cpu_branching's single-family SWE dataset into data/ (offline, stdlib-only — no network,
# no GPU).
#
# Run INSIDE the persistent container agentcore_p4 (slimerl/slime). From the repo root on the host:
#     docker exec agentcore_p4 bash /work/code/demos/demo_cpu_branching/prepare_data.sh
#
# Produces swe_train.jsonl + swe_eval.jsonl — every row a SWE bug-fix task carrying its own hidden
# RLVR checker in metadata.task. Unlike demo_cpu_mixtask this is ONE family: the interesting variable
# here is the FAN-OUT (one episode -> many trainable leaves via the real Claude Code CLI), not task
# type. Fully self-contained: imports only this dir's vendored agentcore_slime_demo/. Idempotent.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REPEAT="${REPEAT:-8}"

python3 "$DIR/build_swe_dataset.py" \
   --out-dir "$DIR/data" \
   --repeat "$REPEAT"

echo "done -> $DIR/data : swe_train.jsonl + swe_eval.jsonl"
