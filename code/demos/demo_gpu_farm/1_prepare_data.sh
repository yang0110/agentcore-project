#!/bin/bash
# Step 1/6 — download + prepare this demo's dataset into data/. Runs INSIDE agentcore_p4 (docker exec'd
# from the host automatically). Knobs are forwarded, e.g. DATASET=gsm8k bash 1_prepare_data.sh.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/deploy/steps_common.sh"

log "step 1/6: prepare data  ($C_DEMO_DIR/data)"
[ -n "${PREPARE_PRE:-}" ] && in_container "$PREPARE_PRE"
in_container "bash $C_DEMO_DIR/prepare_data.sh"
[ -n "${PREPARE_POST:-}" ] && in_container "$PREPARE_POST"
in_container "wc -l $C_DEMO_DIR/data/*.jsonl | tail -n 12"
log "done — next: bash 2_build_image.sh"
