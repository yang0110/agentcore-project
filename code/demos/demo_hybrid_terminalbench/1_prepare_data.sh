#!/bin/bash
# Step 1/6 — build the train/eval rows into data/ (INSIDE agentcore_p4, docker exec'd from the host
# automatically). Keeps only tasks whose Runtime is deployed + oracle-verified, so re-run after step 4.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/deploy/steps_common.sh"
log "step 1/6: prepare data  ($C_DEMO_DIR/data)"
in_container "bash $C_DEMO_DIR/prepare_data.sh"
in_container "wc -l $C_DEMO_DIR/data/*.jsonl"
log "done — next: bash 5_run.sh (or 2_build_image.sh if the pool is not built yet)"
