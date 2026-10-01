#!/bin/bash
# Step 2/6 — fetch the Terminal-Bench corpus (once) and build + push one linux/arm64 image per task (HOST;
# resumable, ~2 h cold). Images over the 2048 MB microVM cap, or that fail to build, are recorded in
# terminalbench/tasks/images.csv and left out of the pool.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/deploy/steps_common.sh"
need_host
cd "$HERE"
[ -f terminalbench/tasks/manifest.json ] || python3 terminalbench/fetch_terminalbench.py
log "step 2/6: build + push the task images (no Runtimes yet)"
TB_NO_REGISTER=1 python3 terminalbench/deploy_terminalbench.py
log "done — next: bash 3_push_ecr.sh"
