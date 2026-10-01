#!/bin/bash
# Step 3/6 — report only: step 2 already pushed every image to ECR. Shows terminalbench/tasks/images.csv.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/deploy/steps_common.sh"
log "step 3/6: images (pushed by step 2)"
column -s, -t < "$HERE/terminalbench/tasks/images.csv" | sed -n 1,40p
log "done — next: bash 4_deploy_runtime.sh"
