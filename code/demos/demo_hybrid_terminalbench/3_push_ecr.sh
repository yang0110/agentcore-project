#!/bin/bash
# Step 3/6 — report only: step 2 already pushed every microVM image to ECR and built the m5 images on the m5.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/deploy/steps_common.sh"
log "step 3/6: microVM images (pushed to ECR by step 2a)"
column -s, -t < "$HERE/terminalbench/tasks/images.csv" | cut -c1-160 | sed -n 1,40p
if [ -f "$HERE/terminalbench/tasks/images_docker.csv" ]; then
  log "m5 images (built on the m5 by step 2c)"
  column -s, -t < "$HERE/terminalbench/tasks/images_docker.csv" | cut -c1-160
fi
log "done — next: bash 4_deploy_runtime.sh"
