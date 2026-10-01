#!/bin/bash
# Step 4/6 — register one VPC AgentCore Runtime per pushed task image (HOST, real AWS $). Idempotent:
# existing Runtimes are reused. Writes the pool map terminalbench/tasks/arn_by_image.json.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/deploy/steps_common.sh"
need_host
log "step 4/6: register the task Runtimes"
(cd "$HERE" && python3 terminalbench/deploy_terminalbench.py)
log "done — $(python3 -c "import json;print(len(json.load(open('$POOL_MAP'))))") Runtimes in the pool; next: bash 1_prepare_data.sh"
