#!/bin/bash
# Step 6/6 — delete every task Runtime in the pool map (HOST). ECR images, data/, logs/ and ckpt/ are kept.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/deploy/steps_common.sh"
need_host
[ -f "$POOL_MAP" ] || die "no $POOL_MAP — nothing to delete"
log "step 6/6: delete the task Runtimes"
python3 - "$POOL_MAP" <<'PY'
import json, sys
sys.path.insert(0, 'deploy')
import boto3
import agentcore_deploy as d
c = boto3.client('bedrock-agentcore-control', region_name=d.REGION)
for task, arn in json.load(open(sys.argv[1])).items():
    try: c.delete_agent_runtime(agentRuntimeId=arn.rsplit('/', 1)[1]); d.err(f'deleted {task}')
    except Exception as e: d.err(f'{task}: {e}')
PY
echo "kept: ECR repos ${NAME_PREFIX}-tb-* — delete with: aws ecr delete-repository --force --repository-name <repo>"
echo "kept: data/, logs/, ckpt/ (checkpoints are ~56 GB each; remove ckpt/<run> from inside $CONTAINER)"
