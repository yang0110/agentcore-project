#!/bin/bash
# Step 6/6 — delete this demo's microVM Runtimes (${NAME_PREFIX}_tbh_* only: demo_cpu_terminalbench's
# acs_tb_* are never touched) and terminate the m5 host + its SG (HOST). ECR images (shared with
# demo_cpu_terminalbench), data/, logs/ and ckpt/ are kept. KEEP_M5=1 keeps the m5 (images stay built).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/deploy/steps_common.sh"
need_host
cd "$HERE"
if [ -f "$POOL_MAP" ]; then
  log "step 6/6: delete the microVM task Runtimes"
  python3 - "$POOL_MAP" "$HERE/terminalbench/tasks/docker_fallback.json" "${TB_RUNTIME_TAG:-tbh}" <<'PY'
import json, os, sys
sys.path.insert(0, 'deploy')
import boto3
import agentcore_deploy as d
routes = dict(json.load(open(sys.argv[1])))
if os.path.exists(sys.argv[2]):              # microVM Runtimes of tasks that moved to the m5
    routes.update({f"{k} (fallback)": v for k, v in json.load(open(sys.argv[2])).items()})
mine = f"{d.PREFIX}_{sys.argv[3]}_"
c = boto3.client('bedrock-agentcore-control', region_name=d.REGION)
for task, arn in sorted(routes.items()):
    arn = str(arn).removeprefix('gpu:')       # GPU-tier Runtimes are <prefix>_<tag>gpu_<task>
    rid = arn.rsplit('/', 1)[-1]
    if not arn.startswith('arn:') or not rid.startswith((mine, mine[:-1] + 'gpu_')):
        continue                              # an m5 route, or not this demo's Runtime
    try: c.delete_agent_runtime(agentRuntimeId=rid); d.err(f'deleted {task}')
    except Exception as e: d.err(f'{task}: {e}')
PY
  python3 - <<'PY'
import sys, time
sys.path.insert(0, 'deploy')
import agentcore_deploy as d
name = f"{d.PREFIX}_tbh_gpu_cp"
for _ in range(30):                           # the CP can only go once its Runtimes are gone
    try:
        d.delete_capacity_provider(name); break
    except Exception as e:  # noqa: BLE001
        d.err(f"{name}: {e}; retrying"); time.sleep(20)
PY
fi
if [ -f "$M5_STATE" ]; then
  log "terminate any per-episode m5 instances"
  python3 deploy/m5_host.py gc-episodes
fi
if [ "${KEEP_M5:-0}" != 1 ] && [ -f "$M5_STATE" ]; then
  log "terminate the m5 Docker host (builder)"
  python3 deploy/m5_host.py down
fi
echo "kept: ECR repos ${NAME_PREFIX}-tb-* (shared with demo_cpu_terminalbench) and ${NAME_PREFIX}-tbhdk-* (m5 images) — delete with: aws ecr delete-repository --force --repository-name <repo>"
echo "kept: the per-episode AMI (tag Name=${NAME_PREFIX}-tbh-m5-ep) — deregister with: aws ec2 deregister-image --image-id <ami>"
echo "kept: data/, logs/, ckpt/ (checkpoints are ~56 GB each; remove ckpt/<run> from inside $CONTAINER)"
