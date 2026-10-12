#!/bin/bash
# Step 6/6 (HOST) — delete the HyperPod cluster, then its FSx (checkpoints on it go too; KEEP_FSX=1 keeps
# it) and the cluster SG + its rules. The S3 staging bucket, the node role and the trainer-image ECR repo
# are kept.
# The sandbox tiers (microVM Runtimes ${NAME_PREFIX}_tbh_*, the GPU capacity providers, the m5 host, the
# plain-EC2 g6 pool) are the SAME AWS resources demo_terminalbench_kernelbench uses, so they are only
# removed with CLEAN_SANDBOXES=1 (then exactly as that demo's 6_cleanup.sh: KEEP_M5=1 keeps the m5).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/deploy/steps_common.sh"
need_host
cd "$HERE"
if [ -f deploy/hyperpod/hyperpod.json ]; then
  log "step 6/6: delete the HyperPod cluster + FSx + cluster SG"
  python3 deploy/hyperpod/hyperpod.py down
fi
if [ "${CLEAN_SANDBOXES:-0}" != 1 ]; then
  echo "kept: the sandbox tiers (shared with demo_terminalbench_kernelbench) — CLEAN_SANDBOXES=1 bash 6_cleanup.sh removes them"
  exit 0
fi
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
import re, sys, time
sys.path.insert(0, 'deploy')
import boto3
import agentcore_deploy as d
# Every GPU-tier provider: <prefix>_tbh_gpu_cp, plus one per instance-type list (deploy_gpu_tasks.cp_name).
mine = re.compile(rf"^{re.escape(d.PREFIX)}_tbh_gpu(_[a-z0-9]+)*_cp$")
c = boto3.client("bedrock-agentcore-control", region_name=d.REGION)
names = sorted(cp["name"] for cp in c.list_capacity_providers().get("capacityProviders", []) if mine.match(cp.get("name", "")))
for name in names:
    for _ in range(30):                       # a CP can only go once its Runtimes are gone
        try:
            d.delete_capacity_provider(name); break
        except Exception as e:  # noqa: BLE001
            d.err(f"{name}: {e}; retrying"); time.sleep(20)
if not names:
    d.err("no GPU capacity provider to delete")
PY
fi
if [ -f "$HERE/deploy/harness_cli.json" ]; then
  log "delete the decoupled Claude Code CLI harness Runtime"
  python3 - "$HERE/deploy/harness_cli.json" <<'PY'
import json, sys
sys.path.insert(0, 'deploy')
import boto3
import agentcore_deploy as d
arn = json.load(open(sys.argv[1])).get('arn', '')
rid = arn.rsplit('/', 1)[-1]
if rid:
    c = boto3.client('bedrock-agentcore-control', region_name=d.REGION)
    try: c.delete_agent_runtime(agentRuntimeId=rid); d.err(f'deleted harness-cli {rid}')
    except Exception as e: d.err(f'harness-cli {rid}: {e}')
PY
  rm -f "$HERE/deploy/harness_cli.json"
fi
if [ -f "$M5_STATE" ]; then
  log "terminate any per-episode m5 instances"
  python3 deploy/m5_host.py gc-episodes
fi
if [ -f "$HERE/deploy/g6_host.json" ]; then
  log "terminate the plain-EC2 g6 instances + drop their SG (TB_KB_SUBSTRATE=ec2 tier)"
  python3 deploy/g6_host.py down
fi
if [ "${KEEP_M5:-0}" != 1 ] && [ -f "$M5_STATE" ]; then
  log "terminate the m5 Docker host (builder)"
  python3 deploy/m5_host.py down
fi
echo "kept: ECR repos ${NAME_PREFIX}-tb-* (microVM images), ${NAME_PREFIX}-tbhdk-* (m5 images) and ${NAME_PREFIX}-tbgpu-* (g6 images) — delete with: aws ecr delete-repository --force --repository-name <repo>"
echo "kept: IAM roles ${NAME_PREFIX}-exec-role, ${NAME_PREFIX}-operator-role, ${NAME_PREFIX}-tbh-m5-builder, ${NAME_PREFIX}-tbh-m5-episode (+ instance profiles); CloudWatch log groups /aws/bedrock-agentcore/runtimes/* (no retention set)"
echo "kept: the plain-EC2 g6 AMI (tag Name=${NAME_PREFIX}-tbh-g6ec2) and its image ECR ${NAME_PREFIX}-tbhdk-kernelbench-gpu-client"
echo "kept: the per-episode AMI (tag Name=${NAME_PREFIX}-tbh-m5-ep) — deregister with: aws ec2 deregister-image --image-id <ami>"
echo "kept: data/, logs/, ckpt/ (checkpoints are ~56 GB each; remove ckpt/<run> from inside $CONTAINER)"
du -sh "$HERE/ckpt" 2>/dev/null | awk '{print "      ckpt/ now holds " $1}' || true
