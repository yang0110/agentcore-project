#!/bin/bash
# Step 4/6 — make every task routable AND check it where it will train (HOST, real AWS $):
#   4a  register one VPC AgentCore Runtime per pushed microVM image (idempotent) -> arn: routes
#   4g  the g6 GPU tier (one session = one GPU instance): the KernelBench Runtime (TB_KERNELBENCH=1,
#       default) and any TB_GPU_TASKS=a,b -> gpu: routes. No Terminal-Bench task itself needs a GPU.
#   4b  validate_oracle.py on every route: the reference solution must pass and the untouched env fail
#   4c  microVM tasks whose oracle failed are retried on the m5 (deploy_docker_tasks.py), then 4d
#       re-validates the m5 tasks. A task keeps the route it passed on; tasks/oracle.csv says which.
# TB_VALIDATE=0 skips 4b-4d (then step 1 trains on every routed task, unvalidated).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/deploy/steps_common.sh"
need_host
cd "$HERE"
SUBS="${TB_SUBSTRATES:-microvm,docker}"
if [[ ",$SUBS," == *,microvm,* ]]; then
  log "step 4a: register the microVM task Runtimes"
  python3 terminalbench/deploy_terminalbench.py
fi
if [ "${TB_KERNELBENCH:-1}" = 1 ] || [ -n "${TB_GPU_TASKS:-}" ]; then
  log "step 4g: the GPU tier (g6 capacity provider): KernelBench${TB_GPU_TASKS:+ + $TB_GPU_TASKS}"
  python3 terminalbench/deploy_gpu_tasks.py
fi
if [[ ",${HARNESS_SET:-mini_claude}," == *claude_cli* ]] || [[ ",${HARNESS_SET:-mini_claude}," == *mini_claude_rt* ]]; then
  log "step 4h: build + deploy the DECOUPLED harness Runtime (its own microVM; serves claude_cli + mini_claude_rt)"
  HARNESS_REPO="${NAME_PREFIX}-harness-cli"
  HARNESS_URI="$(python3 deploy/agentcore_deploy.py ecr-uri --repo "$HARNESS_REPO")"
  PYTHONPATH=deploy python3 -c "import agentcore_deploy as d; d._ensure_buildx()"
  # buildx pushes to ECR, so authenticate first (the token expires ~12h; refresh every run).
  aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "${HARNESS_URI%%/*}"
  # Build context is the demo dir so agentcore_slime_demo/ is COPY-able; arm64 (serverless fleet).
  docker buildx build --builder "$BUILDX_BUILDER" --platform linux/arm64 \
     -f deploy/Dockerfile.harness.cli -t "${HARNESS_URI}:latest" --push "$HERE"
  SZ="$(docker buildx imagetools inspect "${HARNESS_URI}:latest" 2>/dev/null | awk '/Total Size|Size:/{print; exit}')"
  log "harness-cli image pushed${SZ:+ ($SZ)} — must be < 2 GB (microVM cap)"
  HARNESS_ARN="$(python3 deploy/agentcore_deploy.py runtime --name "${NAME_PREFIX}_harness_cli" \
     --image "${HARNESS_URI}:latest" --network VPC --egress nat \
     --env AGENTCORE_STOP_SESSION=0 --env SANDBOX_WORKDIR=/workspace | tail -n1)"
  python3 -c "import json,sys; json.dump({'arn': sys.argv[1]}, open('deploy/harness_cli.json','w'))" "$HARNESS_ARN"
  log "harness CLI Runtime: $HARNESS_ARN -> deploy/harness_cli.json"
fi
if [ "${TB_VALIDATE:-1}" = 1 ]; then
  log "step 4b: validate every task's oracle on its substrate"
  AGENTCORE_REGION="$AWS_REGION" python3 terminalbench/validate_oracle.py
  if [[ ",$SUBS," == *,docker,* ]]; then
    log "step 4c: retry the microVM oracle failures on the m5"
    python3 terminalbench/deploy_docker_tasks.py || echo "!! no m5 task built in 4c (see images_docker.csv)"
    log "step 4d: validate the m5 tasks"
    TB_SUBSTRATE=docker AGENTCORE_REGION="$AWS_REGION" python3 terminalbench/validate_oracle.py
  fi
fi
log "done — $(python3 -c "
import json; m = json.load(open('$POOL_MAP')); d = sum(str(v).startswith('docker://') for v in m.values())
g = sum(str(v).startswith('gpu:') for v in m.values())
print(f'{len(m) - d - g} microVM Runtimes + {d} m5 tasks + {g} g6 GPU Runtimes routed')"); next: bash 4b_hyperpod_up.sh, then bash 1_prepare_data.sh"
