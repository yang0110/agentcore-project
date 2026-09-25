#!/bin/bash
# Step 6/6 — tear down what step 4 created (HOST): delete this demo's AgentCore Runtime(s) (and its
# capacity provider if DELETE_CP=1), or stop the local GPU farm. ECR repos/images and data are kept.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/deploy/steps_common.sh"
need_host
load_env "$RUNTIME_ENV"

if [ "$SUBSTRATE" = local-gpu ] || [ "${LOCAL_GPU:-0}" = 1 ]; then
  log "step 6/6: stop the local GPU farm"; bash "$HERE/deploy/local_gpu_runtime.sh" stop
elif [ -n "${CLEANUP_CMD:-}" ]; then
  log "step 6/6: $CLEANUP_CMD"; (cd "$HERE" && eval "$CLEANUP_CMD")
else
  log "step 6/6: delete Runtime(s)"
  for name in $(image_names); do python3 "$HERE/deploy/agentcore_deploy.py" delete-runtime --name "$(runtime_name "$name")"; done
fi
if [ "${DELETE_CP:-0}" = 1 ] && [ -n "${CP_NAME:-}" ]; then
  python3 "$HERE/deploy/agentcore_deploy.py" delete-cp --name "$CP_NAME"
fi
rm -f "$RUNTIME_ENV"
echo "kept: ECR repos ($(image_names | sed "s/^/$REPO_PREFIX/" | tr '\n' ' ')) — delete with: aws ecr delete-repository --force --repository-name <repo>"
echo "kept: data/, logs/, ckpt/ (checkpoints are ~56 GB each; remove ckpt/<run> by hand from inside $CONTAINER)"
