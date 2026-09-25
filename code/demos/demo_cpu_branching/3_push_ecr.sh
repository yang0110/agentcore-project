#!/bin/bash
# Step 3/6 — push the image(s) built in step 2 to ECR (HOST). Creates the repos if missing and writes
# the pushed URIs to deploy/images.env for step 4.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/deploy/steps_common.sh"
need_host
load_env "$RUNTIME_ENV"

if [ -n "${PUSH_CMD:-}" ]; then log "step 3/6: $PUSH_CMD"; (cd "$HERE" && eval "$PUSH_CMD"); exit 0; fi
if [ "$SUBSTRATE" = local-gpu ] || [ "${LOCAL_GPU:-0}" = 1 ]; then
  log "step 3/6: local GPU farm — the image stays on this host, nothing to push"; exit 0
fi

ACCT="$(aws sts get-caller-identity --query Account --output text)"
REG="$ACCT.dkr.ecr.$AWS_REGION.amazonaws.com"
log "step 3/6: push to $REG"
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$REG"
: > "$IMAGES_ENV"
for name in $(image_names); do
  repo="$(image_repo "$name")"
  aws ecr describe-repositories --region "$AWS_REGION" --repository-names "$repo" >/dev/null 2>&1 || \
    aws ecr create-repository --region "$AWS_REGION" --repository-name "$repo" >/dev/null
  docker tag "$repo:latest" "$REG/$repo:latest"
  docker push "$REG/$repo:latest"
  echo "IMAGE_URI_$name=$REG/$repo:latest" >> "$IMAGES_ENV"
done
cat "$IMAGES_ENV"
log "done — next: bash 4_deploy_runtime.sh"
