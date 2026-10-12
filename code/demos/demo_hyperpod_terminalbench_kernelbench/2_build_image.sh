#!/bin/bash
# Step 2/6 — fetch the WHOLE Terminal-Bench corpus (once), then build every task's sandbox on its substrate (HOST):
#   2a  microVM tasks: one linux/arm64 image per single-container task, pushed to ECR (resumable; images
#       the microVM-only demo already pushed are reused). Images over the 2048 MB cap or that fail to build
#       are recorded in terminalbench/tasks/images.csv and handed to the m5.
#   2b  the m5 Docker host (deploy/m5_host.py up: SGs, EC2 instance, Docker, the compose agent; real AWS $).
#   2c  m5 tasks: compose + the microVM rejects, built natively for linux/amd64 on the m5 (the builder) and
#       pushed to ECR (<prefix>-tbhdk-<task>-<svc>) -> docker:// routes.
#   2d  per-episode mode (TB_M5_MODE=per_episode, the default): IAM roles + a Docker-ready AMI, so every m5
#       episode gets its OWN fresh m5.xlarge (40 GB) that pulls only its task's images from ECR.
# TB_SUBSTRATES=microvm or =docker runs one half only.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/deploy/steps_common.sh"
need_host
cd "$HERE"
SUBS="${TB_SUBSTRATES:-microvm,docker}"
[ -f terminalbench/tasks/manifest.json ] || python3 terminalbench/fetch_terminalbench.py
if [[ ",$SUBS," == *,microvm,* ]]; then
  log "step 2a: build + push the microVM task images (no Runtimes yet)"
  TB_NO_REGISTER=1 python3 terminalbench/deploy_terminalbench.py
fi
if [[ ",$SUBS," == *,docker,* ]]; then
  log "step 2b: the m5 Docker host (builder)"
  python3 deploy/m5_host.py up
  python3 deploy/m5_host.py roles
  log "step 2c: build the m5 tasks on the m5 + push them to ECR"
  python3 terminalbench/deploy_docker_tasks.py
  if [ "${TB_M5_MODE:-per_episode}" = per_episode ] && ! python3 -c "import json,sys; sys.exit(0 if json.load(open('$M5_STATE')).get('episode') else 1)"; then
    log "step 2d: bake the per-episode m5 AMI"
    python3 deploy/m5_host.py bake-ami
  fi
fi
log "done — next: bash 3_push_ecr.sh"
