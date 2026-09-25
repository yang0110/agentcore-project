#!/bin/bash
# Shared helpers for this demo's numbered step scripts (1_prepare_data.sh ... 6_cleanup.sh).
# Identical in every demo's deploy/; per-demo settings live in deploy/demo.env. Sourced, not run:
#     HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$HERE/deploy/steps_common.sh"
#
# Where things run:
#   * HOST (this box): docker build / ECR push / AgentCore deploy / local GPU farm (needs docker, aws, boto3).
#   * agentcore_p4 CONTAINER: data prep + the slime trainer (repo mounted at /work; no docker socket).
#     Steps that belong in the container are wrapped in `in_container`, which `docker exec`s from the
#     host — or runs directly if you are already inside the container.
set -euo pipefail

DEMO="$(basename "$HERE")"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
CONTAINER="${AGENTCORE_CONTAINER:-agentcore_p4}"
CONTAINER_REPO="${CONTAINER_REPO:-/work}"                  # repo root as mounted inside the container
C_DEMO_DIR="$CONTAINER_REPO/code/demos/$DEMO"             # this demo dir, as seen in the container
export AWS_REGION="${AWS_REGION:-us-west-2}"
export NAME_PREFIX="${NAME_PREFIX:-acs}"
BUILDX_BUILDER="${BUILDX_BUILDER:-acsbuilder}"

# shellcheck disable=SC1091
VARIANT="${VARIANT:-}"
source "$HERE/deploy/demo.env"
# VARIANT (optional, demo-specific; e.g. demo_cpu_tasks VARIANT=bigimage) keeps its own env files.
IMAGES_ENV="$HERE/deploy/images${VARIANT:+.$VARIANT}.env"    # written by 3_push_ecr.sh
RUNTIME_ENV="$HERE/deploy/runtime${VARIANT:+.$VARIANT}.env"  # written by 4_deploy_runtime.sh

log() { printf '\n>>> [%s] %s\n' "$DEMO" "$*"; }
die() { printf '!! [%s] %s\n' "$DEMO" "$*" >&2; exit 1; }

# Env vars forwarded into the container (run knobs + everything the sandbox/farm reads).
PASS_RE='^(NR|RB|NS|CONC|TEMP|MODE|DATASET|VARIANT|FAMILY|FARM|LOCAL_GPU|LANES|LANE_SANDBOX|FAMILIES|BALANCED|INFER_CONC|HARNESS_WORKERS|SLIME_CUDA_VISIBLE|SLIME_NUM_GPUS|SANDBOX|EVAL_INTERVAL|EVAL_NS|SAVE_INTERVAL|SAVE|ACTOR_GPUS|ROLLOUT_GPUS|CKPT_BASE|LIMIT|REPEAT|REBUILD|AWS_REGION|AWS_BEARER_TOKEN_BEDROCK|AGENTCORE_[A-Z0-9_]+|B4_[A-Z0-9_]+|TB_[A-Z0-9_]+|SNAPSHOT_[A-Z0-9_]+|RECYCLE_[A-Z0-9_]+|HEARTBEAT|HEARTBEAT_[A-Z0-9_]+|ADAPTER_[A-Z0-9_]+|SANDBOX_[A-Z0-9_]+)$'

inside_container() { [ -d "$C_DEMO_DIR" ] && ! command -v docker >/dev/null 2>&1; }

# in_container "<bash command>"  — run from $CONTAINER_REPO inside agentcore_p4.
in_container() {
  if inside_container; then (cd "$CONTAINER_REPO" && bash -lc "$1"); return; fi
  local args=() v
  for v in $(compgen -e); do [[ "$v" =~ $PASS_RE ]] && args+=(-e "$v=${!v}"); done
  docker exec "${args[@]}" "$CONTAINER" bash -lc "cd $CONTAINER_REPO && $1"
}

need_host() { inside_container && die "run this step on the HOST (needs docker), not inside $CONTAINER"; return 0; }

# IMAGES = "logical:Dockerfile ..." (demo.env; override with IMAGES=... to build a subset).
image_names() { local s; for s in $IMAGES; do echo "${s%%:*}"; done; }
image_dockerfile() { local s; for s in $IMAGES; do [ "${s%%:*}" = "$1" ] && echo "${s#*:}"; done; }
image_repo() { echo "${REPO_PREFIX}$1"; }                  # ECR repo + local tag name
runtime_name() { echo "${RUNTIME_PREFIX}$1"; }             # AgentCore Runtime name

load_env() { local f; for f in "$@"; do [ -f "$f" ] && { set -a; source "$f"; set +a; }; done; return 0; }
