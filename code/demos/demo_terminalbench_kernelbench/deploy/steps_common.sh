#!/bin/bash
# Shared helpers for the numbered step scripts (1_prepare_data.sh ... 6_cleanup.sh); settings live in
# deploy/demo.env. Sourced, not run:
#     HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$HERE/deploy/steps_common.sh"
#
# Where things run:
#   * HOST (this box): docker build / ECR push / AgentCore deploy (needs docker, aws, boto3).
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
source "$HERE/deploy/demo.env"

log() { printf '\n>>> [%s] %s\n' "$DEMO" "$*"; }
die() { printf '!! [%s] %s\n' "$DEMO" "$*" >&2; exit 1; }

# Env vars forwarded into the container (run knobs + everything the sandbox/farm reads). Path knobs
# (DATA, EVAL_DATA, POOL_MAP) are NOT forwarded: a host path does not exist in the container.
PASS_RE='^(NR|RB|NS|CONC|TEMP|HARNESS_WORKERS|HARNESS_[A-Z0-9_]+|AGENT_TIME_BUDGET_SEC|GPUS_PER_ENGINE|EPISODE_MAX_TURNS|EPISODE_MAX_TOKENS|MAX_TOKENS_PER_GPU|MAX_RESPONSE_LEN|INVOKE_[A-Z_]+|EVAL_NAME|EVAL_TEMP|EVAL_INTERVAL|EVAL_NS|SAVE_INTERVAL|SAVE|ACTOR_GPUS|ROLLOUT_GPUS|CKPT_BASE|WARM_IDLE_TIMEOUT|WARM_PING_FRACTION|MINI_SESSION_[A-Z0-9_]+|MINI_AGENT_[A-Z0-9_]+|MINI_GRADE_TIMEOUT|RUNTIME_NETWORK|RUNTIME_EGRESS|RUNTIME_IDLE_TIMEOUT|SLIME_CUDA_VISIBLE|SLIME_NUM_GPUS|SANDBOX|LIMIT|REPEAT|AWS_REGION|AWS_BEARER_TOKEN_BEDROCK|AGENTCORE_[A-Z0-9_]+|TB_[A-Z0-9_]+|HEARTBEAT|HEARTBEAT_[A-Z0-9_]+|ADAPTER_[A-Z0-9_]+|SANDBOX_[A-Z0-9_]+)$'

inside_container() { [ -d "$C_DEMO_DIR" ] && ! command -v docker >/dev/null 2>&1; }

# in_container "<bash command>"  — run from $CONTAINER_REPO inside agentcore_p4.
in_container() {
  if inside_container; then (cd "$CONTAINER_REPO" && bash -lc "$1"); return; fi
  local args=() v
  for v in $(compgen -e); do [[ "$v" =~ $PASS_RE ]] && args+=(-e "$v=${!v}"); done
  docker exec "${args[@]}" "$CONTAINER" bash -lc "cd $CONTAINER_REPO && $1"
}

need_host() { inside_container && die "run this step on the HOST (needs docker), not inside $CONTAINER"; return 0; }

