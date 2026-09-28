#!/bin/bash
# Step 4/6 — create the AgentCore sandbox Runtime(s) from the pushed image(s) (HOST, real AWS $), or
# start the LOCAL GPU farm. Idempotent (existing Runtimes are reused; FORCE_RECREATE=1 to pick up a new
# image). Writes deploy/runtime.env (the ARN / ARN map), which 5_run.sh forwards to run.sh.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/deploy/steps_common.sh"
need_host
load_env "$IMAGES_ENV"

if [ "$SUBSTRATE" = local-gpu ] || [ "${LOCAL_GPU:-0}" = 1 ]; then
  MODE="${FARM:-$LOCAL_FARM_MODE}"
  log "step 4/6: start the LOCAL GPU farm ($MODE) on host GPUs — same image, served at 127.0.0.1:8080"
  docker tag "$(image_repo gpu):latest" "acs-local-gpu-sbx:$MODE"
  SKIP_BUILD=1 bash "$HERE/deploy/local_gpu_runtime.sh" "$MODE"
  echo "FARM=$MODE" > "$RUNTIME_ENV"
  cat "$RUNTIME_ENV"; log "done — next: bash 5_run.sh"; exit 0
fi

if [ -n "${DEPLOY_CMD:-}" ]; then
  log "step 4/6: $DEPLOY_CMD"
  out="$(cd "$HERE" && eval "$DEPLOY_CMD" | tee /dev/stderr)"
  grep '^export ' <<< "$out" | sed 's/^export //' > "$RUNTIME_ENV" || true
else
  log "step 4/6: create $SUBSTRATE Runtime(s) for $(image_names | tr '\n' ' ')"
  declare -A ARN
  for name in $(image_names); do
    uri_var="IMAGE_URI_$name"; uri="${!uri_var:-}"
    [ -n "$uri" ] || die "no $uri_var in deploy/images.env — run 3_push_ecr.sh first"
    ARN[$name]="$(python3 "$HERE/deploy/agentcore_deploy.py" runtime --name "$(runtime_name "$name")" \
                   --image "$uri" "${RUNTIME_ARGS[@]}" | tail -n 1)"
    echo "  $name -> ${ARN[$name]}"
  done
  first="$(image_names | head -n 1)"
  {
    case "$ARN_WIRING" in
      arn)  echo "AGENTCORE_SANDBOX_RUNTIME_ARN=${ARN[$first]}" ;;
      arns) echo "AGENTCORE_SANDBOX_RUNTIME_ARNS=${ARN[$first]}" ;;
    esac
    if [ "${#ARN[@]}" -gt 1 ] || [ "$ARN_WIRING" = by_image ]; then
      map="$(for k in "${!ARN[@]}"; do printf '"%s":"%s",' "$k" "${ARN[$k]}"; done)"
      echo "AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE='{${map%,}}'"
    fi
    # Deploy-time choices the client must match (demo.env RUNTIME_RECORD, e.g. gpu_farm's FARM split).
    for kv in "${RUNTIME_RECORD[@]}"; do echo "$kv"; done
  } > "$RUNTIME_ENV"
fi
echo "--- deploy/runtime.env"; cat "$RUNTIME_ENV"
log "done — next: bash 5_run.sh"
