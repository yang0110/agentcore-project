#!/bin/bash
# Serve the AgentCore GPU sandbox Runtime LOCALLY on this host's reserved GPUs (6,7).
#
# This is STILL an AgentCore runtime session: it runs the SAME sandbox_server_agent image
# (BedrockAgentCoreApp) that a remote g6 would, just via `docker run` on the host, and the demo's
# in-process harness reaches it over the same /invocations HTTP contract at http://127.0.0.1:8080
# (agentcore_p4 is --network host, so localhost inside the training container == this host).
#
# RUN THIS ON THE HOST (needs docker + the nvidia container runtime), NOT inside agentcore_p4
# (that container has no docker socket). GPUs 6,7 are reserved here; the demo run.sh keeps slime on
# GPUs 0-5.
#
# Usage (from repo root or anywhere):
#   bash code/deploy/run_local_gpu_runtime.sh local       # demo_gpu_local: WHOLE-GPU Ray farm on 6,7
#                                                          #   (B4_FARM_GPUS=2, B4_GPU_FRACTION=1.0 ->
#                                                          #   2 whole-GPU slots, one GPU per episode)
#   bash code/deploy/run_local_gpu_runtime.sh stop        # tear the local runtime container down
#   SKIP_BUILD=1 bash code/deploy/run_local_gpu_runtime.sh local    # reuse the already-built image
#
# Then, inside agentcore_p4:
#   bash code/demos/demo_gpu_local/run.sh                  # standalone, no LOCAL_GPU flag
#
# (demo_gpu_single / demo_gpu_shared are remote-only — they require a real g6 sandbox Runtime ARN and
# have no on-box path; this launcher serves only the standalone demo_gpu_local whole-GPU farm.)
set -euo pipefail

MODE="${1:-local}"
NAME="${LOCAL_RT_NAME:-acs_local_gpu_sbx}"
PORT="${LOCAL_RT_PORT:-8080}"
GPUS="${LOCAL_RT_GPUS:-6,7}"                 # host GPUs reserved for the local runtime
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

stop() { docker rm -f "$NAME" >/dev/null 2>&1 && echo "stopped $NAME" || echo "$NAME not running"; }

if [ "$MODE" = "stop" ]; then stop; exit 0; fi

case "$MODE" in
  local)  DEMO_DIR="$REPO_ROOT/code/demos/demo_gpu_local";  DOCKERFILE="deploy/Dockerfile.sandbox.gpu.shared"; TAG="acs-local-gpu-sbx:local";  FARM=1 ;;
  *) echo "usage: $0 {local|stop}"; exit 2 ;;
esac

# Farm-knob defaults (all still overridable by env) — WHOLE-GPU: 2 reserved GPUs, fraction 1.0 =>
# 2 whole-GPU slots (one GPU per episode).
FARM_GPUS_DEFAULT="2"; GPU_FRACTION_DEFAULT="1.0"; MEM_FRACTION_DEFAULT="0.95"

# Build the SAME image the remote deploy uses (native amd64), context = the demo dir so the
# Dockerfile's `COPY agentcore_slime_demo/...` resolves against the vendored copy.
if [ "${SKIP_BUILD:-0}" != "1" ]; then
  echo ">>> building $TAG from $DEMO_DIR/$DOCKERFILE"
  docker build --platform linux/amd64 -t "$TAG" -f "$DEMO_DIR/$DOCKERFILE" "$DEMO_DIR"
fi

stop  # remove any prior instance before re-launching

# Farm knobs (LOCAL_GPU reserves 2 host GPUs): B4_FARM_GPUS MUST be set (the slim image has no
# nvidia-smi for Ray to autodetect); the container sees the 2 reserved GPUs as cuda:0,1.
# --shm-size: the shared-farm image runs its OWN internal Ray cluster to schedule the fractional-GPU
# episode workers, and Ray's object store lives in /dev/shm. Docker's default /dev/shm is 64 MB, so
# Ray falls back to /tmp and its GCS server gets starved and killed ("Failed to connect to GCS within
# 60 seconds ... The program will terminate"), which crashes the whole sandbox container mid-run. Give
# the farm real shared memory.
FARM_ENV=()
SHM_ARGS=()
if [ "$FARM" = "1" ]; then
  SHM_ARGS=(--shm-size="${LOCAL_RT_SHM:-10g}")
  FARM_ENV=(
    -e "B4_FARM_GPUS=${B4_FARM_GPUS:-$FARM_GPUS_DEFAULT}"
    -e "B4_GPU_FRACTION=${B4_GPU_FRACTION:-$GPU_FRACTION_DEFAULT}"
    -e "B4_PER_SANDBOX_MEM_FRACTION=${B4_PER_SANDBOX_MEM_FRACTION:-$MEM_FRACTION_DEFAULT}"
    -e "B4_WARM_CUDA=${B4_WARM_CUDA:-1}"
  )
  [ -n "${B4_MAX_SANDBOXES:-}" ] && FARM_ENV+=(-e "B4_MAX_SANDBOXES=${B4_MAX_SANDBOXES}")
fi

echo ">>> starting $NAME on host GPUs $GPUS, port $PORT (mode=$MODE)"
docker run -d --rm --name "$NAME" \
  --gpus "\"device=${GPUS}\"" \
  -p "127.0.0.1:${PORT}:8080" \
  "${SHM_ARGS[@]}" \
  "${FARM_ENV[@]}" \
  "$TAG" >/dev/null

# The image pip-installs the CUDA torch stack at startup (kept out of the image for the 2GB
# AgentCore cap), so the first /ping can take a couple of minutes. Poll until healthy.
echo ">>> waiting for /ping (torch install on first boot can take ~2-3 min)..."
deadline=$(( $(date +%s) + ${LOCAL_RT_WAIT:-300} ))
until curl -fsS "http://127.0.0.1:${PORT}/ping" >/dev/null 2>&1; do
  if ! docker ps --filter "name=$NAME" --format '{{.Names}}' | grep -q "$NAME"; then
    echo "!! container exited; last logs:"; docker logs "$NAME" 2>&1 | tail -30; exit 1
  fi
  [ "$(date +%s)" -ge "$deadline" ] && { echo "!! /ping timed out; last logs:"; docker logs "$NAME" 2>&1 | tail -30; exit 1; }
  sleep 5
done

echo "OK local AgentCore runtime up: http://127.0.0.1:${PORT}  (name=$NAME, GPUs=$GPUS, mode=$MODE)"
echo "Now inside agentcore_p4:  bash code/demos/demo_gpu_local/run.sh   (standalone; no LOCAL_GPU flag)"
echo "Tear down when done:      bash code/deploy/run_local_gpu_runtime.sh stop"
