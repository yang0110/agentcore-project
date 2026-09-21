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
#   bash code/deploy/run_local_gpu_runtime.sh local           # demo_gpu_local: WHOLE-GPU farm on 6,7
#                                                              #   (B4_FARM_GPUS=2, FRACTION=1.0 -> 2
#                                                              #   whole-GPU slots, one GPU per episode)
#   bash code/deploy/run_local_gpu_runtime.sh shared_multi    # demo_gpu_shared_multi: same whole-GPU
#                                                              #   farm on 6,7, one GPU per Ray session
#   bash code/deploy/run_local_gpu_runtime.sh shared_single   # demo_gpu_shared_single: FRACTIONAL farm
#                                                              #   on GPU 6 (FARM_GPUS=1, FRACTION=0.5 ->
#                                                              #   2 slots sharing one GPU, 0.5 + 0.5)
#   bash code/deploy/run_local_gpu_runtime.sh stop            # tear the local runtime container down
#   SKIP_BUILD=1 bash code/deploy/run_local_gpu_runtime.sh <mode>   # reuse the already-built image
#
# Then, inside agentcore_p4, run the matching demo, e.g.:
#   bash code/demos/demo_gpu_shared_single/run.sh
#
# (demo_gpu_single is remote-only — it requires a real g6 sandbox Runtime ARN and has no on-box path;
# this launcher serves the local whole-GPU / fractional farms for demo_gpu_local + demo_gpu_shared_*.)
set -euo pipefail

MODE="${1:-local}"
NAME="${LOCAL_RT_NAME:-acs_local_gpu_sbx}"
PORT="${LOCAL_RT_PORT:-8080}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

stop() { docker rm -f "$NAME" >/dev/null 2>&1 && echo "stopped $NAME" || echo "$NAME not running"; }

if [ "$MODE" = "stop" ]; then stop; exit 0; fi

# Per-mode config. All three build the SAME shared-farm image; they differ only in which host GPUs
# are reserved and how the in-container Ray farm partitions them (all still overridable by env):
#   local / shared_multi -> WHOLE-GPU: 2 GPUs (6,7), fraction 1.0 => 2 whole-GPU slots, one per session.
#   shared_single        -> FRACTIONAL: 1 GPU (6),  fraction 0.5 => 2 slots sharing one GPU (0.5+0.5).
case "$MODE" in
  local)         DEMO_DIR="$REPO_ROOT/code/demos/demo_gpu_local";          TAG="acs-local-gpu-sbx:local"
                 GPUS_DEFAULT="6,7"; FARM_GPUS_DEFAULT="2"; GPU_FRACTION_DEFAULT="1.0"; MEM_FRACTION_DEFAULT="0.95" ;;
  shared_multi)  DEMO_DIR="$REPO_ROOT/code/demos/demo_gpu_shared_multi";    TAG="acs-local-gpu-sbx:shared_multi"
                 GPUS_DEFAULT="6,7"; FARM_GPUS_DEFAULT="2"; GPU_FRACTION_DEFAULT="1.0"; MEM_FRACTION_DEFAULT="0.95" ;;
  shared_single) DEMO_DIR="$REPO_ROOT/code/demos/demo_gpu_shared_single";   TAG="acs-local-gpu-sbx:shared_single"
                 GPUS_DEFAULT="6";   FARM_GPUS_DEFAULT="1"; GPU_FRACTION_DEFAULT="0.5"; MEM_FRACTION_DEFAULT="0.5" ;;
  *) echo "usage: $0 {local|shared_single|shared_multi|stop}"; exit 2 ;;
esac
DOCKERFILE="deploy/Dockerfile.sandbox.gpu.shared"; FARM=1
GPUS="${LOCAL_RT_GPUS:-$GPUS_DEFAULT}"       # host GPUs reserved for the local runtime

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
# NOT --rm: the KernelBench checker compiles CUDA (nvcc) inside the farm and the container can crash
# mid-run (Ray GCS/actor death, a wedged compile). --rm would delete it (and its logs) on exit, so a
# transient crash became fatal to the whole smoke. Instead: --restart=on-failure so docker brings it
# straight back (the trainer's own 21×30s invoke-retry loop then reconnects on the same 127.0.0.1:8080),
# and we stream the container log to a host file so a crash is diagnosable post-mortem. `stop` still
# `docker rm -f`s it, and every launch calls `stop` first, so a dead/exited container is cleaned up.
FARM_LOG="${LOCAL_RT_LOG:-$REPO_ROOT/code/demos/_smoke_logs/_farm_container_${MODE}.log}"
mkdir -p "$(dirname "$FARM_LOG")" 2>/dev/null || true
docker run -d --restart=on-failure:30 --name "$NAME" \
  --gpus "\"device=${GPUS}\"" \
  -p "127.0.0.1:${PORT}:8080" \
  "${SHM_ARGS[@]}" \
  "${FARM_ENV[@]}" \
  "$TAG" >/dev/null
# Persist the container's stdout/stderr to a host file across restarts (best-effort, backgrounded).
( docker logs -f --timestamps "$NAME" >"$FARM_LOG" 2>&1 & ) 2>/dev/null || true

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
echo "Now inside agentcore_p4:  bash code/demos/$(basename "$DEMO_DIR")/run.sh"
echo "Tear down when done:      bash code/deploy/run_local_gpu_runtime.sh stop"
