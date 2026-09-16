#!/bin/bash
# Sequential B-demo queue: waits for the in-flight gpu_single to finish, then runs
# gpu_shared. Each demo's run.sh blocks until its ray job exits.
set -uo pipefail
BASE=/work/code/demos
STAMP=$(date +%Y%m%d_%H%M%S)
Q=$BASE/b_queue_$STAMP.status
export PYTHONPATH=/work/code
GPU_SBX_B4="arn:aws:bedrock-agentcore:us-west-2:<ACCOUNT_ID>:runtime/acs_sandbox_gpu_b4-7rnu8R3gS7"

log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$Q"; }

log "QUEUE START. Waiting for gpu_single (train_async.py) to finish..."
# wait for the currently-running gpu_single to release the GPUs
sleep 30
while pgrep -f "train_async.py" >/dev/null 2>&1; do sleep 60; done
log "GPUs free — gpu_single done."

run_demo(){
  local dir=$1; shift
  log "BEGIN $dir"
  ( cd "$BASE/$dir" && env "$@" PYTHONPATH=/work/code bash run.sh ) \
      > "$BASE/$dir/train.queued_$STAMP.log" 2>&1
  local rc=$?
  log "END   $dir rc=$rc"
  # small gap to let ray fully release
  sleep 45
}

run_demo demo_gpu_shared \
  AGENTCORE_SANDBOX_RUNTIME_ARN="$GPU_SBX_B4"

log "QUEUE DONE."
