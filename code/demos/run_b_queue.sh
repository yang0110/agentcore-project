#!/bin/bash
# Sequential B-demo queue: waits for the in-flight gpu_single to finish, then runs the
# LOCAL GPU farm demos (one farm mode per host-farm start). Each demo's run.sh blocks until its ray job exits.
#
# NOTE: the farm demos are LOCAL two-step-launch demos — each needs the matching host farm
# container running (started OUTSIDE this queue, on the HOST). demo_gpu_farm_fractional (FARM_GPUS=1
# on GPU 6, fraction 0.5) and demo_gpu_farm_whole (FARM_GPUS=2 on 6,7, fraction 1.0) use DIFFERENT server-side
# partitions, so the host farm must be (re)started in the matching mode before each:
#   bash code/deploy/run_local_gpu_runtime.sh fractional   # before demo_gpu_farm_fractional
#   bash code/deploy/run_local_gpu_runtime.sh whole        # before demo_gpu_farm_whole
# So this queue is best used ONE mode at a time (comment out the other), not both back to
# back, unless you restart the host farm in between.
set -uo pipefail
BASE=/work/code/demos
STAMP=$(date +%Y%m%d_%H%M%S)
Q=$BASE/b_queue_$STAMP.status
export PYTHONPATH=/work/code

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

run_demo demo_gpu_farm_fractional
# run_demo demo_gpu_farm_whole   # uncomment AFTER restarting the host farm in whole mode

log "QUEUE DONE."
