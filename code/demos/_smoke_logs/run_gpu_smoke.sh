#!/bin/bash
# HOST-side GPU smoke driver (companion to run_smoke_plan.sh, which only drives CPU trainers from
# inside agentcore_p4). GPU demos need the two-step: a LOCAL farm container on host GPUs 6,7
# (run_local_gpu_runtime.sh) + the trainer inside agentcore_p4 on GPUs 0-5. Runs each serially,
# NR=11 (10 train + 1 eval + one final ckpt), tees logs, verifies + parses, records PASS/FAIL,
# deletes each ckpt. Continues past failures. Ordered to minimize farm restarts.
set -uo pipefail
REPO=/home/ubuntu/workspace/agentcore-project
LOGDIR="$REPO/code/demos/_smoke_logs"
RES="$LOGDIR/gpu_smoke_results.tsv"
NR="${NR:-11}"; EVAL_INTERVAL="${EVAL_INTERVAL:-10}"; EVAL_NS="${EVAL_NS:-2}"
TRAIN_TIMEOUT="${TRAIN_TIMEOUT:-2400}"
mkdir -p "$LOGDIR"
[ -f "$RES" ] || printf 'demo\tstatus\tsecs\tnote\n' > "$RES"
row(){ printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" | tee -a "$RES"; }

CUR_MODE=""
farm(){ # $1 = mode; (re)start only if mode changed
  local m="$1"
  [ "$m" = "$CUR_MODE" ] && return 0
  echo "[gpu] (re)starting farm mode=$m"
  bash "$REPO/code/deploy/run_local_gpu_runtime.sh" "$m" > "$LOGDIR/_farm_$m.log" 2>&1 || {
    echo "[gpu] farm start FAILED (mode=$m); see _farm_$m.log"; return 1; }
  CUR_MODE="$m"
}

run(){ # $1=demo  $2=farm-mode  $3..=extra KEY=VAL env for the trainer
  local d="$1" mode="$2"; shift 2
  echo "[gpu] === $d (farm=$mode ${*:-}) $(date -u +%TZ) ==="
  farm "$mode" || { row "$d" FAIL 0 "farm mode=$mode did not come up"; return; }
  # Two views of the same file via the /work mount: hlog = host path (host-side grep/test can read
  # even root-owned 644 logs); clog = container path (all IN-CONTAINER writes go here, so root — not
  # the host `ubuntu` user — creates/overwrites the log; fixes the Permission-denied false FAIL).
  local hlog="$LOGDIR/$d.log" clog="/work/code/demos/_smoke_logs/$d.log" t0=$SECONDS
  docker exec $(for kv in "$@" NR="$NR" EVAL_INTERVAL="$EVAL_INTERVAL" EVAL_NS="$EVAL_NS"; do printf -- '-e %s ' "$kv"; done) \
    agentcore_p4 bash -lc "cd /work && timeout $TRAIN_TIMEOUT bash code/demos/$d/run.sh > $clog 2>&1"
  local rc=$?
  local save; save=$(grep -aoE -- "--save /[^ ]+" "$hlog" | head -1 | awk '{print $2}')
  local has_final=0
  [ -n "$save" ] && docker exec agentcore_p4 test -f "$save/latest_checkpointed_iteration.txt" && has_final=1
  # metrics.py + its outputs all use container (/work) paths and append inside the container (root).
  docker exec agentcore_p4 bash -lc \
      "python3 code/demos/$d/metrics.py $clog --csv /work/code/demos/_smoke_logs/$d.csv --eval-csv /work/code/demos/_smoke_logs/$d.eval.csv --plot /work/code/demos/_smoke_logs/$d.png >> $clog 2>&1" \
      || echo "[gpu] metrics.py non-zero for $d"
  if grep -q "succeeded" "$hlog" && ! grep -qE "No space|CheckpointException|unexpected pos" "$hlog" && [ "$has_final" = 1 ]; then
    row "$d" PASS $((SECONDS-t0)) "succeeded; final ckpt $(docker exec agentcore_p4 cat "$save/latest_checkpointed_iteration.txt" 2>/dev/null)"
  elif [ "$rc" = 124 ]; then row "$d" TIMEOUT $((SECONDS-t0)) "exceeded ${TRAIN_TIMEOUT}s"
  else
    local r="rc=$rc"; grep -qE "No space" "$hlog" && r="$r ENOSPC"
    grep -qiE "out of memory|CUDA out of memory|Killed" "$hlog" && r="$r OOM/killed"
    [ "$has_final" = 0 ] && r="$r no-final-ckpt"; grep -q succeeded "$hlog" || r="$r no-success-marker"
    row "$d" FAIL $((SECONDS-t0)) "$r (see _smoke_logs/$d.log)"
  fi
  [ -n "$save" ] && docker exec agentcore_p4 rm -rf "$save" 2>/dev/null
}

echo "[gpu] start $(date -u +%FT%TZ) NR=$NR"
# local-mode farm group (whole-GPU on 6,7): native local demo + the three LOCAL_GPU=1 remote demos
run demo_gpu_local          local
run demo_gpu_customruntime  local  LOCAL_GPU=1
run demo_gpu_single         local  LOCAL_GPU=1
run demo_gpu_snapshot       local  LOCAL_GPU=1 SNAPSHOT_EVERY=1
# whole-GPU shared framing
run demo_gpu_shared_multi   shared_multi
# fractional single-GPU farm
run demo_gpu_shared_single  shared_single
bash "$REPO/code/deploy/run_local_gpu_runtime.sh" stop >/dev/null 2>&1
echo "[gpu] done $(date -u +%FT%TZ)"; cat "$RES"
