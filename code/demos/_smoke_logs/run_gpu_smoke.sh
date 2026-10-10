#!/bin/bash
# HOST-side GPU smoke driver (companion to run_smoke_plan.sh, which only drives CPU trainers from
# inside slime_trainer). GPU demos need the two-step: a LOCAL farm container on host GPUs 6,7
# (run_local_gpu_runtime.sh) + the trainer inside slime_trainer on GPUs 0-5. Runs each serially,
# NR=11 (10 train + 1 eval + one final ckpt), tees logs, verifies + parses, records PASS/FAIL,
# deletes each ckpt. Continues past failures. Ordered to minimize farm restarts.
set -uo pipefail
REPO=/home/ubuntu/workspace/agentcore-project
LOGDIR="$REPO/code/demos/_smoke_logs"
RES="$LOGDIR/gpu_smoke_results.tsv"
NR="${NR:-11}"; EVAL_INTERVAL="${EVAL_INTERVAL:-10}"; EVAL_NS="${EVAL_NS:-2}"
# GPU smoke uses a SMALLER GRPO group than the CPU default (NS=8). KernelBench episodes are heavy —
# each holds a whole-GPU farm slot for ~17s+ and compiles CUDA (nvcc), serialized host-wide by a
# flock. At NS=8 that's 32 episodes/step: the 2-slot local farm wedges (a Ray GPU actor stuck on a
# runaway nvcc holds the flock -> the whole step deadlocks; observed customruntime completing 1
# episode then hanging 30 min). NS=2 => 8 episodes/step, 4x less cumulative nvcc, so a step is
# tractable AND the wedge risk drops. Keeps the defining smoke budget (10 steps, 1 eval) intact.
NS="${NS:-2}"
# 3600s (60 min): a single farm GCS crash + docker restart + torch-reinstall recovery can burn
# ~10-13 min in the trainer's 30s-capped retry backoff (all concurrent episodes wait it out). 2400s
# left no headroom for even one crash on the whole-GPU farm demo; 3600s survives one crash and still finishes.
TRAIN_TIMEOUT="${TRAIN_TIMEOUT:-3600}"
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

run(){ # $1=demo[:label]  $2=farm-mode  $3..=extra KEY=VAL env for the trainer
  # <demo>:<label> is just a log label (e.g. demo_gpu_single:snapshot), logged as <demo>-<label>.
  # (The farm demo is ONE folder again since 2026-09-26: demo_gpu_farm:<mode> + FARM=<mode>.)
  local dir="${1%%:*}" d="${1/:/-}" mode="$2"; shift 2
  echo "[gpu] === $d (farm=$mode ${*:-}) $(date -u +%TZ) ==="
  farm "$mode" || { row "$d" FAIL 0 "farm mode=$mode did not come up"; return; }
  # Two views of the same file via the /work mount: hlog = host path (host-side grep/test can read
  # even root-owned 644 logs); clog = container path (all IN-CONTAINER writes go here, so root — not
  # the host `ubuntu` user — creates/overwrites the log; fixes the Permission-denied false FAIL).
  local hlog="$LOGDIR/$d.log" clog="/work/code/demos/_smoke_logs/$d.log" t0=$SECONDS
  docker exec $(for kv in "$@" NR="$NR" NS="$NS" EVAL_INTERVAL="$EVAL_INTERVAL" EVAL_NS="$EVAL_NS"; do printf -- '-e %s ' "$kv"; done) \
    slime_trainer bash -lc "cd /work && timeout $TRAIN_TIMEOUT bash code/demos/$dir/run.sh > $clog 2>&1"
  local rc=$?
  local save; save=$(grep -aoE -- "--save /[^ ]+" "$hlog" | head -1 | awk '{print $2}')
  local has_final=0
  [ -n "$save" ] && docker exec slime_trainer test -f "$save/latest_checkpointed_iteration.txt" && has_final=1
  # metrics.py + its outputs all use container (/work) paths and append inside the container (root).
  # NOTE: must `cd /work` first — `bash -lc` starts in HOME (/root), so a repo-relative
  # `code/demos/...` path would resolve to /root/code/... and fail ("can't open file").
  docker exec slime_trainer bash -lc \
      "cd /work && python3 code/demos/$dir/metrics.py $clog --csv /work/code/demos/_smoke_logs/$d.csv --eval-csv /work/code/demos/_smoke_logs/$d.eval.csv --plot /work/code/demos/_smoke_logs/$d.png >> $clog 2>&1" \
      || echo "[gpu] metrics.py non-zero for $d"
  if grep -q "succeeded" "$hlog" && ! grep -qE "No space|CheckpointException|unexpected pos" "$hlog" && [ "$has_final" = 1 ]; then
    row "$d" PASS $((SECONDS-t0)) "succeeded; final ckpt $(docker exec slime_trainer cat "$save/latest_checkpointed_iteration.txt" 2>/dev/null)"
  elif [ "$rc" = 124 ]; then row "$d" TIMEOUT $((SECONDS-t0)) "exceeded ${TRAIN_TIMEOUT}s"
  else
    local r="rc=$rc"; grep -qE "No space" "$hlog" && r="$r ENOSPC"
    grep -qiE "out of memory|CUDA out of memory|Killed" "$hlog" && r="$r OOM/killed"
    [ "$has_final" = 0 ] && r="$r no-final-ckpt"; grep -q succeeded "$hlog" || r="$r no-success-marker"
    row "$d" FAIL $((SECONDS-t0)) "$r (see _smoke_logs/$d.log)"
  fi
  [ -n "$save" ] && docker exec slime_trainer rm -rf "$save" 2>/dev/null
}

echo "[gpu] start $(date -u +%FT%TZ) NR=$NR"
# whole-GPU farm group (6,7): demo_gpu_farm FARM=whole + demo_gpu_single's LOCAL_GPU=1 path, plain and
# with its snapshot angle (demo_gpu_snapshot / demo_gpu_customruntime were merged into demo_gpu_single 2026-09-25)
run demo_gpu_farm:whole       whole  FARM=whole
run demo_gpu_single           whole  LOCAL_GPU=1
run demo_gpu_single:snapshot  whole  LOCAL_GPU=1 SNAPSHOT_EVERY=1
# fractional single-GPU farm (GPU 6, 0.5 + 0.5)
run demo_gpu_farm:fractional  fractional  FARM=fractional
bash "$REPO/code/deploy/run_local_gpu_runtime.sh" stop >/dev/null 2>&1
echo "[gpu] done $(date -u +%FT%TZ)"; cat "$RES"
