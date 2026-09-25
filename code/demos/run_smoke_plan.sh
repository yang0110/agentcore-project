#!/bin/bash
# Plan-aligned smoke driver — implements smoke-test-experiment-plan.md at NR=11 (10 train + 1 eval,
# single final checkpoint). Runs each RL trainer SERIALLY (each claims all 8 GPUs), tees a detailed
# per-run log to _smoke_logs/<demo>.log, parses it to <demo>.csv/.eval.csv/.png via the demo's
# metrics.py, deletes the run's checkpoint right after verifying (disk never accumulates), and
# appends a results row. Continues past failures so one bad demo does not abort the sweep.
# Runs INSIDE agentcore_p4 (as root => can delete its own root-owned checkpoints).
#
#   PHASE=1  -> Group A (incl. demo_cpu_tasks:mixed) + HEARTBEAT=1 + scale_lanes (13, no extra env)   [default: 1,2]
#   PHASE=2  -> demo_cpu_tasks VARIANT=imagepool (needs AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE; was demo_cpu_imagepool)
# Override CPU_TRAINERS / NR / TRAIN_TIMEOUT / PHASES to resume a subset.
set -uo pipefail

BASE="${BASE:-/work/code/demos}"
LOGDIR="${LOGDIR:-$BASE/_smoke_logs}"
RESULTS="${RESULTS:-$LOGDIR/smoke_plan_results.tsv}"
NR="${NR:-11}"                       # 10 train steps + 1 eval (plan spec)
EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
EVAL_NS="${EVAL_NS:-2}"
TRAIN_TIMEOUT="${TRAIN_TIMEOUT:-2400}"
PHASES="${PHASES:-1 2}"

# Entries are <demo>[:<DATASET>][+KEY=VAL...] — demo_cpu_tasks is ONE folder that trains many datasets,
# so each dataset is its own entry (run with DATASET=<name>; logged/reported as <demo>-<name>). A
# "+KEY=VAL" suffix adds env knobs, e.g. "+HEARTBEAT=1" (was the standalone demo_cpu_heartbeat).
CPU_TRAINERS="${CPU_TRAINERS-demo_cpu_tasks:mbpp demo_cpu_tasks:math demo_cpu_tasks:mixed demo_cpu_tasks:gsm8k demo_cpu_tasks:math500 demo_cpu_tasks:deepscaler demo_cpu_tasks:humaneval demo_cpu_tasks:gretelsql demo_cpu_tasks:hotpotqa demo_cpu_tasks:xlam demo_cpu_tasks:mcp demo_cpu_tasks:deepscaler+HEARTBEAT=1 demo_cpu_scale_lanes}"

mkdir -p "$LOGDIR"
[ -f "$RESULTS" ] || printf 'demo\tstatus\tsecs\tnote\n' > "$RESULTS"
row() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" | tee -a "$RESULTS"; }

free_gpus() { pkill -9 sglang 2>/dev/null; ray stop --force >/dev/null 2>&1; pkill -9 ray 2>/dev/null; sleep 5; }

# Run one trainer end-to-end. $1=demo name (or demo:DATASET), $2..=extra env exports (as KEY=VAL).
run_trainer() {
  local entry="$1"; shift
  local knobs=""
  [ "$entry" != "${entry%%+*}" ] && { knobs="${entry#*+}"; entry="${entry%%+*}"; set -- ${knobs//+/ } "$@"; }
  local d="${entry%%:*}" ds=""
  [ "$entry" != "$d" ] && { ds="${entry#*:}"; set -- "DATASET=$ds" "$@"; }
  # log/results label: one per dataset (+knobs) for demo_cpu_tasks
  local tag="$d${ds:+-$ds}$(printf '%s' "${knobs:+-$knobs}" | tr '+=' '-_' | tr 'A-Z' 'a-z')"
  echo "[plan] TRAINER $tag  ($(date -u +%TZ))"
  free_gpus
  local log="$LOGDIR/$tag.log"
  local t0=$SECONDS
  # build data if the demo has a prep step (idempotent; cheap if already built)
  [ -f "$BASE/$d/prepare_data.sh" ] && env "$@" bash "$BASE/$d/prepare_data.sh" >>"$log" 2>&1
  # -u strips any globally-exported image->ARN map so it never leaks into Group-A trainers (incl. DATASET=mixed)
  # (only VARIANT=imagepool re-adds it via "$@"); a leaked map made the mixed run build invalid runtime-env-json.
  ( cd /work && env -u AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE "$@" \
      NR="$NR" EVAL_INTERVAL="$EVAL_INTERVAL" EVAL_NS="$EVAL_NS" \
      timeout "$TRAIN_TIMEOUT" bash "$BASE/$d/run.sh" ) > "$log" 2>&1
  local rc=$?
  local save; save=$(grep -aoE -- "--save /[^ ]+" "$log" | head -1 | awk '{print $2}')
  local has_final=0
  [ -n "$save" ] && [ -f "$save/latest_checkpointed_iteration.txt" ] && has_final=1
  # parse metrics (non-fatal)
  python3 "$BASE/$d/metrics.py" "$log" \
      --csv "$LOGDIR/$tag.csv" --eval-csv "$LOGDIR/$tag.eval.csv" --plot "$LOGDIR/$tag.png" \
      >>"$log" 2>&1 || echo "[plan] metrics.py non-zero for $tag" >&2
  if grep -q "succeeded" "$log" \
     && ! grep -qE "No space|CheckpointException|unexpected pos" "$log" \
     && [ "$has_final" = "1" ]; then
    row "$tag" PASS $((SECONDS-t0)) "succeeded; final ckpt $(cat "$save/latest_checkpointed_iteration.txt" 2>/dev/null)"
  elif [ "$rc" = "124" ]; then
    row "$tag" TIMEOUT $((SECONDS-t0)) "exceeded ${TRAIN_TIMEOUT}s"
  else
    local reason="rc=$rc"
    grep -qE "No space" "$log" && reason="$reason ENOSPC"
    grep -qE "CheckpointException|unexpected pos" "$log" && reason="$reason ckpt-corrupt"
    [ "$has_final" = "0" ] && reason="$reason no-final-ckpt"
    grep -q "succeeded" "$log" || reason="$reason no-success-marker"
    row "$tag" FAIL $((SECONDS-t0)) "$reason (see _smoke_logs/$tag.log)"
  fi
  [ -n "$save" ] && rm -rf "$save" 2>/dev/null   # reclaim disk; keep the log/csv/png
}

echo "[plan] start $(date -u +%FT%TZ)  NR=$NR PHASES='$PHASES'"

for ph in $PHASES; do
  case "$ph" in
    1) for d in $CPU_TRAINERS; do run_trainer "$d"; done ;;
    2) if [ -n "${AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE:-}" ]; then
         run_trainer demo_cpu_tasks+VARIANT=imagepool "AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE=$AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE"
       else
         row demo_cpu_tasks-variant_imagepool SKIP 0 "AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE unset"
       fi ;;
  esac
done

free_gpus
echo "[plan] done $(date -u +%FT%TZ)"
echo "[plan] results:"; cat "$RESULTS"
