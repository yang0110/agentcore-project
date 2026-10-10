#!/bin/bash
# Step 5/6 — run RL (slime trainer inside slime_trainer) against both substrates from steps 2-4, tee the full
# log to logs/<tag>.log, then parse it to <tag>.csv / .eval.csv / .png with metrics.py.
# Defaults = the full-epoch smoke (NR=epoch: one pass over the train split, rows/RB steps, eval every
# EVAL_INTERVAL + final ckpt). Override any knob, e.g. the old 10-step smoke:
#     NR=11 bash 5_run.sh          or a quick one:  NR=3 TB_PREOPEN_BATCHES=1 bash 5_run.sh
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/deploy/steps_common.sh"
[ -f "$POOL_MAP" ] || die "no terminalbench/tasks/route_by_image.json — run steps 2-4 first"
if grep -q '"docker://' "$POOL_MAP" && [ ! -f "$M5_STATE" ]; then
  die "the route map has m5 tasks but $M5_STATE is missing — run: python3 deploy/m5_host.py up"
fi
export NR="${NR:-epoch}" EVAL_INTERVAL="${EVAL_INTERVAL:-10}" EVAL_NS="${EVAL_NS:-2}"
_SFX="-hybrid"
if [ "${HEARTBEAT:-1}" = 0 ]; then _SFX+="-nohb"; fi
TAG="${RUN_TAG:-run}${_SFX}_$(date +%Y%m%d_%H%M%S)"
C_LOG="$C_DEMO_DIR/logs/$TAG.log"
export TB_JOB_ID="tb-$TAG"     # the ray job's id: a Ctrl-C here stops it (docker exec forwards no signal)
_interrupted() {
  trap - INT TERM
  echo "!! interrupted: stopping ray job $TB_JOB_ID in $CONTAINER (run.sh then stops ray + sglang)" >&2
  in_container "timeout 120 ray job stop --address=http://127.0.0.1:8265 $TB_JOB_ID" || true
  if [ -f "$M5_STATE" ]; then python3 "$HERE/deploy/m5_host.py" gc-episodes || true; fi
  exit 130
}
trap _interrupted INT TERM
mkdir -p "$HERE/logs" 2>/dev/null || true
log "step 5/6: run.sh  NR=$NR EVAL_INTERVAL=$EVAL_INTERVAL -> logs/$TAG.log"
in_container "mkdir -p $C_DEMO_DIR/logs && set -o pipefail && bash $C_DEMO_DIR/run.sh 2>&1 | tee $C_LOG" || RUN_RC=$?
log "parse metrics"
in_container "python3 $C_DEMO_DIR/metrics.py $C_LOG --csv ${C_LOG%.log}.csv --eval-csv ${C_LOG%.log}.eval.csv --plot ${C_LOG%.log}.png" \
  || echo "!! metrics.py failed (log kept at logs/$TAG.log)"
if [ -f "$M5_STATE" ]; then   # warm / per-episode m5 instances never outlive a run
  python3 "$HERE/deploy/m5_host.py" gc-episodes || true
fi
log "done — logs/$TAG.{log,csv,eval.csv,png}; tear down with: bash 6_cleanup.sh"
exit "${RUN_RC:-0}"
