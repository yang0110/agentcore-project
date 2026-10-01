#!/bin/bash
# Step 5/6 — run RL (slime trainer inside agentcore_p4) against the task pool from step 4, tee the full log
# to logs/<tag>.log, then parse it to <tag>.csv / .eval.csv / .png with metrics.py.
# Defaults = the 10-step smoke (NR=11: 10 train steps + 1 eval + final ckpt). Override any knob, e.g.
#     NR=3 TB_PREOPEN_BATCHES=1 bash 5_run.sh
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/deploy/steps_common.sh"
[ -f "$POOL_MAP" ] || die "no terminalbench/tasks/arn_by_image.json — run steps 2-4 first"
export NR="${NR:-11}" EVAL_INTERVAL="${EVAL_INTERVAL:-10}" EVAL_NS="${EVAL_NS:-2}"
_SFX="-$SANDBOX"
if [ "${HEARTBEAT:-1}" = 0 ]; then _SFX+="-nohb"; fi
TAG="${RUN_TAG:-run}${_SFX}_$(date +%Y%m%d_%H%M%S)"
C_LOG="$C_DEMO_DIR/logs/$TAG.log"
mkdir -p "$HERE/logs" 2>/dev/null || true
log "step 5/6: run.sh  NR=$NR EVAL_INTERVAL=$EVAL_INTERVAL -> logs/$TAG.log"
in_container "mkdir -p $C_DEMO_DIR/logs && set -o pipefail && bash $C_DEMO_DIR/run.sh 2>&1 | tee $C_LOG"
log "parse metrics"
in_container "python3 $C_DEMO_DIR/metrics.py $C_LOG --csv ${C_LOG%.log}.csv --eval-csv ${C_LOG%.log}.eval.csv --plot ${C_LOG%.log}.png" \
  || echo "!! metrics.py failed (log kept at logs/$TAG.log)"
log "done — logs/$TAG.{log,csv,eval.csv,png}; tear down with: bash 6_cleanup.sh"
