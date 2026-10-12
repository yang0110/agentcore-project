#!/bin/bash
# Step 5/6 — run RL on the HyperPod cluster against the sandbox tiers from steps 2-4 (HOST):
#   stage this demo + the 9B model + base ckpt to S3 (stage_s3.py, in slime_trainer) -> /fsx/work (pull),
#   sbatch deploy/hyperpod/slurm/train.sbatch on both GPU nodes, stream its log here, then copy the log back
#   to logs/<tag>.log and parse it to <tag>.csv / .eval.csv / .png with metrics.py.
# Defaults = the full-epoch run (NR=epoch). Override any knob, e.g. the 10-step smoke:  NR=11 bash 5_run.sh
# Run knobs (PASS_RE in deploy/steps_common.sh) are forwarded into the job's environment.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/deploy/steps_common.sh"
need_host
[ -f "$POOL_MAP" ] || die "no terminalbench/tasks/route_by_image.json — run steps 2-4 first"
[ -f "$HERE/deploy/hyperpod/hyperpod.json" ] || die "no HyperPod cluster state — run: bash 4b_hyperpod_up.sh"
if grep -q '"docker://' "$POOL_MAP" && [ ! -f "$M5_STATE" ]; then
  die "the route map has m5 tasks but $M5_STATE is missing — run: python3 deploy/m5_host.py up"
fi
if [ "${TB_KB_SUBSTRATE:-split}" != agentcore ] && [ ! -f "$HERE/deploy/g6_host.json" ]; then
  die "TB_KB_SUBSTRATE=${TB_KB_SUBSTRATE:-split} puts KernelBench levels on the EC2 GPU pool but deploy/g6_host.json is missing — run: python3 deploy/g6_host.py image && python3 deploy/g6_host.py bake-ami (or TB_KB_SUBSTRATE=agentcore)"
fi
export NR="${NR:-epoch}" EVAL_INTERVAL="${EVAL_INTERVAL:-10}" EVAL_NS="${EVAL_NS:-2}"
_SFX="-hyperpod"
if [ "${HEARTBEAT:-1}" = 0 ]; then _SFX+="-nohb"; fi
TAG="${RUN_TAG:-run}${_SFX}_$(date +%Y%m%d_%H%M%S)"
export TB_JOB_ID="tb-$TAG"
HP="python3 $HERE/deploy/hyperpod/hyperpod.py"
BUCKET="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['bucket'])" "$HERE/deploy/hyperpod/hyperpod.json")"
JOB=""
_gc() {
  if [ -f "$M5_STATE" ]; then python3 "$HERE/deploy/m5_host.py" gc-episodes || true; fi   # never outlive a run
  if [ -f "$HERE/deploy/g6_host.json" ]; then python3 "$HERE/deploy/g6_host.py" gc || true; fi
}
_interrupted() {
  trap - INT TERM
  echo "!! interrupted: cancelling Slurm job ${JOB:-?} (run.sh stops its ray job on the signal)" >&2
  [ -n "$JOB" ] && $HP cancel "$JOB" || true
  _gc
  exit 130
}
trap _interrupted INT TERM
mkdir -p "$HERE/logs"
log "step 5/6: stage to s3://$BUCKET/work -> /fsx/work"
in_container "python3 $C_DEMO_DIR/deploy/hyperpod/stage_s3.py"
$HP pull
# The run knobs set here, as K=V words for `env` on the controller (sbatch exports them to the job).
# Never a secret: the words travel in the SSM command (CloudTrail) and the Slurm job record; the nodes
# authenticate with their own role.
HP_RUN_ENV=""
for v in $(compgen -e); do
  [ "$v" = AWS_BEARER_TOKEN_BEDROCK ] && continue
  [[ "$v" =~ $PASS_RE ]] && HP_RUN_ENV+=" $v=$(printf '%q' "${!v}")"
done
export HP_RUN_ENV
log "submit  NR=$NR EVAL_INTERVAL=$EVAL_INTERVAL -> logs/$TAG.log"
JOB="$($HP submit "$TAG" | sed -n 's/^OUT JOB=//p')"
[ -n "$JOB" ] || die "sbatch failed"
log "Slurm job $JOB — following its log (Ctrl-C cancels the job)"
$HP follow "$JOB" "$TAG" || RUN_RC=$?
aws s3 cp --only-show-errors "s3://$BUCKET/logs/$TAG.log" "$HERE/logs/$TAG.log" || echo "!! no log in s3://$BUCKET/logs/$TAG.log"
C_LOG="$C_DEMO_DIR/logs/$TAG.log"
log "parse metrics"
in_container "python3 $C_DEMO_DIR/metrics.py $C_LOG --csv ${C_LOG%.log}.csv --eval-csv ${C_LOG%.log}.eval.csv --plot ${C_LOG%.log}.png" \
  || echo "!! metrics.py failed (log kept at logs/$TAG.log)"
_gc
log "done — logs/$TAG.{log,csv,eval.csv,png}; checkpoints stay on FSx (/fsx/work/code/demos/$DEMO/ckpt); tear down with: bash 6_cleanup.sh"
exit "${RUN_RC:-0}"
