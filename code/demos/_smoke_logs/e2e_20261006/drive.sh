#!/bin/bash
# End-to-end re-run 2026-10-06: steps 2 -> 3 -> 4 -> 1 -> 5 (48 steps) of demo_terminalbench_kernelbench.
# Step 6 is NOT run (the deployed pool is kept). One log per step next to this file; status.tsv = step, rc, minutes.
set -u
D=/home/ubuntu/workspace/agentcore-project/code/demos/demo_terminalbench_kernelbench
OUT="$(cd "$(dirname "$0")" && pwd)"
UPD=/home/ubuntu/workspace/agentcore-project/code/demos/_smoke_logs/aws_cleanup/idle900_microvm_20261006.log
echo "waiting for the microVM idle-timeout update to finish"
until grep -q "^updated " "$UPD" 2>/dev/null; do sleep 30; done
tail -1 "$UPD"
cd "$D"
run() {  # name, command...
  local n=$1; shift; local t0=$(date +%s)
  echo ">>> $(date -u +%FT%TZ) $n: $*"
  "$@" > "$OUT/$n.log" 2>&1; local rc=$?
  printf "%s\t%s\t%.1f\n" "$n" "$rc" "$(echo "($(date +%s)-$t0)/60" | bc -l)" >> "$OUT/status.tsv"
  echo "<<< $n rc=$rc"; return $rc
}
run step2_build   bash 2_build_image.sh   || exit 1
run step3_report  bash 3_push_ecr.sh      || exit 1
run step4_deploy  bash 4_deploy_runtime.sh || exit 1
run step1_data    bash 1_prepare_data.sh  || exit 1
export RUN_TAG=e2e48 NR=49 EVAL_INTERVAL=10
run step5_train48 bash 5_run.sh
echo "done"
