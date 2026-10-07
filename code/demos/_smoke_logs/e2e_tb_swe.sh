#!/bin/bash
# End-to-end build + deploy + smoke of demo_cpu_terminalbench then demo_cpu_swebench (sequential: the
# slime trainer takes all 8 GPUs). One demo failing does not stop the other. Status -> $OUT.status.tsv
# TB_STEPS / SWE_STEPS override the step lists (to resume). Pipe this script's output into the log
# (| cat >> log): 4_deploy_runtime.sh's `tee /dev/stderr` would truncate a plain file.
OUT="$1"; cd "$(dirname "$0")/.."
TB_STEPS="${TB_STEPS:-2_build_image.sh 3_push_ecr.sh 4_deploy_runtime.sh 1_prepare_data.sh 5_run.sh}"
SWE_STEPS="${SWE_STEPS:-1_prepare_data.sh 2_build_image.sh 3_push_ecr.sh 4_deploy_runtime.sh 5_run.sh}"
step() {  # demo step-script
  local t0=$SECONDS; echo "===== $(date -Is) $1 $2"
  (cd "$1" && bash "$2"); local rc=$?
  printf '%s\t%s\t%s\t%ss\n' "$1" "$2" "$rc" "$((SECONDS-t0))" >> "$OUT.status.tsv"; return $rc
}
# Terminal-Bench: steps 2-4 (resumable: pushed images / existing Runtimes are reused), THEN 1 (rows are
# filtered to the deployed + oracle-verified pool), then the 10-step smoke.
( for s in $TB_STEPS; do step demo_cpu_terminalbench $s || exit 1; done )
# SWE-bench: numbered order (step 4 also oracle-checks and rebuilds the data).
( for s in $SWE_STEPS; do step demo_cpu_swebench $s || exit 1; done )
echo "===== $(date -Is) ALL DONE"; cat "$OUT.status.tsv"
