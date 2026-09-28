#!/bin/bash
# One short smoke per demo (NR=3 = 2 train steps + 1 eval + final ckpt), run on the HOST, serially.
# Each demo runs through its own step scripts (4_deploy_runtime.sh where it needs one, 5_run.sh, 6_cleanup.sh
# for the per-run resources), under the shared GPU lock, with slime pinned off GPUs 0/1 (other users').
# The final checkpoint goes to <demo>/ckpt/smoke_all and is deleted after the run (root-owned → via the container).
# Results: _smoke_logs/smoke_all_results.tsv (demo, status, secs, log).
#
#   bash run_all_demos_smoke.sh                       # all entries below
#   ONLY="demo_gpu_farm:whole demo_cpu_tasks" bash run_all_demos_smoke.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCK="${LOCK:-/tmp/acs_gpu.lock}"
NR="${NR:-3}"; EVAL_INTERVAL="${EVAL_INTERVAL:-2}"; EVAL_NS="${EVAL_NS:-2}"
OUT="$HERE/_smoke_logs"; RES="$OUT/smoke_all_results.tsv"
mkdir -p "$OUT"; [ -f "$RES" ] || printf 'demo\tstatus\tsecs\tlog\n' > "$RES"

PIN6="SLIME_CUDA_VISIBLE=2,3,4,5,6,7 SLIME_NUM_GPUS=6"
PIN4="SLIME_CUDA_VISIBLE=2,3,4,5 SLIME_NUM_GPUS=4"      # GPUs 6,7 belong to the local farm
# demo[:label] | extra env | deploy before (host) | cleanup after (host)   (label only names the log/row)
ENTRIES=(
  "demo_cpu_tasks|$PIN6 ACTOR_GPUS=4 ROLLOUT_GPUS=2||"
  "demo_cpu_terminalbench|$PIN6 ACTOR_GPUS=4 ROLLOUT_GPUS=2||"
  "demo_cpu_swebench|$PIN6 ACTOR_GPUS=4 ROLLOUT_GPUS=2||"
  "demo_cpu_instance_pool|$PIN6 ACTOR_GPUS=4 ROLLOUT_GPUS=2||"
  "demo_cpu_scale_lanes|$PIN6 ACTOR_GPUS=2 ROLLOUT_GPUS=4||"
  "demo_gpu_single|$PIN6 ACTOR_GPUS=4 ROLLOUT_GPUS=2|2_build_image.sh 3_push_ecr.sh 4_deploy_runtime.sh|6_cleanup.sh"
  "demo_gpu_farm:fractional|$PIN4 ACTOR_GPUS=2 ROLLOUT_GPUS=2 FARM=fractional|2_build_image.sh 4_deploy_runtime.sh|6_cleanup.sh"
  "demo_gpu_farm:whole|$PIN4 ACTOR_GPUS=2 ROLLOUT_GPUS=2 FARM=whole|2_build_image.sh 4_deploy_runtime.sh|6_cleanup.sh"
)

for e in "${ENTRIES[@]}"; do
  IFS='|' read -r d knobs pre post <<<"$e"
  [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $d "* ]] && continue
  dir="${d%%:*}"; name="${d/:/-}"
  log="$OUT/smoke_all_$name.log"; t0=$SECONDS; st=PASS
  echo "[smoke-all] $d  $(date -u +%FT%TZ)  (waiting for $LOCK)"
  (
    flock 9
    cd "$HERE/$dir"
    for s in $pre; do echo "[smoke-all] $d: $s"; env $knobs bash "$s" || exit 10; done
    env $knobs NR="$NR" EVAL_INTERVAL="$EVAL_INTERVAL" EVAL_NS="$EVAL_NS" \
        SAVE="/work/code/demos/$dir/ckpt/smoke_all" RUN_TAG=smoke_all bash 5_run.sh
    rc=$?
    for s in $post; do echo "[smoke-all] $d: $s"; env $knobs bash "$s"; done
    docker exec agentcore_p4 rm -rf "/work/code/demos/$dir/ckpt/smoke_all"
    exit $rc
  ) 9>"$LOCK" > "$log" 2>&1
  rc=$?
  if [ $rc = 10 ]; then st="FAIL(deploy)"
  elif [ $rc != 0 ] || ! grep -aq "succeeded" "$log"; then st="FAIL(rc=$rc)"; fi
  printf '%s\t%s\t%s\t%s\n' "$d" "$st" $((SECONDS-t0)) "$log" | tee -a "$RES"
done
echo "[smoke-all] done $(date -u +%FT%TZ)"; cat "$RES"
