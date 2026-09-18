#!/bin/bash
# Serial re-run of all 8 standalone per-task CPU demos, end-to-end (smoke: NR=3 EVAL_INTERVAL=2),
# to confirm the per-task reorganization introduced no bugs. Each run uses ALL 8 GPUs, so runs are
# strictly serial; between demos we wait for GPU memory to drain before starting the next.
# Run INSIDE agentcore_p4:  bash /work/code/demos/rerun_all_standalone.sh
set -u
DEMOS=/work/code/demos
PREFIXES="gsm8k math500 deepscaler humaneval gretelsql hotpotqa xlam mcp"
export NR=3 EVAL_INTERVAL=2 SANDBOX=agentcore-runtime

drain_gpu() {  # wait up to ~90s for GPU mem to fall below 500 MiB total
  for i in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
    [ "${used:-99999}" -lt 500 ] && return 0
    sleep 3
  done
  return 1
}

echo "############ RE-RUN START $(date -u +%FT%TZ) ############"
for p in $PREFIXES; do
  d="$DEMOS/demo_cpu_$p"
  log="$d/rerun.log"
  echo ""
  echo "=================== START demo_cpu_$p  $(date -u +%TZ) ==================="
  ( cd /work && bash "$d/run.sh" ) > "$log" 2>&1
  rc=$?
  # local cleanup so the next demo starts clean
  pkill -9 -f "train_async|sglang|ray::" 2>/dev/null
  ray stop --force >/dev/null 2>&1
  # eval result(s) from the log
  evline=$(grep -oE "eval/$p[^ ]*[: ]+[0-9.]+" "$log" | tail -3 | tr '\n' ' ')
  [ -z "$evline" ] && evline=$(grep -oE "eval/[a-z0-9_]+ [0-9.]+" "$log" | tail -3 | tr '\n' ' ')
  if [ $rc -eq 0 ]; then
    echo "RESULT demo_cpu_$p: OK (rc=0)  eval=[ ${evline:-none captured} ]"
  else
    echo "RESULT demo_cpu_$p: FAIL (rc=$rc)  last log lines:"
    tail -15 "$log" | sed 's/^/    /'
  fi
  drain_gpu || echo "WARN demo_cpu_$p: GPU did not drain below 500MiB after cleanup"
done
echo ""
echo "############ RE-RUN DONE $(date -u +%FT%TZ) ############"
echo "=== SUMMARY ==="
for p in $PREFIXES; do
  log="$DEMOS/demo_cpu_$p/rerun.log"
  if grep -q "Training finished\|save.*iter\|eval/" "$log" 2>/dev/null; then st=ran; else st="?"; fi
  echo "demo_cpu_$p: log=$log"
done
