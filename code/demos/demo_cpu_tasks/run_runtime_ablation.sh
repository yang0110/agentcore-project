#!/bin/bash
# Runtime-count throughput ablation for Demo A (the mini-harness family zoo).
#
# QUESTION: does spreading concurrent rollout episodes across MORE remote sandbox Runtimes raise
# rollout goodput? We sweep the NUMBER of sandbox Runtimes (AGENTCORE_SANDBOX_RUNTIME_ARNS,
# round-robin in agentcore_slime_demo/sandbox.py) over RUNTIMES={1 2 4}, running the SAME 5-step
# smoke config (NR=6 => 5 train steps + 1 eval at step 5, turns=3, resp-len 1024) for each FAMILY.
# Per (runtime_count, family) we record goodput (useful tok/s) + MFU + step_time + eval reward.
#
# This is a REAL-AWS run: each point drives real AgentCore CPU sandbox Runtimes. Each run grabs all
# 8 GPUs + its own ray head, so points are strictly sequential.
#
# Launch INSIDE the agentcore_p4 container:
#   sudo -n docker exec agentcore_p4 bash -lc \
#     'cd /work/code/demos/demo_cpu_tasks && ALL_ARNS=<arn1,arn2,arn3,arn4> bash run_runtime_ablation.sh \
#        > /work/code/demos/demo_cpu_tasks/logs/ablation.out 2>&1'
set -uo pipefail

# Master comma-separated list of DEPLOYED CPU sandbox Runtime ARNs (>= max(RUNTIMES)). The sweep
# uses the first K of these for runtime-count K.
ALL_ARNS="${ALL_ARNS:?set ALL_ARNS to a comma-separated list of >=4 deployed CPU sandbox Runtime ARNs}"
RUNTIMES="${RUNTIMES:-1 2 4}"
FAMILIES="${FAMILIES:-coding regex planning config data swe tool_use sql math ctf rag}"

# Same 5-step smoke config for every point (fixed so only runtime_count / family vary).
export SANDBOX=agentcore-runtime
export NR="${NR:-6}"                 # 5 train steps (steps = NR-1) + 1 eval
export EVAL_INTERVAL="${EVAL_INTERVAL:-5}"
export CONC="${CONC:-16}"            # episode concurrency: must exceed 1 for fan-out to matter
export MODE="${MODE:-B}"            # fully-async (deep buffer) — the goodput regime
export AWS_REGION="${AWS_REGION:-us-west-2}"
export AGENTCORE_REGION="${AGENTCORE_REGION:-us-west-2}"

cd /work/code/demos/demo_cpu_tasks
mkdir -p logs
SUMMARY=logs/ablation_summary.csv
echo "runtime_count,family,rc,mean_goodput_tok_per_s,mean_mfu,mean_step_time,mean_rollout_time,mean_effective_tok_per_gpu_s,mean_sandbox_ms_p50,mean_sglang_gen_ms_p50,eval_reward" > "$SUMMARY"

# first K comma-separated entries of ALL_ARNS
first_k() { echo "$ALL_ARNS" | tr ',' '\n' | sed -n "1,${1}p" | paste -sd, -; }

echo "=== ablation START $(date -u) runtimes=[$RUNTIMES] families=[$FAMILIES] ===" | tee -a logs/ablation.log
for K in $RUNTIMES; do
   arns="$(first_k "$K")"
   n_have=$(echo "$arns" | tr ',' '\n' | grep -c .)
   if [ "$n_have" -lt "$K" ]; then
      echo "!!! only $n_have ARNs available, need $K — skipping K=$K" | tee -a logs/ablation.log
      continue
   fi
   export AGENTCORE_SANDBOX_RUNTIME_ARNS="$arns"
   for f in $FAMILIES; do
      tag="r${K}_${f}"
      echo "--- $(date -u) BEGIN K=$K family=$f arns=$arns ---" | tee -a logs/ablation.log
      # Scratch checkpoint dir (weights not needed for a throughput/MFU smoke). Fresh each run so
      # the final forced save never overwrites a stale dir; deleted right after metrics extraction.
      SAVE_DIR="$PWD/ckpt/_abl_${tag}"
      rm -rf "$SAVE_DIR"
      SAVE="$SAVE_DIR" FAMILY="$f" bash run.sh > "logs/ablate_${tag}.log" 2>&1
      rc=$?
      # Parse goodput/MFU/step_time from the run log; summarize means across the 5 steps.
      python3 metrics.py "logs/ablate_${tag}.log" --csv "logs/ablate_${tag}.csv" \
              --eval-csv "logs/ablate_${tag}.eval.csv" >/dev/null 2>&1
      python3 - "$K" "$f" "$rc" "logs/ablate_${tag}.csv" "logs/ablate_${tag}.eval.csv" "$SUMMARY" <<'PY'
import csv, sys, statistics as st
K, fam, rc, csvp, evalp, summ = sys.argv[1:7]
def col(path, name):
    vals=[]
    try:
        for row in csv.DictReader(open(path)):
            v=row.get(name,"")
            if v not in ("",None):
                try: vals.append(float(v))
                except ValueError: pass
    except FileNotFoundError: pass
    return vals
def mean(xs): return round(st.mean(xs),4) if xs else ""
gp=mean(col(csvp,"perf/goodput_tok_per_s"))
mfu=mean(col(csvp,"perf/mfu"))
stp=mean(col(csvp,"perf/step_time"))
# rollout-side throughput — the direct target of the runtime-count ablation
rt=mean(col(csvp,"perf/rollout_time"))
eff=mean(col(csvp,"perf/effective_tokens_per_gpu_per_sec"))
# sandbox response time vs pure SGLang decode time (split out of rollout_time)
sbx=mean(col(csvp,"perf/sandbox_ms_p50"))
gen=mean(col(csvp,"perf/sglang_gen_ms_p50"))
# eval reward: first eval/* column that isn't eval/step
er=""
try:
    r=next(csv.DictReader(open(evalp)))
    for k,v in r.items():
        if k.startswith("eval/") and k!="eval/step" and v not in ("",None):
            er=v; break
except (FileNotFoundError, StopIteration): pass
open(summ,"a").write(f"{K},{fam},{rc},{gp},{mfu},{stp},{rt},{eff},{sbx},{gen},{er}\n")
print(f"    K={K} {fam} rc={rc} goodput={gp} mfu={mfu} step_time={stp} rollout_time={rt} eff_tok/gpu/s={eff} sandbox_p50={sbx} sglang_gen_p50={gen} eval={er}")
PY
      rm -rf "$SAVE_DIR"   # reclaim the ~53GB scratch checkpoint
      # clean ray/sglang between points so the next starts from a clean 8-GPU slate
      pkill -9 sglang 2>/dev/null || true
      ray stop --force >/dev/null 2>&1 || true
      pkill -9 -f "ray::" 2>/dev/null || true
      sleep 8
   done
done
echo "=== ablation DONE $(date -u) — summary: $SUMMARY ===" | tee -a logs/ablation.log
column -s, -t "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
