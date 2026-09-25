#!/bin/bash
# demo_cpu_tasks — ONE demo over many RLVR datasets AND their mixes (DATASET=mbpp|gsm8k|…, default
# mbpp; DATASET=a,b,c or DATASET=mixed for a MIXED-task batch — see the DATASET selector below; a single
# dataset is just the one-element mix). REMOTE-sandbox, OUR multi-turn run_episode harness. REAL slime training,
# ASYNC (decoupled) mode. This dir owns its data/ + ckpt/, but imports the SHARED agentcore_slime_demo
# library from its own vendored copy in this dir (on PYTHONPATH). Runs INSIDE the persistent
# container agentcore_p4 (slimerl/slime).
#
# Harness/sandbox split (SAME decoupled shape as demo_gpu_single): the run_episode harness runs IN-PROCESS
# on this box inside generate.generate, and the sandbox (write_file / exec / read_file) is a
# SINGLE remote AgentCore Runtime (sandbox_server_agent.py), reached over invoke_agent_runtime.
# There is only ONE runtime here — the sandbox; there is NO separate harness/adapter Runtime.
# Unlike demo_gpu_single (KernelBench needs CUDA => GPU g6 sandbox), this demo's checkers (MBPP, math, SQL, …) are plain
# Python/IO => it points at a CPU-backed sandbox Runtime (e.g. the a2 CPU sandbox acs_sandbox_a2
# on the acs_cpu_cp capacity provider).
#
# Async/decoupled: agentic rollouts have long, variable latency (multi-turn code repair in a
# sandbox). slime's async driver (train_async.py) runs trainer GPUs and SGLang inference GPUs on
# DISJOINT devices. Hard-requires no --colocate (train_async.py asserts it).
#
# ASYNC MODE (MODE=A|B, DEFAULT B) for the ablation — see the MODE block below and
# docs/sandbox-concurrency-ablation.md "Two async modes". The DEFAULT is FULLY-ASYNC (Mode B): on
# top of train_async.py's 1-step overlap it selects slime's --rollout-function-path
# generate_rollout_fully_async, a background worker holding a DEEP, warm buffer of in-flight +
# completed rollouts across step boundaries, so staleness genuinely GROWS with concurrency (CONC).
# That deep buffer is what lets the sandbox-concurrency ablation observe the real
# staleness/discard/clip trade-off. MODE=A drops the override and runs train_async.py's stock
# 1-step-async driver (staleness pinned to ~1 step regardless of CONC) for the A-vs-B comparison.
# Both modes share the identical custom generate hook + Sample contract
# (tokens/logprobs/loss-mask/reward); only the rollout DRIVER (and matching eval path) changes.
#
# GPU split (8xL40S): 4 train + 4 inference + 0 env. MBPP checker is plain Python/IO => CPU envs,
# so all 8 GPUs serve the policy (train + infer).
#
# Sandbox backend (SANDBOX): agentcore-runtime -> the SINGLE remote AgentCore Runtime sandbox
# (default; requires AGENTCORE_SANDBOX_RUNTIME_ARN); agentcore -> AWS Code Interpreter microVM;
# SANDBOX=local is DISABLED for CPU demos (remote-only; see the guard below). Tunables: MODE (A|B
# async mode), RB (rollout-batch-size), NS (n-samples-per-prompt), NR (num-rollout), TEMP
# (sampling temp), CONC (sglang-server-concurrency), EVAL_INTERVAL/EVAL_NS.
#
# CONCURRENCY KNOBS (decoupled, for the sandbox-concurrency ablation — see
# docs/sandbox-concurrency-ablation.md):
#   * CONC       -> SANDBOX/EPISODE concurrency. #episodes (= #concurrent remote sandboxes) in
#                   flight is S = CONC * num_engines = CONC * 2. This is the knob to SWEEP. In the
#                   fully-async (Mode B) path it also sets BUFFER DEPTH: the worker keeps up to
#                   S in-flight AND up to S completed groups queued ahead of the trainer (which
#                   drains RB/step), so raising CONC raises staleness. One knob moves both, which is
#                   exactly the trade-off the ablation measures.
#   * INFER_CONC -> INFERENCE concurrency I. A process-global semaphore around ONLY the SGLang
#                   decode POST, so at most I episodes decode at once no matter how large S is.
#                   Default 0 = disabled (I follows S, the original coupled behavior). Set it to a
#                   FIXED value (e.g. 8) to hold inference load constant while sweeping CONC, which
#                   isolates the effect of sandbox count from inference load.
#
# NOTE step count: slime resumes at loaded_iter+1; a fresh torch_dist ckpt is iter 0 => the loop
# runs range(1, NR) = (NR-1) real training steps. NR=11 => 10 train + 1 eval (SMOKE; no more needed).
# NOTE GRPO signal: group-relative advantage is 0 when every sample in a prompt's group gets the
# same reward. Reward variance within groups (harder tasks / higher temp) yields nonzero gradient.
set -euo pipefail

# VARIANT=bigimage (was demo_cpu_bigimage): the SAME trainer, but the sandbox is a LARGE (>2 GB) custom
# image on a CPU capacity-provider INSTANCE (m5), run warm-shared: SANDBOX=agentcore-ray-farm + one
# shared runtimeSessionId (cold-boot paid once) with a fresh CpuFarm env per episode sub_session.
# Deploy it with deploy/deploy_bigimage.py (or VARIANT=bigimage bash 2_..4_*.sh); defaults to DATASET=deepscaler.
VARIANT="${VARIANT:-}"
if [ "$VARIANT" = "bigimage" ]; then
   SANDBOX="${SANDBOX:-agentcore-ray-farm}"; DATASET="${DATASET:-${FAMILY:-deepscaler}}"
fi
SANDBOX="${SANDBOX:-agentcore-runtime}"   # agentcore-runtime -> SINGLE remote AgentCore Runtime sandbox (default);
                                          # agentcore -> AWS Code Interpreter microVM. CPU demo => REMOTE-ONLY (no local).
# CPU demos run ONLY on remote managed serverless AgentCore sandbox sessions — never local. The local
# (in-container LocalSandbox) path is rejected here so a CPU training run can never silently fall back
# off the AgentCore substrate the demo is meant to exercise. (GPU demos keep a local fallback because
# a remote g6 instance may be unavailable.)
if [ "$SANDBOX" = "local" ]; then
   echo "SANDBOX=local is not supported for CPU demos (remote-only) — use agentcore-runtime (default) or agentcore." >&2
   exit 1
fi
# The remote sandbox Runtime ARN (sandbox_server_agent.py). REQUIRED for SANDBOX=agentcore-runtime.
# this demo's checkers (MBPP, math, SQL, …) are pure Python => a CPU-backed sandbox Runtime (e.g. acs_sandbox_a2 on acs_cpu_cp).
# AGENTCORE_SANDBOX_RUNTIME_ARNS (plural, comma-separated) fans episodes out round-robin across
# SEVERAL sandbox Runtimes — the runtime-count throughput ablation knob. Either the plural list OR
# the single ARN satisfies the requirement below.
# PINNED to K=2: the runtime-count ablation (logs/ablation_summary.csv) found throughput flat
# across K=1/2/4 (sandbox is not the bottleneck for this demo's pure-Python CPU checker — the 2 SGLang
# engines / trainer are), with K=2 the marginal goodput peak. Runtime count is NOT tied to
# num_engines; it only spreads S=CONC*num_engines concurrent sessions across K endpoints. Override
# by exporting AGENTCORE_SANDBOX_RUNTIME_ARNS (or ..._ARN) to sweep again.
AGENTCORE_SANDBOX_RUNTIME_ARNS="${AGENTCORE_SANDBOX_RUNTIME_ARNS:-arn:aws:bedrock-agentcore:us-west-2:<ACCOUNT_ID>:runtime/acs_sandbox_cpu-1gzBVp675S,arn:aws:bedrock-agentcore:us-west-2:<ACCOUNT_ID>:runtime/acs_sandbox_cpu_1-adRZyEHM6g}"
if [ "$SANDBOX" = "agentcore-runtime" ] && [ -z "$AGENTCORE_SANDBOX_RUNTIME_ARNS" ]; then
   AGENTCORE_SANDBOX_RUNTIME_ARN="${AGENTCORE_SANDBOX_RUNTIME_ARN:?set AGENTCORE_SANDBOX_RUNTIME_ARN (or ..._ARNS for the multi-Runtime sweep) to the CPU sandbox Runtime ARN (sandbox_server_agent.py; e.g. acs_sandbox_cpu from deploy_two_runtime.py)}"
fi
# SHARED-INSTANCE CPU FARM (SANDBOX=agentcore-ray-farm, the bigimage variant): all episodes invoke under
# ONE shared runtimeSessionId (B4_SHARED_RUNTIME_SESSION) so they land on the SAME warm m5 — the ~60-80 s
# capacity-provider cold-boot is paid ONCE per run — while the server's SANDBOX_FARM=cpu CpuFarm gives
# every per-episode sub_session a FRESH isolated env (own workdir + HOME/TMPDIR/caches, wiped on close).
# The ray-farm client reads ONLY the singular ARN (the baked K=2 ARNS list above is ignored here).
if [ "$SANDBOX" = "agentcore-ray-farm" ]; then
   AGENTCORE_SANDBOX_RUNTIME_ARN="${AGENTCORE_SANDBOX_RUNTIME_ARN:?set AGENTCORE_SANDBOX_RUNTIME_ARN to the big-image sandbox Runtime ARN printed by deploy/deploy_bigimage.py}"
   B4_SHARED_RUNTIME_SESSION="${B4_SHARED_RUNTIME_SESSION:-bigcpu-farm-$(date +%s)-$(head -c8 /dev/urandom | od -An -tx1 | tr -d ' \n')}"
   echo "shared-instance CPU farm: all episodes reuse runtimeSessionId=${B4_SHARED_RUNTIME_SESSION} (one warm m5)" >&2
fi
RB="${RB:-4}"          # prompts per rollout step
NS="${NS:-8}"          # GRPO group size (samples per prompt)
NR="${NR:-11}"        # num-rollout; real training steps = NR-1 (see NOTE) => 10 train + 1 eval (SMOKE; no more needed).
                       # Keep the train set >= NR*RB prompts so one epoch spans the whole run and
                       # the ONLY checkpoint is the final one (slime force-saves at epoch
                       # boundaries too). The MBPP pool is built at 500 prompts (>= 100*4).
TEMP="${TEMP:-1.0}"    # rollout sampling temperature (higher => more intra-group diversity)
GBS=$(( RB * NS ))     # global batch = all samples, 1 optimizer step per rollout
# ASYNC MODE (the ablation's two async modes — see docs/sandbox-concurrency-ablation.md):
#   B (DEFAULT) = FULLY-ASYNC. slime's generate_rollout_fully_async runs a background worker with a
#                 DEEP, warm buffer of in-flight + completed rollouts across step boundaries, so
#                 staleness genuinely GROWS with concurrency (CONC). This is the vehicle the
#                 sandbox-concurrency ablation needs to observe the real staleness/discard/clip
#                 trade-off. Eval routes through our mode_b_eval wrapper (see EVAL_ARGS).
#   A           = 1-STEP-ASYNC (stock). train_async.py's default rollout path: exactly ONE rollout
#                 step overlaps the trainer, so staleness is pinned to ~1 step regardless of CONC.
#                 Kept reachable from this same runner for the A-vs-B comparison. Eval uses the
#                 stock sync rollout (no fully-async worker to quiesce).
# Both modes run under train_async.py (no --colocate) and share the identical custom-generate hook
# + Sample contract; only the rollout DRIVER (and the matching eval path) changes.
MODE="${MODE:-B}"
case "$MODE" in B|b) MODE=B;; A|a) MODE=A;; *) echo "MODE must be A or B (got '$MODE')" >&2; exit 1;; esac
# Held-out eval every N steps (0 => off). Held-out solve-rate is a KEY metric, so eval stays ON
# under fully-async (Mode B). It does NOT work through the stock rollout path, though: slime's
# GenerateState is a process singleton whose asyncio.Semaphore binds to the FIRST event loop that
# acquires it — the background fully-async worker's loop — while eval runs generation on the MAIN
# loop, which would crash with "Semaphore is bound to a different event loop". We route eval
# through agentcore_slime_demo.mode_b_eval.generate_rollout (see EVAL_ARGS), which quiesces the
# worker and rebuilds the singleton on the eval loop so stock eval runs cleanly. NOTE: each eval
# discards the worker's warm buffer (it re-warms over the next few rollouts), so keep EVAL_INTERVAL
# well above the re-warm time (default 10 is fine for the smoke; use a larger value for long runs).
EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
EVAL_NS="${EVAL_NS:-2}"               # samples per eval prompt
# Checkpoint policy (disk-saving): we do NOT snapshot mid-training — the ONLY checkpoint written is
# the final one. slime saves whenever should_run_periodic_action() fires; here that is ONLY the last
# rollout (rollout_id == num_rollout-1, always forced). Its two other triggers are both inactive:
# (a) step % save-interval == 0 is disabled by the huge SAVE_INTERVAL below, and (b) the
# epoch-boundary trigger (step % num_rollout_per_epoch) is OFF because we pass an explicit
# --num-rollout (NR) => slime leaves num_rollout_per_epoch = None (it is computed only under
# --num-epoch). Net: exactly one checkpoint, at the end of training, regardless of dataset size.
SAVE_INTERVAL="${SAVE_INTERVAL:-1000000}"

SLIME=/root/slime

# CAPABILITY-ROUTING BY IMAGE (the custom-image mix). The `datasci` family (in DATASET=mixed) needs
# pandas — absent from the generic CPU image. Each task carries an `image` tag (task.image: "cpu" for
# the stdlib families, "datasci" for the pandas one), and make_sandbox routes an episode to the
# Runtime whose image matches, via this JSON map:
#     {"cpu": <CPU sandbox ARN>, "datasci": <pandas sandbox ARN>}
# Build + deploy both images with the step scripts (bash 2_build_image.sh && bash 3_push_ecr.sh &&
# bash 4_deploy_runtime.sh — they write this map to deploy/runtime.env), or set it here by hand. If unset (or a
# tag is missing from the map), that image falls back to the round-robin CPU pool above — datasci
# episodes then fail to import pandas and score 0. Single-dataset runs never need it.
AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE="${AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE:-}"

DEMO_DIR=/work/code/demos/demo_cpu_tasks
CKPT="${CKPT_BASE:-/work/code/p4_slime_train/ckpt/Qwen3-4B_torch_dist}"  # SHARED base Megatron ckpt
HF=/root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c

# ---- Demo A config (baked in) --------------------------------------------------------------
ACTOR_GPUS="${ACTOR_GPUS:-4}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-4}"

# DATASET selector — ONE demo, many datasets, and any MIX of them. The in-process harness is
# task-agnostic: the SAME run_episode loop trains every dataset below, differing only in the seed rows
# + the hidden checker (see agentcore_slime_demo/task_families.py). Each row self-describes its task in
# metadata.task, and generate._task_from_sample rebuilds the right task PER ROW — so a multi-family
# batch is transparent to the harness. Each dataset has its own data/<name>_{train,eval}.jsonl (built by
# prepare_data.sh), its own eval/<name> metric and its own ckpt/Qwen3-4B_<name>_async.
#   real HF (RLVR):  mbpp (default; alias coding) gsm8k math500 deepscaler humaneval gretelsql
#                    hotpotqa xlam mcp
#   synthetic zoo:   math swe tool_use sql ctf rag regex config data planning (build_family_datasets.py)
#   mixed:           the prebuilt math+sql+ctf+datasci mix (build_mixed_datasets.py; datasci needs the
#                    pandas image via AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE above)
# MIX: DATASET=a,b,c interleaves those datasets' train/eval files round-robin (balanced: each truncated
# to the shortest) into ckpt/_mix/<a+b+c>_{train,eval}.jsonl, eval metric eval/mix. GRPO normalizes
# advantage WITHIN a same-prompt group (one family); mixing is ACROSS groups, so the families' reward
# scales never need reconciling. A single dataset is the one-element mix (no interleave step).
# FAMILY is kept as an alias of DATASET (older scripts pass FAMILY=<name>).
DATASET="${DATASET:-${FAMILY:-mbpp}}"
IFS=',' read -r -a _DSETS <<< "$DATASET"
for i in "${!_DSETS[@]}"; do [ "${_DSETS[$i]}" = "coding" ] && _DSETS[$i]=mbpp; done
if [ -z "${DATA:-}" ]; then
  for d in "${_DSETS[@]}"; do
    if [ ! -f "$DEMO_DIR/data/${d}_train.jsonl" ]; then
      echo "unknown DATASET=$d (no $DEMO_DIR/data/${d}_train.jsonl). Available:" >&2
      ls "$DEMO_DIR/data" | sed -n 's/_train\.jsonl$//p' | tr '\n' ' ' >&2; echo >&2
      exit 2
    fi
  done
fi
if [ "${#_DSETS[@]}" -gt 1 ]; then
  _DS="$(IFS=+; echo "${_DSETS[*]}")"; _EVAL_NAME=mix
  if [ -z "${DATA:-}" ]; then
    mkdir -p "$DEMO_DIR/ckpt/_mix"
    for split in train eval; do
      python3 - "$DEMO_DIR/ckpt/_mix/${_DS}_${split}.jsonl" \
          $(for d in "${_DSETS[@]}"; do printf '%s ' "$DEMO_DIR/data/${d}_${split}.jsonl"; done) <<'PY'
import sys
out, srcs = sys.argv[1], sys.argv[2:]
rows = [[l for l in open(p) if l.strip()] for p in srcs]
n = min(len(r) for r in rows)
with open(out, "w") as f:
    for i in range(n):
        for r in rows:
            f.write(r[i] if r[i].endswith("\n") else r[i] + "\n")
print(f"[mix] {out}: {n} rows x {len(rows)} datasets", file=sys.stderr)
PY
    done
    DATA="$DEMO_DIR/ckpt/_mix/${_DS}_train.jsonl"
    EVAL_DATA="${EVAL_DATA:-$DEMO_DIR/ckpt/_mix/${_DS}_eval.jsonl}"
  fi
else
  _DS="${_DSETS[0]}"; _EVAL_NAME="$_DS"
fi
DATA="${DATA:-$DEMO_DIR/data/${_DS}_train.jsonl}"
EVAL_DATA="${EVAL_DATA:-$DEMO_DIR/data/${_DS}_eval.jsonl}"
EVAL_NAME="${EVAL_NAME:-$_EVAL_NAME}"
SAVE="${SAVE:-$DEMO_DIR/ckpt/Qwen3-4B_${_DS}${VARIANT:+_$VARIANT}_async}"

# clean any prior ray/sglang from a previous run
pkill -9 sglang 2>/dev/null || true; ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true; sleep 2

set -ex
export PYTHONUNBUFFERED=1
source "$SLIME/scripts/models/qwen3-4B.sh"

CKPT_ARGS=(
   --hf-checkpoint "$HF"
   --ref-load "$CKPT"
   --load "$CKPT"
   --save "$SAVE"
   --save-interval "$SAVE_INTERVAL"   # save-only-at-end (see SAVE_INTERVAL note above)
   # ---- Checkpoint-save deadlock fix (two flags that MUST go together) ---------------------
   # Megatron's checkpoint writer forks a writer process off the MAIN training process
   # (mp.get_context("fork")); forking a process holding CUDA/NCCL/inductor threads can inherit a
   # locked mutex with no thread to release it -> the writer hangs forever AFTER the .distcp files
   # are written. The safe backend is Megatron's SPAWN-based PersistentAsyncCaller, selected only
   # when BOTH --async-save is set AND init_persistent_async_worker() is called. slime forces
   # use_persistent_ckpt_worker=True but its own init() bypasses megatron.training.initialize, so
   # it NEVER makes that call -> --async-save alone falls back to the fork-based caller and can
   # still deadlock. So we supply the missing call via slime's supported init hook:
   --async-save
   --custom-megatron-init-path agentcore_slime_demo.megatron_init.enable_persistent_async_ckpt
)

# Held-out EVAL. The eval rollout FUNCTION depends on the async MODE:
#   Mode B (fully-async): the stock rollout path can't run eval (the fully-async worker's event loop
#     owns slime's GenerateState singleton semaphore, so in-process eval would crash with "Semaphore
#     is bound to a different event loop"). We route eval through
#     agentcore_slime_demo.mode_b_eval.generate_rollout, which quiesces the fully-async worker (waits
#     for its loop thread to fully die), drops the GenerateState singleton so the semaphore rebuilds
#     on THIS loop, runs slime's stock eval unchanged, then drops the singleton again so the next
#     training rollout's worker rebuilds it cleanly on the worker loop. (generate_rollout_fully_async
#     itself raises on evaluation=True, which is exactly why eval needs its own function-path.)
#   Mode A (1-step-async): no background worker exists, so eval uses slime's stock sync rollout.
if [ "$MODE" = "B" ]; then
   EVAL_FN=agentcore_slime_demo.mode_b_eval.generate_rollout
else
   EVAL_FN=slime.rollout.sglang_rollout.generate_rollout
fi
EVAL_ARGS=()
if [ "$EVAL_INTERVAL" != "0" ] && [ -f "$EVAL_DATA" ]; then
   EVAL_ARGS=(
      --eval-interval "$EVAL_INTERVAL"
      --eval-prompt-data "$EVAL_NAME" "$EVAL_DATA"
      --n-samples-per-eval-prompt "$EVAL_NS"
      --eval-temperature "$TEMP"
      --eval-function-path "$EVAL_FN"
   )
fi

ROLLOUT_ARGS=(
   --prompt-data "$DATA"
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --rollout-shuffle
   --num-rollout "$NR"
   --rollout-batch-size "$RB"
   --n-samples-per-prompt "$NS"
   --rollout-max-response-len 1024
   --rollout-temperature "$TEMP"
   --global-batch-size "$GBS"
   # NOTE: the rollout DRIVER (--rollout-function-path) is set by MODE_ARGS below, not here, so it
   # tracks the A/B async MODE. The custom generate hook is the same for both modes: the worker
   # (Mode B) and the stock driver (Mode A) both call slime's generate_and_rm_group, which
   # dispatches to our --custom-generate-function-path, so our token-ids / logprobs / loss-mask /
   # reward Sample contract is preserved UNCHANGED (reward set on the Sample => no --custom-rm-path).
   --custom-generate-function-path agentcore_slime_demo.generate.generate
)

# Async MODE driver flags (see the MODE note near the top). Mode B swaps in slime's fully-async
# rollout: a background worker that keeps a fixed pool of in-flight trajectories (concurrency =
# sglang_server_concurrency * num_engines = CONC*2) AND up to that many COMPLETED groups queued
# ahead of the trainer across rollout boundaries. The trainer drains only RB groups/step, so the
# queue grows deep and samples age => staleness scales with CONC. Mode A omits the override, leaving
# train_async.py's stock 1-step-async driver (staleness pinned to ~1 step regardless of CONC).
MODE_ARGS=()
if [ "$MODE" = "B" ]; then
   MODE_ARGS=(--rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async)
fi

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus "$ROLLOUT_GPUS"
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.7
   # SANDBOX/EPISODE concurrency S. slime gates custom-generate with
   # Semaphore(sglang_server_concurrency * num_engines), so with num_engines = ROLLOUT_GPUS /
   # per-engine = 4/2 = 2, the number of episodes (= concurrent remote sandboxes) in flight is
   # S = CONC * 2. This is the knob to SWEEP for the sandbox-concurrency ablation
   # (docs/sandbox-concurrency-ablation.md). Each episode opens an HTTP session AND a remote
   # sandbox session, so raising it also raises client-side load (asyncio thread pool, boto3
   # clients) — see the ablation doc's "can the in-process harness handle large C" section.
   --sglang-server-concurrency "${CONC:-4}"
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
# Give ray ALL 8 physical GPUs. slime's placement reserves (ACTOR_GPUS + ROLLOUT_GPUS) of them.
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus 8 \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM:${DEMO_DIR}:/root/slime\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"SANDBOX\": \"${SANDBOX}\",
    \"AGENTCORE_SANDBOX_RUNTIME_ARN\": \"${AGENTCORE_SANDBOX_RUNTIME_ARN:-}\",
    \"AGENTCORE_SANDBOX_RUNTIME_ARNS\": \"${AGENTCORE_SANDBOX_RUNTIME_ARNS:-}\",
    \"B4_SHARED_RUNTIME_SESSION\": \"${B4_SHARED_RUNTIME_SESSION:-}\",
    \"AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE\": \"$(printf '%s' "${AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE:-}" | sed 's/\\/\\\\/g; s/\"/\\\"/g')\",
    \"AWS_REGION\": \"${AWS_REGION:-us-west-2}\",
    \"AWS_BEARER_TOKEN_BEDROCK\": \"${AWS_BEARER_TOKEN_BEDROCK:-}\",
    \"INFER_CONC\": \"${INFER_CONC:-0}\"
  }
}"

cd "$SLIME"
# ASYNC driver + DECOUPLED placement: train_async.py, NO --colocate. Use an ABSOLUTE path: the ray
# job entrypoint's cwd is the ray-head cwd (/work here), not this script's `cd "$SLIME"`.
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 "$SLIME/train_async.py" \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "$ACTOR_GPUS" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${MODE_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"
