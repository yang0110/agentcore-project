#!/bin/bash
# demo_cpu_harness_scale — REAL slime CPU trainer that SCALES THE HARNESS TIER. Same async/decoupled
# training shape as demo_cpu_mbpp, but the custom-generate hook hands every episode to a WARM
# in-process harness worker pool (agentcore_slime_demo.harness_pool.HarnessPool) instead of driving
# one run_episode inline. The pool owns W = HARNESS_WORKERS warm workers + one shared AnthropicAdapter;
# a free worker PULLS the next episode off a shared asyncio.Queue (work-stealing), runs the
# MiniClaudeHarness loop (LLM via the adapter, tools in a fresh remote CPU sandbox Runtime), scores
# it (RLVR numeric check, tol 1e-4), and finish_session drains the trajectory into loss-masked GRPO
# samples. This dir owns its data/ + ckpt/ and imports its own vendored agentcore_slime_demo copy.
# Runs INSIDE the persistent container agentcore_p4 (slimerl/slime).
#
# THE PROPERTY: async-RL rollout throughput = min( inference , sandbox , harness ). The harness is a
# replicated tier too; hold it single and it silently becomes the ceiling. W = HARNESS_WORKERS is the
# knob that lifts the harness tier: slime admits up to S = CONC * num_engines episodes concurrently,
# but only W are actively DRIVEN by a harness at once, so raising W lifts the harness tier until it
# hits the scarcer of inference/sandbox. The OFFLINE min-of-tiers sweep (which plots the knee +
# plateau across W, and a balanced inference=sandbox=harness=N joint sweep) lives in
# harness_scaling_demo.py — kept alongside this trainer as the analysis tool; see README.md.
#
# HARNESS POOL => STOCK (Mode A) DRIVER ONLY. harness_pool_generate.generate returns a LIST of
# Samples (finish_session's fan-out shape). slime's stock async rollout handles the fan-out; the
# fully-async (Mode B) driver's sort key chokes on the nested list, so this demo pins MODE=A and
# REJECTS MODE=B (guard below). Eval therefore uses slime's stock sync rollout (no fully-async worker
# to quiesce). This is the one structural difference from demo_cpu_mbpp (which defaults Mode B).
#
# ADAPTER TOOL PARSER (the A3/B3 dead-reward fix): SGLANG_ARGS bakes in
# --sglang-tool-call-parser qwen25 --sglang-reasoning-parser qwen3. Without them the AnthropicAdapter
# cannot turn Qwen3's raw tool syntax into tool_use blocks, the mini-harness tool loop is dead, and
# reward is structurally 0. Do not remove them.
#
# Sandbox backend (SANDBOX): agentcore-runtime -> remote AgentCore Runtime sandbox (default; K=2
# pinned CPU ARNs, round-robined per episode); agentcore -> AWS Code Interpreter microVM.
# SANDBOX=local is DISABLED for CPU demos (remote-only; guard below).
#
# NOTE step count: slime resumes at loaded_iter+1; a fresh torch_dist ckpt is iter 0 => the loop runs
# range(1, NR) = (NR-1) real training steps. NR=11 => 10 train + 1 eval (SMOKE; no more needed).
# NOTE GRPO signal: group-relative advantage is 0 when every sample in a prompt's group gets the same
# reward. Reward variance within groups (harder tasks / higher temp) yields nonzero gradient.
set -euo pipefail

SANDBOX="${SANDBOX:-agentcore-runtime}"   # agentcore-runtime -> remote AgentCore Runtime sandbox (default);
                                          # agentcore -> AWS Code Interpreter microVM. CPU demo => REMOTE-ONLY.
# CPU demos run ONLY on remote managed serverless AgentCore sandbox sessions — never local, so a CPU
# training run can never silently fall back off the AgentCore substrate the demo is meant to exercise.
if [ "$SANDBOX" = "local" ]; then
   echo "SANDBOX=local is not supported for CPU demos (remote-only) — use agentcore-runtime (default) or agentcore." >&2
   exit 1
fi
# The remote sandbox Runtime ARN(s). AGENTCORE_SANDBOX_RUNTIME_ARNS (plural, comma-separated) fans
# episodes out round-robin across SEVERAL sandbox Runtimes (the K sandbox tier). Pinned K=2 default;
# either the plural list OR the single ARN satisfies the requirement below.
AGENTCORE_SANDBOX_RUNTIME_ARNS="${AGENTCORE_SANDBOX_RUNTIME_ARNS:-arn:aws:bedrock-agentcore:us-west-2:<ACCOUNT_ID>:runtime/acs_sandbox_cpu-1gzBVp675S,arn:aws:bedrock-agentcore:us-west-2:<ACCOUNT_ID>:runtime/acs_sandbox_cpu_1-adRZyEHM6g}"
if [ "$SANDBOX" = "agentcore-runtime" ] && [ -z "$AGENTCORE_SANDBOX_RUNTIME_ARNS" ]; then
   AGENTCORE_SANDBOX_RUNTIME_ARN="${AGENTCORE_SANDBOX_RUNTIME_ARN:?set AGENTCORE_SANDBOX_RUNTIME_ARN (or ..._ARNS) to the CPU sandbox Runtime ARN (sandbox_server_agent.py)}"
fi
RB="${RB:-4}"          # prompts per rollout step
NS="${NS:-8}"          # GRPO group size (samples per prompt)
NR="${NR:-11}"         # num-rollout; real training steps = NR-1 (see NOTE) => 10 train + 1 eval.
                       # Keep the train set >= NR*RB prompts (deepscaler pool is 400 >= 11*4).
TEMP="${TEMP:-1.0}"    # rollout sampling temperature (higher => more intra-group diversity)
GBS=$(( RB * NS ))     # global batch = all samples, 1 optimizer step per rollout

# ---- THE HARNESS TIER -----------------------------------------------------------------------------
# W = HARNESS_WORKERS: the number of WARM in-process harness workers the HarnessPool spins up (read
# from the env inside the rollout process; passed through RUNTIME_ENV_JSON below). This is the knob
# this demo scales. EPISODE_MAX_TURNS bounds each episode's mini-harness tool loop.
HARNESS_WORKERS="${HARNESS_WORKERS:-4}"
EPISODE_MAX_TURNS="${EPISODE_MAX_TURNS:-4}"
EPISODE_MAX_TOKENS="${EPISODE_MAX_TOKENS:-1024}"

# HARNESS POOL requires the STOCK driver — reject fully-async (Mode B) up front (see header).
MODE="${MODE:-A}"
case "$MODE" in A|a) MODE=A;;
   B|b) echo "MODE=B (fully-async) is unsupported here: harness_pool_generate.generate returns a fan-out LIST of Samples, which slime's fully-async sort key cannot order. Use MODE=A (stock, the default)." >&2; exit 1;;
   *) echo "MODE must be A (got '$MODE')" >&2; exit 1;; esac

# Held-out eval every N steps (0 => off). With NR=11 eval fires once at step 10.
EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
EVAL_NS="${EVAL_NS:-2}"               # samples per eval prompt
# Checkpoint policy: the ONLY checkpoint written is the final one. A huge SAVE_INTERVAL disables the
# periodic save, and an explicit --num-rollout leaves num_rollout_per_epoch=None so the epoch-boundary
# save never fires; only slime's forced final-rollout checkpoint is written.
SAVE_INTERVAL="${SAVE_INTERVAL:-1000000}"

SLIME=/root/slime
DEMO_DIR=/work/code/demos/demo_cpu_harness_scale
CKPT="${CKPT_BASE:-/work/code/p4_slime_train/ckpt/Qwen3-4B_torch_dist}"  # SHARED base Megatron ckpt
HF=/root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c

# ---- placement ------------------------------------------------------------------------------------
ACTOR_GPUS="${ACTOR_GPUS:-4}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-4}"
# num_engines E = ROLLOUT_GPUS / per-engine(2) = 2. Default CONC so admission S = CONC*E ~= W (the
# balanced start point the README describes): CONC = ceil(W / E). Raise W to lift the harness tier;
# raise CONC to admit more episodes than there are workers (they queue for a free worker).
NENG=$(( ROLLOUT_GPUS / 2 ))
CONC="${CONC:-$(( (HARNESS_WORKERS + NENG - 1) / NENG ))}"

# Task family. One task, end-to-end: DeepScaleR math (coding-shape: solution.py sets a module-level
# `answer`; hidden run_checks.py numeric-asserts it, tol 1e-4). data/ ships populated.
FAMILY="${FAMILY:-deepscaler}"
_DS="$FAMILY"; _EVAL_NAME="$FAMILY"
DATA="${DATA:-$DEMO_DIR/data/${_DS}_train.jsonl}"
EVAL_DATA="${EVAL_DATA:-$DEMO_DIR/data/${_DS}_eval.jsonl}"
EVAL_NAME="${EVAL_NAME:-$_EVAL_NAME}"
SAVE="${SAVE:-$DEMO_DIR/ckpt/Qwen3-4B_${_DS}_harnessscale_async}"

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
   # Checkpoint-save deadlock fix (both flags MUST go together): --async-save + slime's supported
   # init hook, which calls init_persistent_async_worker() to select Megatron's SPAWN-based
   # PersistentAsyncCaller instead of the fork-based caller that can deadlock after writing .distcp.
   --async-save
   --custom-megatron-init-path agentcore_slime_demo.megatron_init.enable_persistent_async_ckpt
)

# Held-out EVAL through the STOCK sync rollout (Mode A): it calls the SAME custom-generate hook, so
# eval episodes also flow through the warm harness pool.
EVAL_ARGS=()
if [ "$EVAL_INTERVAL" != "0" ] && [ -f "$EVAL_DATA" ]; then
   EVAL_ARGS=(
      --eval-interval "$EVAL_INTERVAL"
      --eval-prompt-data "$EVAL_NAME" "$EVAL_DATA"
      --n-samples-per-eval-prompt "$EVAL_NS"
      --eval-temperature "$TEMP"
      --eval-function-path slime.rollout.sglang_rollout.generate_rollout
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
   # THE HARNESS-POOL HOOK: hands each episode to the warm worker pool (W = HARNESS_WORKERS). Returns
   # a fan-out LIST of Samples from finish_session; stock driver only (see header MODE guard).
   --custom-generate-function-path agentcore_slime_demo.harness_pool_generate.generate
)

# STOCK async driver only (no fully-async override) — the harness-pool fan-out shape requires it.
MODE_ARGS=()

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
   # Episode-admission concurrency S = CONC * num_engines (num_engines = ROLLOUT_GPUS/2 = 2). Only W
   # of the admitted episodes are driven by a harness worker at once; the rest queue for a free worker.
   --sglang-server-concurrency "$CONC"
   # Adapter tool parser (A3/B3 dead-reward fix): let SGLang emit Qwen3 tool_use/reasoning blocks the
   # AnthropicAdapter parses. WITHOUT these the mini-harness tool loop is dead and reward is 0.
   --sglang-tool-call-parser qwen25
   --sglang-reasoning-parser qwen3
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

# HARNESS_WORKERS / EPISODE_MAX_* are read by HarnessPool INSIDE the rollout process, so they MUST be
# forwarded through the ray runtime env (not just the launching shell).
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM:${DEMO_DIR}:/root/slime\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"SANDBOX\": \"${SANDBOX}\",
    \"AGENTCORE_SANDBOX_RUNTIME_ARN\": \"${AGENTCORE_SANDBOX_RUNTIME_ARN:-}\",
    \"AGENTCORE_SANDBOX_RUNTIME_ARNS\": \"${AGENTCORE_SANDBOX_RUNTIME_ARNS:-}\",
    \"AWS_REGION\": \"${AWS_REGION:-us-west-2}\",
    \"AWS_BEARER_TOKEN_BEDROCK\": \"${AWS_BEARER_TOKEN_BEDROCK:-}\",
    \"HARNESS_WORKERS\": \"${HARNESS_WORKERS}\",
    \"EPISODE_MAX_TURNS\": \"${EPISODE_MAX_TURNS}\",
    \"EPISODE_MAX_TOKENS\": \"${EPISODE_MAX_TOKENS}\",
    \"INFER_CONC\": \"${INFER_CONC:-0}\"
  }
}"

cd "$SLIME"
# ASYNC driver + DECOUPLED placement: train_async.py, NO --colocate. Absolute path: the ray job
# entrypoint's cwd is the ray-head cwd (/work here), not this script's `cd "$SLIME"`.
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
