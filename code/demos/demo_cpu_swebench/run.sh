#!/bin/bash
# demo_cpu_swebench — REAL slime CPU trainer on a SWE-bench Verified subset: one prebuilt SWE-bench image
# per dataset row, every episode ISOLATED on its own fresh AgentCore session.
#
# Each episode routes to ITS instance's OWN Runtime (one image / one ARN per SWE-bench instance; the pool
# map is AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE, written by deploy/deploy_swebench.py to
# swebench/arn_by_image.json — an unmapped instance is an error, never routed elsewhere). The images are
# Epoch AI's arm64 rebuilds of the SWE-bench images (0.7-1 GB compressed for the default subset), under the
# managed microVM fleet's arm64 / 2 GB cap, so every Runtime is a plain microVM Runtime: one session = one
# fresh microVM.
# The episode runs the MiniClaudeHarness agentic loop in /testbed (LLM via the adapter under training,
# tools in the sandbox), then swebench.grade_swebench writes the HIDDEN test_patch in, runs the repo's
# official test command and scores FAIL_TO_PASS + PASS_TO_PASS with SWE-bench's own log parsers — the
# RLVR reward — and the session is stopped. Self-contained: imports this dir's vendored
# agentcore_slime_demo copy. Runs INSIDE the persistent container agentcore_p4 (slimerl/slime).
#
# WHY A FRESH SESSION PER EPISODE: the agent edits the repo and may pip-install into the testbed env;
# grading must see exactly the environment SWE-bench validated, so no session is ever reused. The microVM
# cold start this costs (~25-30 s for a ~1-2 GB image, probe_image_cap.py) is hidden by opening each episode's session at submit time and, via the
# look-ahead data source, for the NEXT TB_PREOPEN_BATCHES rollouts before slime draws them. A
# SessionKeeper heartbeat keeps open sessions alive across the weight-sync stall.
#
# FULLY-ASYNC DRIVER (slime's generate_rollout_fully_async, the only one): terminalbench_pool_generate.generate
# (the shared image-pool hook, vendored from the former demo_cpu_terminalbench) returns one bare Sample per
# episode, and the look-ahead data source slides with the driver's per-group draws. Eval goes through
# agentcore_slime_demo.mode_b_eval (the fully-async worker owns slime's GenerateState loop).
#
# ADAPTER TOOL PARSER (the A3/B3 dead-reward fix): SGLANG_ARGS bakes in --sglang-tool-call-parser
# qwen25 --sglang-reasoning-parser qwen3. Without them the tool loop is dead and reward is structurally 0.
set -euo pipefail
case "${HARNESS_WHERE:-session}" in session) ;; *) echo "HARNESS_WHERE=$HARNESS_WHERE is not supported: the agent loop always runs in the sandbox session" >&2; exit 1 ;; esac

SANDBOX="${SANDBOX:-agentcore-runtime}"   # fresh AgentCore session per episode (the pool opens it).
if [ "$SANDBOX" = "local" ]; then
   echo "SANDBOX=local is not supported for CPU demos (remote-only) — use agentcore-runtime (default)." >&2
   exit 1
fi

DEMO_DIR=/work/code/demos/demo_cpu_swebench
POOL_MAP="${POOL_MAP:-$DEMO_DIR/swebench/arn_by_image.json}"   # {instance -> Runtime ARN}, from deploy
# The image pool map is REQUIRED: every episode routes to its instance's Runtime through it.
if [ ! -f "$POOL_MAP" ]; then
   echo "pool map $POOL_MAP not found — run deploy/deploy_swebench.py first (writes swebench/arn_by_image.json)." >&2
   exit 1
fi
AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE="$(cat "$POOL_MAP")"

RB="${RB:-4}"          # prompts per rollout step
NS="${NS:-8}"          # GRPO group size (samples per prompt)
NR="${NR:-11}"         # num-rollout; real training steps = NR-1 => 10 train + 1 eval (SMOKE)
TEMP="${TEMP:-1.0}"
GBS=$(( RB * NS ))

# ---- isolation + pre-open + heartbeat knobs (read by TerminalBenchPool inside the rollout process) --------
HARNESS_WORKERS="${HARNESS_WORKERS:-$GBS}"    # harness tier: dispatch workers (default RB x NS = one step)
EPISODE_MAX_TURNS="${EPISODE_MAX_TURNS:-12}"  # agentic turns per SWE-bench episode (= SWE_MAX_TURNS in the data rows)
EPISODE_MAX_TOKENS="${EPISODE_MAX_TOKENS:-1024}"
MINI_AGENT_MAX_TOOL_CHARS="${MINI_AGENT_MAX_TOOL_CHARS:-2000}"  # clip each tool result fed back (bounds trajectory length)
TB_EXEC_TIMEOUT="${TB_EXEC_TIMEOUT:-180}"     # per bash-tool command timeout (test runs / pip are routine)
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-24576}"  # >= one full trajectory: 12 x (1024 gen + ~600 tool) + prompt
WARM_IDLE_TIMEOUT="${WARM_IDLE_TIMEOUT:-900}" # session idle timeout the heartbeat stays under
WARM_PING_FRACTION="${WARM_PING_FRACTION:-0.5}"
HEARTBEAT="${HEARTBEAT:-1}"   # keeper pings every open session idle past WARM_PING_FRACTION x WARM_IDLE_TIMEOUT (0 = off, A/B)
TB_PREOPEN_BATCHES="${TB_PREOPEN_BATCHES:-2}"   # THE pre-open knob: batches kept booted ahead (x RB x NS = 64 microVM sessions; 0 = off)

if [ -n "${MODE:-}" ]; then
  echo "MODE is gone: the fully-async driver is the only rollout driver. Unset MODE." >&2
  exit 1
fi

EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
EVAL_NS="${EVAL_NS:-2}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000000}"

SLIME=/root/slime
CKPT="${CKPT_BASE:-/work/code/p4_slime_train/ckpt/Qwen3-4B_torch_dist}"
HF=/root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c

ACTOR_GPUS="${ACTOR_GPUS:-4}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-4}"
NENG=$(( ROLLOUT_GPUS / 2 ))
# Admit W running + W queued episodes (S = CONC*NENG) so a free worker never idles. Pre-opening no longer
# needs a deep queue: the look-ahead data source announces upcoming episodes, not the admission gate.
CONC="${CONC:-$(( (2 * HARNESS_WORKERS + NENG - 1) / NENG ))}"

DATA="${DATA:-$DEMO_DIR/data/swebench_train.jsonl}"
EVAL_DATA="${EVAL_DATA:-$DEMO_DIR/data/swebench_eval.jsonl}"
EVAL_NAME="${EVAL_NAME:-swebench}"
SAVE="${SAVE:-$DEMO_DIR/ckpt/Qwen3-4B_swebench_async}"

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
   --save-interval "$SAVE_INTERVAL"
   --async-save
   --custom-megatron-init-path agentcore_slime_demo.megatron_init.enable_persistent_async_ckpt
)

EVAL_ARGS=()
if [ "$EVAL_INTERVAL" != "0" ] && [ -f "$EVAL_DATA" ]; then
   EVAL_ARGS=(
      --eval-interval "$EVAL_INTERVAL"
      --eval-prompt-data "$EVAL_NAME" "$EVAL_DATA"
      --n-samples-per-eval-prompt "$EVAL_NS"
      --eval-temperature "$TEMP"
      --eval-function-path agentcore_slime_demo.mode_b_eval.generate_rollout
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
   # THE DRIVER: a background worker keeps CONC x engines groups in flight across step boundaries.
   --rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async
   # THE POOL HOOK: routes each episode to its instance's Runtime on its own fresh, pre-opened session.
   --custom-generate-function-path agentcore_slime_demo.terminalbench_pool_generate.generate
   # THE LOOK-AHEAD: peeks the next TB_PREOPEN_BATCHES x RB prompts so their sessions boot early.
   --data-source-path agentcore_slime_demo.tb_lookahead_source.LookaheadDataSource
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu "$MAX_TOKENS_PER_GPU"
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
   --sglang-server-concurrency "$CONC"
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
# Shared box only: SLIME_CUDA_VISIBLE=2,3,4,5,6,7 SLIME_NUM_GPUS=6 pins ray to a subset (default: all 8).
[ -n "${SLIME_CUDA_VISIBLE:-}" ] && export CUDA_VISIBLE_DEVICES="$SLIME_CUDA_VISIBLE"
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${SLIME_NUM_GPUS:-8}" \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# The isolation/pre-open/heartbeat knobs and the image-pool map are read by TerminalBenchPool INSIDE the
# rollout process, so they MUST be forwarded through the ray runtime env (not just the launching shell).
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM:${DEMO_DIR}:/root/slime\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"SANDBOX\": \"${SANDBOX}\",
    \"ADAPTER_PUBLIC_HOST\": \"${ADAPTER_PUBLIC_HOST:-}\",
    \"ADAPTER_PORT_BASE\": \"${ADAPTER_PORT_BASE:-18100}\",
    \"MINI_SESSION_POLL_SEC\": \"${MINI_SESSION_POLL_SEC:-25}\",
    \"AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE\": $(python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))' <<< "${AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE}"),
    \"AWS_REGION\": \"${AWS_REGION:-us-west-2}\",
    \"AWS_BEARER_TOKEN_BEDROCK\": \"${AWS_BEARER_TOKEN_BEDROCK:-}\",
    \"HARNESS_WORKERS\": \"${HARNESS_WORKERS}\",
    \"EPISODE_MAX_TURNS\": \"${EPISODE_MAX_TURNS}\",
    \"EPISODE_MAX_TOKENS\": \"${EPISODE_MAX_TOKENS}\",
    \"WARM_IDLE_TIMEOUT\": \"${WARM_IDLE_TIMEOUT}\",
    \"WARM_PING_FRACTION\": \"${WARM_PING_FRACTION}\",
    \"HEARTBEAT\": \"${HEARTBEAT}\",
    \"TB_PREOPEN_BATCHES\": \"${TB_PREOPEN_BATCHES}\",
    \"MINI_AGENT_MAX_TOOL_CHARS\": \"${MINI_AGENT_MAX_TOOL_CHARS}\",
    \"TB_EXEC_TIMEOUT\": \"${TB_EXEC_TIMEOUT}\"
  }
}"

cd "$SLIME"
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 "$SLIME/train_async.py" \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "$ACTOR_GPUS" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"
