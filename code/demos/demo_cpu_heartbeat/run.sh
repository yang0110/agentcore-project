#!/bin/bash
# demo_cpu_heartbeat — REAL slime CPU trainer (async/decoupled, Mode B) that holds its REMOTE AgentCore
# sandbox sessions WARM across the live weight-sync barrier. Same trainer shape as demo_cpu_mbpp
# (train_async, NR=11 => 10 train + 1 eval, save-at-end) — the ONLY addition is the keep-warm lever:
#
#   During slime's trainer->inference weight sync, generation stalls, so every open sandbox session
#   sits idle. AgentCore reclaims an idle session purely on ABSENCE of invokes (15-min microVM /
#   ~600s GPU-instance idle timeout). A long enough sync barrier (or a slow rollout tail) can let an
#   in-flight session be idle-reclaimed mid-episode, losing its workspace. This demo wraps every
#   episode's sandbox in a KeptSandbox tied to a process-wide SessionKeeper (agentcore_slime_demo/
#   session_keeper.py) that fires a cheap no-op heartbeat at HALF the idle timeout ONLY for a session
#   that has actually gone idle — i.e. during the sync stall — so the idle clock never trips. Real
#   episode traffic refreshes the same clock, so outside a stall the keeper fires ZERO heartbeats.
#
# The heartbeat lives OUTSIDE the harness (that is the whole point): the run_episode loop and the
# sandbox are oblivious; only KeptSandbox knows both exist. heartbeat_demo.py is the seconds-long,
# offline animation of the same mechanism (naive -> heartbeat -> recover) — see README.md.
#
# Everything else matches demo_cpu_mbpp verbatim (remote-only, Mode A/B async, GRPO). Runs INSIDE
# the persistent container agentcore_p4 (slimerl/slime).
#
# NOTE step count: slime resumes at loaded_iter+1; a fresh torch_dist ckpt is iter 0 => the loop
# runs range(1, NR) = (NR-1) real training steps. NR=11 => 10 train + 1 eval (SMOKE; no more needed).
set -euo pipefail

SANDBOX="${SANDBOX:-agentcore-runtime}"   # agentcore-runtime -> SINGLE remote AgentCore Runtime sandbox (default);
                                          # agentcore -> AWS Code Interpreter microVM. CPU demo => REMOTE-ONLY (no local).
if [ "$SANDBOX" = "local" ]; then
   echo "SANDBOX=local is not supported for CPU demos (remote-only) — use agentcore-runtime (default) or agentcore." >&2
   exit 1
fi
# The remote sandbox Runtime ARN(s). PINNED to the K=2 CPU pool default (same as demo_cpu_mbpp).
AGENTCORE_SANDBOX_RUNTIME_ARNS="${AGENTCORE_SANDBOX_RUNTIME_ARNS:-arn:aws:bedrock-agentcore:us-west-2:<ACCOUNT_ID>:runtime/acs_sandbox_cpu-1gzBVp675S,arn:aws:bedrock-agentcore:us-west-2:<ACCOUNT_ID>:runtime/acs_sandbox_cpu_1-adRZyEHM6g}"
if [ "$SANDBOX" = "agentcore-runtime" ] && [ -z "$AGENTCORE_SANDBOX_RUNTIME_ARNS" ]; then
   AGENTCORE_SANDBOX_RUNTIME_ARN="${AGENTCORE_SANDBOX_RUNTIME_ARN:?set AGENTCORE_SANDBOX_RUNTIME_ARN (or ..._ARNS) to the CPU sandbox Runtime ARN}"
fi

# ---- keep-warm knobs (demo_cpu_heartbeat's headline; consumed in agentcore_slime_demo/generate.py) --
HEARTBEAT_KEEP_WARM="${HEARTBEAT_KEEP_WARM:-1}"       # 1 => wrap every sandbox in KeptSandbox + SessionKeeper; 0 => plain (== demo_cpu_mbpp)
HEARTBEAT_IDLE="${HEARTBEAT_IDLE:-600}"               # session idle timeout to defend (s); heartbeat fires at ping_fraction*this
HEARTBEAT_PING_FRACTION="${HEARTBEAT_PING_FRACTION:-0.5}"  # heartbeat once idle >= fraction*idle_timeout (0.5 => >=2 chances before expiry)

RB="${RB:-4}"          # prompts per rollout step
NS="${NS:-8}"          # GRPO group size (samples per prompt)
NR="${NR:-11}"        # num-rollout; real training steps = NR-1 => 10 train + 1 eval (SMOKE).
TEMP="${TEMP:-1.0}"    # rollout sampling temperature
GBS=$(( RB * NS ))     # global batch = all samples, 1 optimizer step per rollout
MODE="${MODE:-B}"
case "$MODE" in B|b) MODE=B;; A|a) MODE=A;; *) echo "MODE must be A or B (got '$MODE')" >&2; exit 1;; esac
EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
EVAL_NS="${EVAL_NS:-2}"               # samples per eval prompt
SAVE_INTERVAL="${SAVE_INTERVAL:-1000000}"   # save-only-at-end (see demo_cpu_mbpp note)

SLIME=/root/slime
DEMO_DIR=/work/code/demos/demo_cpu_heartbeat
CKPT="${CKPT_BASE:-/work/code/p4_slime_train/ckpt/Qwen3-4B_torch_dist}"  # SHARED base Megatron ckpt
HF=/root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c

ACTOR_GPUS="${ACTOR_GPUS:-4}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-4}"

# Task family. Defaults to deepscaler (numeric-answer math, RLVR numeric match) — a pure-CPU checker,
# so the sandbox is CPU-backed and the keep-warm heartbeat is the only thing this demo adds.
FAMILY="${FAMILY:-deepscaler}"
if [ "$FAMILY" = "coding" ] || [ "$FAMILY" = "mbpp" ]; then
  _DS=mbpp; _EVAL_NAME=mbpp
else
  _DS="$FAMILY"; _EVAL_NAME="$FAMILY"
fi
DATA="${DATA:-$DEMO_DIR/data/${_DS}_train.jsonl}"
EVAL_DATA="${EVAL_DATA:-$DEMO_DIR/data/${_DS}_eval.jsonl}"
EVAL_NAME="${EVAL_NAME:-$_EVAL_NAME}"
SAVE="${SAVE:-$DEMO_DIR/ckpt/Qwen3-4B_${_DS}_heartbeat_async}"

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
   --save-interval "$SAVE_INTERVAL"
   --async-save
   --custom-megatron-init-path agentcore_slime_demo.megatron_init.enable_persistent_async_ckpt
)

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
   --custom-generate-function-path agentcore_slime_demo.generate.generate
)

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
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus 8 \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM:${DEMO_DIR}:/root/slime\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"SANDBOX\": \"${SANDBOX}\",
    \"AGENTCORE_SANDBOX_RUNTIME_ARN\": \"${AGENTCORE_SANDBOX_RUNTIME_ARN:-}\",
    \"AGENTCORE_SANDBOX_RUNTIME_ARNS\": \"${AGENTCORE_SANDBOX_RUNTIME_ARNS:-}\",
    \"AWS_REGION\": \"${AWS_REGION:-us-west-2}\",
    \"AWS_BEARER_TOKEN_BEDROCK\": \"${AWS_BEARER_TOKEN_BEDROCK:-}\",
    \"INFER_CONC\": \"${INFER_CONC:-0}\",
    \"HEARTBEAT_KEEP_WARM\": \"${HEARTBEAT_KEEP_WARM}\",
    \"HEARTBEAT_IDLE\": \"${HEARTBEAT_IDLE}\",
    \"HEARTBEAT_PING_FRACTION\": \"${HEARTBEAT_PING_FRACTION}\"
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
   "${MODE_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"
