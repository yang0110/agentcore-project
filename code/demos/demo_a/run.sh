#!/bin/bash
# Demo A — MBPP, CPU sandbox, OUR multi-turn run_episode harness. REAL slime training, ASYNC
# (decoupled) mode. This dir owns its data/ + ckpt/, but imports the SHARED agentcore_slime_demo
# library from /work/code via PYTHONPATH (no vendored copy). Runs INSIDE the persistent
# container agentcore_p4 (slimerl/slime).
#
# Async/decoupled: agentic rollouts have long, variable latency (multi-turn code repair in a
# sandbox). slime's async driver (train_async.py) runs trainer GPUs and SGLang inference GPUs on
# DISJOINT devices and generates rollout N+1 WHILE training on rollout N. Hard-requires no
# --colocate (train_async.py asserts it).
#
# GPU split (8xL40S): 4 train + 4 inference + 0 env. MBPP checker is plain Python/IO => CPU envs,
# so all 8 GPUs serve the policy (train + infer).
#
# Sandbox backend (SANDBOX): agentcore -> real AWS Code Interpreter microVM (local mode removed;
# every demo runs against AWS). Tunables: RB (rollout-batch-size), NS (n-samples-per-prompt), NR
# (num-rollout), TEMP (sampling temp), CONC (sglang-server-concurrency), EVAL_INTERVAL/EVAL_NS.
#
# NOTE step count: slime resumes at loaded_iter+1; a fresh torch_dist ckpt is iter 0 => the loop
# runs range(1, NR) = (NR-1) real training steps. NR=100 => 99 steps.
# NOTE GRPO signal: group-relative advantage is 0 when every sample in a prompt's group gets the
# same reward. Reward variance within groups (harder tasks / higher temp) yields nonzero gradient.
set -euo pipefail

SANDBOX="${SANDBOX:-agentcore}"   # AWS Code Interpreter microVM (local mode removed)
RB="${RB:-4}"          # prompts per rollout step
NS="${NS:-8}"          # GRPO group size (samples per prompt)
NR="${NR:-100}"        # num-rollout; real training steps = NR-1 (see NOTE) => 99 steps.
                       # Keep the train set >= NR*RB prompts so one epoch spans the whole run and
                       # the ONLY checkpoint is the final one (slime force-saves at epoch
                       # boundaries too). The MBPP pool is built at 500 prompts (>= 100*4).
TEMP="${TEMP:-1.0}"    # rollout sampling temperature (higher => more intra-group diversity)
GBS=$(( RB * NS ))     # global batch = all samples, 1 optimizer step per rollout
EVAL_INTERVAL="${EVAL_INTERVAL:-10}"  # run held-out eval every N training steps (0 => off)
EVAL_NS="${EVAL_NS:-2}"               # samples per eval prompt
# Checkpoint policy (disk-saving): we do NOT snapshot mid-training. slime ALWAYS force-saves at
# the final rollout AND at every epoch boundary (step % (dataset_size/RB) == 0) regardless of
# save-interval. So a huge save-interval + a train set with dataset_size >= NR*RB (one epoch spans
# the whole run) => the ONLY checkpoint written is the one at the end of training.
SAVE_INTERVAL="${SAVE_INTERVAL:-1000000}"

SLIME=/root/slime
DEMO_DIR=/work/code/demos/demo_a
CKPT="${CKPT_BASE:-/work/code/p4_slime_train/ckpt/Qwen3-4B_torch_dist}"  # SHARED base Megatron ckpt
HF=/root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c

# ---- Demo A config (baked in) --------------------------------------------------------------
ACTOR_GPUS="${ACTOR_GPUS:-4}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-4}"
DATA="${DATA:-$DEMO_DIR/data/mbpp_train.jsonl}"
EVAL_DATA="${EVAL_DATA:-$DEMO_DIR/data/mbpp_eval.jsonl}"
EVAL_NAME="${EVAL_NAME:-mbpp}"
SAVE="${SAVE:-$DEMO_DIR/ckpt/Qwen3-4B_mbpp_async}"
GPU_ENV=("\"AGENTCORE_GPU_GATE\": \"off\"")   # CPU envs, no GpuPool

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

# Held-out EVAL every EVAL_INTERVAL steps. slime routes eval through our SAME custom generate
# (falls back to --custom-generate-function-path) with evaluation=True, so an eval "solve rate"
# is one full agentic episode per eval prompt. Eval metrics are logged as `eval/<name>` tagged
# with `eval/step`; metrics.py writes them to a SEPARATE eval CSV.
EVAL_ARGS=()
if [ "$EVAL_INTERVAL" != "0" ] && [ -f "$EVAL_DATA" ]; then
   EVAL_ARGS=(
      --eval-interval "$EVAL_INTERVAL"
      --eval-prompt-data "$EVAL_NAME" "$EVAL_DATA"
      --n-samples-per-eval-prompt "$EVAL_NS"
      --eval-temperature "$TEMP"
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
   # Cap rollout concurrency. slime gates custom-generate with
   # Semaphore(sglang_server_concurrency * num_engines). Our custom generate opens an HTTP
   # session AND a sandbox subprocess per episode, so keep this small.
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

GPU_ENV_JSON=$(IFS=,; echo "${GPU_ENV[*]}")

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM:${DEMO_DIR}:/work/code:/root/slime\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"SANDBOX\": \"${SANDBOX}\",
    \"AWS_REGION\": \"${AWS_REGION:-us-west-2}\",
    \"AWS_BEARER_TOKEN_BEDROCK\": \"${AWS_BEARER_TOKEN_BEDROCK:-}\",
    ${GPU_ENV_JSON}
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
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"
