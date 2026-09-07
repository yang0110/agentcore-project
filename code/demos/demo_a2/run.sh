#!/bin/bash
# Demo A2 — MBPP (CPU sandbox) with the REAL CLAUDE CODE agent harness, ASYNC/decoupled slime.
# Runner: this dir owns its data/ + ckpt/; agentcore_slime_demo is the SHARED lib from /work/code (PYTHONPATH).
# Runs INSIDE the persistent container (agentcore_p4). In-container path: /work/code/demos/demo_a2.
#
# Demo A2 == Demo A + Claude Code:
#   Same task (MBPP), same sandbox family (CPU microVM / Code Interpreter), same slime training
#   path and same Qwen3-4B policy. The ONE difference vs Demo A is the ROLLOUT HARNESS: instead
#   of our lightweight run_episode loop (agentcore_slime_demo.generate.generate), the agent loop
#   is Claude Code's OWN CLI running inside the sandbox
#   (agentcore_slime_demo.claude_code_generate.generate). Claude Code's "Anthropic" traffic is
#   pointed at slime's AnthropicAdapter (ANTHROPIC_BASE_URL -> adapter -> SGLang policy), so every
#   completion is served by the policy under training and its tokens are captured for GRPO.
#   Agent Lightning's "train with real harnesses, zero changes": the ONLY edit to slime's shipped
#   coding_agent_rl generate is swapping E2BSandbox -> AgentCoreSandbox.
#
# THIS PATH DOES NOT RUN OFFLINE. It requires (see README.md "Pending pieces" + demo-plan.md):
#   * a CUSTOM AgentCore Runtime container image (multi-user: useradd/setsid/npm) — see Dockerfile
#   * Node 22 + Claude Code tarballs on the host (SLIME_AGENT_NODE_TARBALL / SLIME_AGENT_CC_TARBALL)
#   * ADAPTER_PUBLIC_HOST — an address the sandbox can dial back to reach the adapter
#
# GPU split (same as Demo A): 4 train + 4 inference + 0 env (CPU envs => no GpuPool).
# Tunables: RB (rollout-batch-size), NS (n-samples-per-prompt), NR (num-rollout),
#           TEMP (sampling temp), CONC (sglang-server-concurrency).
#
# NOTE step count: slime resumes at loaded_iter+1; a fresh torch_dist ckpt is iter 0 => the loop
# runs range(1, NR) = (NR-1) real training steps. NR=2 => 1 step.
set -euo pipefail

SANDBOX="${SANDBOX:-agentcore}"  # Claude Code needs the real multi-user microVM; 'local' won't host the CLI
RB="${RB:-4}"          # prompts per rollout step
NS="${NS:-8}"          # GRPO group size (samples per prompt)
NR="${NR:-100}"        # num-rollout; real training steps = NR-1 => 99 steps.
                       # Keep the train set >= NR*RB prompts so one epoch spans the whole run and
                       # the ONLY checkpoint is the final one (slime force-saves at epoch
                       # boundaries too). The MBPP train pool here is built at 500 prompts
                       # (>= 100*4), so NR up to 125 stays save-only-at-end. Override per run.
TEMP="${TEMP:-1.0}"    # rollout sampling temperature (higher => more intra-group diversity)
GBS=$(( RB * NS ))     # global batch = all samples, 1 optimizer step per rollout
EVAL_INTERVAL="${EVAL_INTERVAL:-10}"  # run held-out eval every N training steps (0 => off)
EVAL_NS="${EVAL_NS:-2}"               # samples per eval prompt
# Checkpoint policy (disk-saving): no mid-training snapshots. slime ALWAYS force-saves at the final
# rollout AND at every EPOCH boundary (step % (dataset_size/RB) == 0). To get EXACTLY ONE
# checkpoint (at the end), the train set must span one epoch: dataset_size >= NR * RB. The MBPP
# train pool is built at 500 prompts (=> 125 steps/epoch >= NR=100), so this holds.
SAVE_INTERVAL="${SAVE_INTERVAL:-1000000}"

SLIME=/root/slime
DEMO_DIR=/work/code/demos/demo_a2
CKPT_BASE="${CKPT_BASE:-/work/code/p4_slime_train/ckpt/Qwen3-4B_torch_dist}"  # shared base ckpt (referenced, not copied)
HF=/root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c

# Demo A2: 4 train + 4 inference + 0 env. CPU envs => no GpuPool.
ACTOR_GPUS="${ACTOR_GPUS:-4}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-4}"
DATA="${DATA:-$DEMO_DIR/data/mbpp_train.jsonl}"
EVAL_DATA="${EVAL_DATA:-$DEMO_DIR/data/mbpp_eval.jsonl}"
EVAL_NAME="${EVAL_NAME:-mbpp}"
SAVE="${SAVE:-$DEMO_DIR/ckpt/Qwen3-4B_mbpp_claudecode_async}"

# ---- Claude Code harness env (MUST be set for a real run; empty defaults are placeholders) ----
# The rollout is Claude Code running inside the sandbox, talking back to slime's adapter.
ADAPTER_PUBLIC_HOST="${ADAPTER_PUBLIC_HOST:-}"       # address the sandbox dials back (REQUIRED)
SLIME_AGENT_NODE_TARBALL="${SLIME_AGENT_NODE_TARBALL:-}"  # Node 22 tarball for install_cli (REQUIRED)
SLIME_AGENT_CC_TARBALL="${SLIME_AGENT_CC_TARBALL:-}"      # Claude Code tarball for install_cli (REQUIRED)
ANTHROPIC_MODEL="${ANTHROPIC_MODEL:-}"               # model name Claude Code requests (served by policy)

# clean any prior ray/sglang from a previous run
pkill -9 sglang 2>/dev/null || true; ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true; sleep 2

set -ex
export PYTHONUNBUFFERED=1
source "$SLIME/scripts/models/qwen3-4B.sh"

CKPT_ARGS=(
   --hf-checkpoint "$HF"
   --ref-load "$CKPT_BASE"
   --load "$CKPT_BASE"
   --save "$SAVE"
   --save-interval "$SAVE_INTERVAL"   # save-only-at-end (see SAVE_INTERVAL note above)
   # ---- Checkpoint-save deadlock fix (two flags that MUST go together) ---------------------
   # The end-of-training save intermittently DEADLOCKED: Megatron's default checkpoint writer
   # forks a writer process off the MAIN training process, and forking a process holding
   # CUDA/NCCL/inductor threads can inherit a locked mutex -> the child hangs forever AFTER the
   # .distcp files are written. The safe backend is Megatron's SPAWN-based PersistentAsyncCaller,
   # selected only when BOTH --async-save is set AND init_persistent_async_worker() is called.
   # slime forces use_persistent_ckpt_worker=True but its own init() never calls that function,
   # so --async-save alone falls back to the fork path. So we do BOTH:
   #   1. --async-save
   #   2. --custom-megatron-init-path ...enable_persistent_async_ckpt  (calls the missing init)
   # See agentcore_slime_demo/megatron_init.py and docs/demos-guide.md §7.
   --async-save
   --custom-megatron-init-path agentcore_slime_demo.megatron_init.enable_persistent_async_ckpt
)

# Held-out EVAL every EVAL_INTERVAL steps, routed through the SAME Claude Code generate with
# evaluation=True (one full agentic episode per eval prompt, no gradient). eval metrics logged as
# eval/<name>; metrics.py writes them to a separate eval CSV.
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
   # THE Demo-A2 difference: the real Claude Code harness, not our lightweight run_episode.
   --custom-generate-function-path agentcore_slime_demo.claude_code_generate.generate
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
   # Cap rollout concurrency. Each episode boots a sandbox + a Claude Code CLI session, so keep small.
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
    \"PYTHONPATH\": \"/root/Megatron-LM:${DEMO_DIR}:/work/code:/root/slime\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"SANDBOX\": \"${SANDBOX}\",
    \"AWS_REGION\": \"${AWS_REGION:-us-west-2}\",
    \"AWS_BEARER_TOKEN_BEDROCK\": \"${AWS_BEARER_TOKEN_BEDROCK:-}\",
    \"AGENTCORE_GPU_GATE\": \"off\",
    \"ADAPTER_PUBLIC_HOST\": \"${ADAPTER_PUBLIC_HOST}\",
    \"SLIME_AGENT_NODE_TARBALL\": \"${SLIME_AGENT_NODE_TARBALL}\",
    \"SLIME_AGENT_CC_TARBALL\": \"${SLIME_AGENT_CC_TARBALL}\",
    \"ANTHROPIC_MODEL\": \"${ANTHROPIC_MODEL}\"
  }
}"

cd "$SLIME"
# ASYNC driver + DECOUPLED placement: train_async.py, NO --colocate. Absolute path required (the
# ray job entrypoint's cwd is the ray-head cwd (/work), not this script's `cd "$SLIME"`).
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
