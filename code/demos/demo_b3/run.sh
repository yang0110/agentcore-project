#!/bin/bash
# Demo B3 — KernelBench, GPU sandbox, with a MINI CLAUDE-CODE-LIKE harness.
# Runner (this dir owns data/ + ckpt/; agentcore_slime_demo is the SHARED lib from /work/code). Runs INSIDE the
# persistent slime container (agentcore_p4), async/decoupled (train_async.py, NO --colocate).
#
# Demo B3 == Demo B + a "mimic Claude Code" harness we wrote ourselves. Same task/sandbox story as
# Demo B (KernelBench CUDA ops, checker runs torch on GPU inside the sandbox, GpuPool dense-packs
# env checkers onto physical GPUs 6,7), but the multi-turn loop is
# agentcore_slime_demo.mini_agent.MiniClaudeHarness — a ~1-file pure-Python coding agent
# (write_file / bash / read_file) reproducing Claude Code's model->tool->sandbox->result loop.
# The Qwen3-4B policy is still trained by slime; the agent's Anthropic Messages calls go through
# slime's AnthropicAdapter (-> SGLang). See agentcore_slime_demo/mini_agent_generate.py.
#
# Difference vs Demo B2 (the REAL Claude Code CLI): the mini agent runs in THIS trainer process,
# NOT inside the sandbox, so the sandbox needs NO Node/CLI install and NO custom multi-user
# container (no Dockerfile here) — only CUDA+torch for the KernelBench checker, which the standard
# GPU Runtime image already has. The agent reaches the adapter over localhost, so no dial-back host.
#
# NOTE step count: slime resumes at loaded_iter+1; a fresh torch_dist ckpt is iter 0 => the loop
# runs range(1, NR) = (NR-1) real training steps. NR=20 => 19 steps.
set -euo pipefail

SANDBOX="${SANDBOX:-agentcore}"   # vanilla AWS GPU Runtime sandbox (local mode removed)
# ---- Task set: the REAL KernelBench (ScalingIntelligence/KernelBench) ----------------------------
# Train = level_1 (100 single-op CUDA problems); held-out eval = 20 level_2 fusion problems (disjoint).
# Build with build_dataset.py --dataset kernelbench --level level_1 --limit 100 (train) and
# --level level_2 --limit 20 (eval). This replaced the old 3-op smoke stub, so real RB/NR now work.
# Save-only-at-end holds while dataset_size (100) >= NR * RB: slime force-saves at every epoch boundary
# (step % (dataset_size/RB) == 0), so with RB=4 the epoch is 25 steps and NR=20 (19 real) never crosses it.
RB="${RB:-4}"          # prompts per rollout step
NS="${NS:-8}"          # GRPO group size (samples per prompt)
NR="${NR:-20}"         # num-rollout; real training steps = NR-1 => 19 steps. Keep NR*RB <= 100.
TEMP="${TEMP:-1.0}"    # rollout sampling temperature
GBS=$(( RB * NS ))     # global batch = all samples, 1 optimizer step per rollout
EVAL_INTERVAL="${EVAL_INTERVAL:-5}"   # run held-out eval every N training steps (0 => off)
EVAL_NS="${EVAL_NS:-2}"               # samples per eval prompt
SAVE_INTERVAL="${SAVE_INTERVAL:-1000000}"  # save-only-at-end (final rollout force-saves anyway)

SLIME=/root/slime
DEMO_DIR=/work/code/demos/demo_b3
CKPT="${CKPT_BASE:-/work/code/p4_slime_train/ckpt/Qwen3-4B_torch_dist}"   # shared base ckpt (referenced, not copied)
HF=/root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c

DATA="${DATA:-$DEMO_DIR/data/kernelbench_train.jsonl}"
EVAL_DATA="${EVAL_DATA:-$DEMO_DIR/data/kernelbench_eval.jsonl}"
EVAL_NAME="${EVAL_NAME:-kernelbench}"
SAVE="${SAVE:-$DEMO_DIR/ckpt/Qwen3-4B_kernelbench_mini_async}"

# GPU split: 4 train + 2 inference + 2 env. slime reserves 6 (actor 4 + rollout 2); GpuPool owns 6,7.
ACTOR_GPUS="${ACTOR_GPUS:-4}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-2}"
# Route the KernelBench checker exec (imports torch, runs on cuda) through a process-wide GpuPool
# that leases ONLY physical GPUs 6,7 to the bursty env checkers.
GPU_ENV=(
   "\"AGENTCORE_GPU_GATE\": \"all\""
   "\"GPU_POOL_DEVICES\": \"${GPU_POOL_DEVICES:-6,7}\""
   "\"GPU_POOL_CONCURRENCY\": \"${GPU_POOL_CONCURRENCY:-16}\""
   "\"GPU_POOL_PER_ENV_MIB\": \"${GPU_POOL_PER_ENV_MIB:-2048}\""
)

# ---- Mini-harness env ----
# The mini agent runs IN-PROCESS and reaches the adapter over localhost, so ADAPTER_PUBLIC_HOST is
# optional (defaults to 127.0.0.1 in mini_agent_generate). No Node/Claude-Code tarballs and no
# custom container — that is the simplification vs B2. See agentcore_slime_demo/mini_agent.py.
ADAPTER_PUBLIC_HOST="${ADAPTER_PUBLIC_HOST:-}"       # optional; only if binding the adapter on a routable NIC
MINI_AGENT_MAX_TURNS="${MINI_AGENT_MAX_TURNS:-12}"   # max model<->tool turns per episode

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
   # ---- Checkpoint-save deadlock fix (two flags that MUST go together) ---------------------
   # Megatron's default async writer forks off the MAIN training process; forking a process that
   # holds CUDA/NCCL/inductor threads can inherit a locked mutex and hang forever AFTER the .distcp
   # files are written (intermittent). The safe backend is the SPAWN-based PersistentAsyncCaller,
   # which Megatron only selects when BOTH --async-save is set AND init_persistent_async_worker() is
   # called. slime forces use_persistent_ckpt_worker=True but never calls that init, so we do it via
   # slime's custom-megatron-init hook (agentcore_slime_demo/megatron_init.py). See demos-guide.md §7.
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
   # THE KEY DIFFERENCE from Demo B: our mini Claude-Code-like harness instead of run_episode (and
   # instead of B2's real CLI). Same slime training-data path; the agent loop is our in-process one.
   --custom-generate-function-path agentcore_slime_demo.mini_agent_generate.generate
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
   # Cap rollout concurrency (Semaphore(sglang_server_concurrency * num_engines)). Each episode
   # spins an in-process agent loop + a sandbox, so keep this small.
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
# Give ray ALL 8 physical GPUs. slime reserves (ACTOR_GPUS + ROLLOUT_GPUS) = 6; the remaining 2
# stay unreserved and the GpuPool leases them (6,7) to the env checkers directly.
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
    \"ADAPTER_PUBLIC_HOST\": \"${ADAPTER_PUBLIC_HOST}\",
    \"MINI_AGENT_MAX_TURNS\": \"${MINI_AGENT_MAX_TURNS}\",
    ${GPU_ENV_JSON}
  }
}"

cd "$SLIME"
# ASYNC driver + DECOUPLED placement. Absolute path to train_async.py (the ray-job cwd is /work).
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
