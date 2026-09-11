#!/bin/bash
# Demo B4 — KernelBench, REMOTE g6 sandbox, with SEVERAL episodes SHARING that g6's ONE GPU via Ray.
# Runner (this dir owns data/ + ckpt/; agentcore_slime_demo is the SHARED lib from /work/code). Runs INSIDE the
# persistent slime container (agentcore_p4), async/decoupled (train_async.py, NO --colocate).
#
# Demo B4 == Demo B3's task + mini harness, but a DIFFERENT sandbox GPU story:
#   * ALL of series B uses a REMOTE g6 AgentCore Runtime #2 as the sandbox, and this box gives ALL 8
#     of its GPUs to slime (4 train + 4 infer). B4 does NOT reserve any local GPU.
#   * B3: AgentCore maps 1 session -> 1 g6 instance, so N concurrent rollouts want N g6 instances.
#         On a capacity-starved g6 fleet that is exactly what stalled B3.
#   * B4: pack SEVERAL concurrent KernelBench sandboxes onto ONE g6 instance and let RAY SPLIT THAT
#         g6's single GPU between them. Because AgentCore maps 1 runtimeSessionId -> 1 instance, every
#         B4 episode invokes the g6 under ONE shared runtimeSessionId (so they reuse the same g6) and
#         carries a per-episode sub_session; a Ray farm running INSIDE the g6 (sandbox_ray_farm.
#         RayGpuFarm, driven by sandbox_server_agent) leases one FRACTIONAL-GPU actor per sub_session.
#         Ray's fraction is scheduling-only (no memory guard; L4/L40S have NO MIG), so each sandbox
#         ALSO caps its GPU memory in software (an injected sitecustomize.py). See ray_farm_generate.py.
#
# Self-contained HARNESS on ONE box: the mini harness runs IN-PROCESS (dials the adapter over
# localhost), so B4 needs NO harness Runtime #1 and NO ADAPTER_PUBLIC_HOST / VPC dial-back. The only
# remote piece is the shared-GPU sandbox Runtime #2 (its g6 image bakes `ray` + the B4_* farm config).
#
# DEPLOY the shared-GPU sandbox: build the g6 sandbox image with `ray` installed and the farm config
# baked as ENV (B4_GPU_FRACTION, B4_MAX_SANDBOXES, B4_PER_SANDBOX_MEM_FRACTION, B4_FARM_GPUS, plus
# SANDBOX_PYLIB/LD_LIBRARY_PATH so the fractional actors' torch resolves), then export its ARN as
# AGENTCORE_SANDBOX_RUNTIME_ARN. Those B4_* vars live on the G6, NOT here (the client only picks the
# shared runtimeSessionId + per-episode sub_session).
#
# NOTE step count: slime resumes at loaded_iter+1; a fresh torch_dist ckpt is iter 0 => the loop
# runs range(1, NR) = (NR-1) real training steps. NR=11 => 10 train (SMOKE; no more needed).
set -euo pipefail

SANDBOX="${SANDBOX:-agentcore-runtime}"   # sandbox = its OWN remote g6 AgentCore Runtime #2 (GPU shared via ray)
# ---- Task set: the REAL KernelBench (ScalingIntelligence/KernelBench) ----------------------------
# Train = level_1 (100 single-op CUDA problems); held-out eval = 20 level_2 fusion problems (disjoint).
# Build with build_dataset.py --dataset kernelbench --level level_1 --limit 100 (train) and
# --level level_2 --limit 20 (eval). Save-only-at-end holds while dataset_size (100) >= NR * RB.
RB="${RB:-4}"          # prompts per rollout step
NS="${NS:-8}"          # GRPO group size (samples per prompt)
NR="${NR:-11}"         # num-rollout; real training steps = NR-1 => 10 train (SMOKE; no more needed). Keep NR*RB <= 100.
TEMP="${TEMP:-1.0}"    # rollout sampling temperature
GBS=$(( RB * NS ))     # global batch = all samples, 1 optimizer step per rollout
EVAL_INTERVAL="${EVAL_INTERVAL:-10}"   # run held-out eval every N training steps (0 => off)
EVAL_NS="${EVAL_NS:-2}"               # samples per eval prompt
SAVE_INTERVAL="${SAVE_INTERVAL:-1000000}"  # save-only-at-end (final rollout force-saves anyway)

SLIME=/root/slime
DEMO_DIR=/work/code/demos/demo_b4
CKPT="${CKPT_BASE:-/work/code/p4_slime_train/ckpt/Qwen3-4B_torch_dist}"   # shared base ckpt (referenced, not copied)
HF=/root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c

DATA="${DATA:-$DEMO_DIR/data/kernelbench_train.jsonl}"
EVAL_DATA="${EVAL_DATA:-$DEMO_DIR/data/kernelbench_eval.jsonl}"
EVAL_NAME="${EVAL_NAME:-kernelbench}"
SAVE="${SAVE:-$DEMO_DIR/ckpt/Qwen3-4B_kernelbench_rayfarm_async}"

# ---- GPU split: ALL 8 to slime (4 train + 4 infer). The env-checker GPU is REMOTE (the shared g6),
# so no local GpuPool and no reserved env GPUs here — identical to B2/B3.
ACTOR_GPUS="${ACTOR_GPUS:-4}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-4}"

# ---- Remote GPU-shared sandbox env (REQUIRED for a real run) ----
# The mini harness is IN-PROCESS; the ONLY remote piece is the shared-GPU sandbox Runtime #2 (g6).
AGENTCORE_SANDBOX_RUNTIME_ARN="${AGENTCORE_SANDBOX_RUNTIME_ARN:?set to the B4 GPU-shared sandbox Runtime #2 ARN (g6 with ray farm baked in)}"
# Optional: pin the shared runtimeSessionId (else ray_farm_generate mints one per process). All B4
# episodes invoke the g6 under this ONE id, so they reuse the same g6 instance whose GPU is shared.
B4_SHARED_RUNTIME_SESSION="${B4_SHARED_RUNTIME_SESSION:-}"
# In-process harness dials the adapter over localhost; no public port / SG rule needed. Ephemeral
# port (0) is fine since nothing external connects.
ADAPTER_BIND_HOST="${ADAPTER_BIND_HOST:-127.0.0.1}"
ADAPTER_PORT="${ADAPTER_PORT:-0}"
EPISODE_MAX_TURNS="${EPISODE_MAX_TURNS:-3}"          # per-episode agentic-turn cap (generate() reads it)
MINI_AGENT_MAX_TURNS="${MINI_AGENT_MAX_TURNS:-3}"    # env fallback for the harness
# Runtime-invoke transient-retry budget (agentcore_runtime_sandbox._retry_invoke). B4 invokes the g6
# once per tool op (open + writes + exec + close), so a transient 502/blip on ANY of them would kill
# the whole job at the default 8 attempts (~150s) — a2 died this way. 20 attempts ≈ 10 min patience.
INVOKE_MAX_RETRIES="${INVOKE_MAX_RETRIES:-20}"

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
   # PP=1 here (mini-harness trajectories are short, like B3), so the SPAWN-based persistent async
   # ckpt worker is safe (see demo_a2/run.sh for why PP=2 breaks it via pidfd_getfd EPERM). Both
   # flags go together.
   --async-save
   --custom-megatron-init-path agentcore_slime_demo.megatron_init.enable_persistent_async_ckpt
)

EVAL_ARGS=()
if [ "$EVAL_INTERVAL" != "0" ] && [ -f "$EVAL_DATA" ]; then
   EVAL_ARGS=(
      --eval-interval "$EVAL_INTERVAL"
      --eval-prompt-data "$EVAL_NAME" "$EVAL_DATA"
      --n-samples-per-eval-prompt "$EVAL_NS"
      --eval-temperature "${EVAL_TEMP:-0.2}"
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
   # THE Demo-B4 difference: the mini harness runs in-process and its sandbox is a sub_session slot
   # on the SHARED g6 (many sandboxes splitting that g6's ONE GPU via ray), not a dedicated g6.
   --custom-generate-function-path agentcore_slime_demo.ray_farm_generate.generate
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
   # Keep rollout concurrency <= the g6 farm's sandbox count (B4_MAX_SANDBOXES, baked on the g6) so
   # episodes don't block on the farm's checkout gate. Default 4; raise together with the farm.
   --sglang-server-concurrency "${CONC:-4}"
   # REQUIRED for the mini-harness tool loop: Qwen3 emits Hermes-style <tool_call>{json}</tool_call>,
   # which the AnthropicAdapter can only turn into tool_use blocks via SGLang's qwen25 parser. Left
   # unset, tool calls are never parsed, the harness stops after 1 turn, and reward is 0 for every
   # rollout (the all-zeros failure demo_a3 hit on 2026-09-08).
   --sglang-tool-call-parser qwen25
   # Strip Qwen3 <think> blocks into reasoning_content so the tool loop stays clean.
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
# Give ray ALL 8 physical GPUs. slime reserves (ACTOR_GPUS + ROLLOUT_GPUS) = 8; the KernelBench
# checker's GPU is on the remote shared g6 sandbox Runtime #2, so no local GPUs are held back.
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus 8 \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM:${DEMO_DIR}:/work/code:/root/slime\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\",
    \"SANDBOX\": \"${SANDBOX}\",
    \"AWS_REGION\": \"${AWS_REGION:-us-west-2}\",
    \"AGENTCORE_REGION\": \"${AGENTCORE_REGION:-${AWS_REGION:-us-west-2}}\",
    \"AWS_BEARER_TOKEN_BEDROCK\": \"${AWS_BEARER_TOKEN_BEDROCK:-}\",
    \"AGENTCORE_SANDBOX_RUNTIME_ARN\": \"${AGENTCORE_SANDBOX_RUNTIME_ARN}\",
    \"B4_SHARED_RUNTIME_SESSION\": \"${B4_SHARED_RUNTIME_SESSION}\",
    \"ADAPTER_BIND_HOST\": \"${ADAPTER_BIND_HOST}\",
    \"ADAPTER_PORT\": \"${ADAPTER_PORT}\",
    \"INVOKE_MAX_RETRIES\": \"${INVOKE_MAX_RETRIES}\",
    \"EPISODE_MAX_TURNS\": \"${EPISODE_MAX_TURNS}\",
    \"MINI_AGENT_MAX_TURNS\": \"${MINI_AGENT_MAX_TURNS}\"
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
