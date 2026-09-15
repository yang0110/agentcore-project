#!/bin/bash
# Demo B (KernelBench) — standalone runner. REAL slime training, ASYNC (decoupled).
# Runs INSIDE the persistent container (agentcore_p4). This dir owns its data/ + ckpt/ and imports
# its own vendored agentcore_slime_demo copy in this dir (PYTHONPATH); it references only genuinely
# SHARED infrastructure (the container, the base Megatron checkpoint, /root/slime). No DEMO branch — Demo B is baked in.
#
# What Demo B is: KernelBench-style CUDA-kernel tasks. The policy writes solution.py, a GPU
# checker runs it (python3 run_checks.py imports torch, compiles/executes on cuda, and compares
# against a torch reference), reads the failure, fixes it — our lightweight run_episode harness
# (agentcore_slime_demo.harness) drives the multi-turn loop IN-PROCESS on this box, but the sandbox
# (write_file / exec / read_file) is a SEPARATE GPU-backed AgentCore Runtime #2 (sandbox_server_agent
# on a g6): the KernelBench torch/CUDA checker runs THERE, reached over invoke_agent_runtime. This is
# the decoupled-sandbox path — harness in-process, sandbox remote (no MCP/two-Runtime harness like B3).
#
# GPU split on this 8xL40S box: 4 train (Megatron) + 4 inference (SGLang) + 0 env. The env checker's
# GPU is REMOTE (sandbox Runtime #2), so no local GPUs are held back for the checker.
#
# Async/decoupled: agentic rollouts have long, variable latency (multi-turn kernel repair on a
# GPU). slime's async driver (train_async.py) runs trainer GPUs and SGLang inference GPUs DISJOINT
# and generates rollout N+1 while training on rollout N. Async hard-requires no --colocate.
#
# NOTE step count: slime resumes at loaded_iter+1; a fresh torch_dist ckpt is iter 0 => the loop
# runs range(1, NR) = (NR-1) real training steps. NR=11 => 10 train (SMOKE; no more needed).
set -euo pipefail

SANDBOX="${SANDBOX:-agentcore-runtime}"   # the sandbox is its OWN GPU-backed AgentCore Runtime #2
                                          # (sandbox_server_agent.py on a g6). The KernelBench torch/CUDA
                                          # checker runs THERE (remote), so no local env GPUs are used.
# ---- Task set: the REAL KernelBench (ScalingIntelligence/KernelBench) ----------------------------
# Train = level_1 (100 single-op CUDA problems); held-out eval = 20 level_2 fusion problems (disjoint,
# no leakage). Build with build_dataset.py --dataset kernelbench --level level_1 --limit 100 (train)
# and --level level_2 --limit 20 (eval). This replaced the old hand-written 3-op smoke stub, so real
# RB/NR now work. Keep the save-only-at-end invariant: dataset_size (100) >= NR * RB (see below);
# slime force-saves at every epoch boundary (step % (dataset_size/RB) == 0), so with RB=4 the epoch
# is 100/4 = 25 steps and NR=11 (10 real steps) never crosses it => the ONLY checkpoint is the final.
RB="${RB:-4}"          # prompts per rollout step
NS="${NS:-8}"          # GRPO group size (samples per prompt)
NR="${NR:-11}"         # num-rollout; real training steps = NR-1 => 10 train (SMOKE; no more needed). Keep NR*RB <= 100.
TEMP="${TEMP:-1.0}"    # rollout sampling temperature (higher => more intra-group diversity)
GBS=$(( RB * NS ))     # global batch = all samples, 1 optimizer step per rollout
EVAL_INTERVAL="${EVAL_INTERVAL:-10}"  # run held-out eval every N training steps (0 => off)
EVAL_NS="${EVAL_NS:-2}"              # samples per eval prompt (eval solve-rate resolution)
# Checkpoint policy: no mid-training snapshots. slime force-saves at the final rollout, so a huge
# save-interval means the periodic save never fires; the ONLY checkpoint is the final one.
SAVE_INTERVAL="${SAVE_INTERVAL:-1000000}"

SLIME=/root/slime
DEMO_DIR=/work/code/demos/demo_gpu_single
# Base Megatron checkpoint is SHARED (7.5 GB) — referenced, not copied. Override via $CKPT_BASE.
CKPT="${CKPT_BASE:-/work/code/p4_slime_train/ckpt/Qwen3-4B_torch_dist}"
HF=/root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c

# ---- GPU split: 4 train + 4 inference + 0 env. The KernelBench checker's GPU is REMOTE (on the
# sandbox Runtime #2 g6 instance), so no reserved local env GPUs here.
ACTOR_GPUS="${ACTOR_GPUS:-4}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-4}"
DATA="${DATA:-$DEMO_DIR/data/kernelbench_train.jsonl}"
EVAL_DATA="${EVAL_DATA:-$DEMO_DIR/data/kernelbench_eval.jsonl}"
EVAL_NAME="${EVAL_NAME:-kernelbench}"
SAVE="${SAVE:-$DEMO_DIR/ckpt/Qwen3-4B_kernelbench_async}"

# ---- Remote GPU sandbox Runtime #2 (REQUIRED) ----
# run_episode's checker exec (python3 run_checks.py imports torch on CUDA) runs on the REMOTE g6
# sandbox Runtime, reached over invoke_agent_runtime. No adapter/harness Runtime: demo_gpu_single's
# generate.generate drives SGLang directly and only the SANDBOX is remote. Deploy the GPU sandbox
# (deploy_two_runtime.py) + run its one-time mode=prepare, then export its ARN:
AGENTCORE_SANDBOX_RUNTIME_ARN="${AGENTCORE_SANDBOX_RUNTIME_ARN:?set to the GPU sandbox Runtime #2 ARN}"

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
   # The end-of-training save intermittently DEADLOCKED: Megatron's checkpoint writer forks a
   # writer off the MAIN training process; forking a process holding CUDA/NCCL/inductor threads
   # can inherit a locked mutex with no thread to release it -> the writer hangs forever AFTER
   # the .distcp files are written. The safe backend is the SPAWN-based PersistentAsyncCaller,
   # selected only when --async-save is set AND init_persistent_async_worker() is called. slime
   # forces use_persistent_ckpt_worker=True but its own init() never makes that call, so
   # --async-save alone falls back to the fork-based caller and can still deadlock. So we do both:
   #   1. --async-save
   #   2. --custom-megatron-init-path ...enable_persistent_async_ckpt (the one line slime omits;
   #      see agentcore_slime_demo/megatron_init.py).
   --async-save
   --custom-megatron-init-path agentcore_slime_demo.megatron_init.enable_persistent_async_ckpt
)

# Held-out EVAL every EVAL_INTERVAL steps (routed through our SAME custom generate with
# evaluation=True). The held-out eval set is 20 KernelBench level_2 fusions, disjoint from the
# level_1 train set (no leakage).
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
# Give ray ALL 8 physical GPUs. slime's placement reserves (ACTOR_GPUS + ROLLOUT_GPUS)=8; the
# KernelBench checker's GPU is on the REMOTE sandbox Runtime #2, so no local GPUs are held back.
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus 8 \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM:${DEMO_DIR}:/root/slime\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"SANDBOX\": \"${SANDBOX}\",
    \"AWS_REGION\": \"${AWS_REGION:-us-west-2}\",
    \"AGENTCORE_REGION\": \"${AGENTCORE_REGION:-${AWS_REGION:-us-west-2}}\",
    \"AWS_BEARER_TOKEN_BEDROCK\": \"${AWS_BEARER_TOKEN_BEDROCK:-}\",
    \"AGENTCORE_SANDBOX_RUNTIME_ARN\": \"${AGENTCORE_SANDBOX_RUNTIME_ARN}\"
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
