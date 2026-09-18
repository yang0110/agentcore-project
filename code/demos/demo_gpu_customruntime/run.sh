#!/bin/bash
# demo_gpu_customruntime (KernelBench) — REAL slime training, ASYNC (decoupled), on YOUR CUSTOM GPU
# AgentCore Runtime. Runs INSIDE the persistent container (agentcore_p4). This dir owns its data/ +
# ckpt/ and imports its own vendored agentcore_slime_demo copy in this dir (PYTHONPATH); it
# references only genuinely SHARED infrastructure (the container, the base Megatron checkpoint,
# /root/slime).
#
# WHAT THIS DEMO IS (the custom-Runtime point, now as a real trainer). KernelBench-style CUDA-kernel
# tasks: the policy writes solution.py, a GPU checker runs it (python3 run_checks.py imports torch,
# compiles/executes on cuda, compares against a torch reference), reads the failure, fixes it — our
# lightweight run_episode harness (agentcore_slime_demo.harness) drives the multi-turn loop
# IN-PROCESS on this box, but the sandbox (write_file / exec / read_file) is a SEPARATE GPU-backed
# AgentCore Runtime on a g6 Instance (sandbox_server_agent.py), reached over invoke_agent_runtime.
#
# WHY IT MUST BE A CUSTOM RUNTIME. The KernelBench checker needs a real CUDA device INSIDE the
# sandbox. The managed AgentCore Code Interpreter (SANDBOX=agentcore) is a CPU-only serverless
# microVM — torch.cuda.is_available() is False there — so this training CANNOT run on it. CUDA only
# appears when the sandbox is YOUR OWN custom Runtime deployed on an AgentCore capacity-provider
# *Instance* (a g6 GPU box), which this demo defaults to (SANDBOX=agentcore-runtime). This is the
# custom-Runtime capability made into a real RL run, not just a probe. gpu_torch_smoke.py remains as
# the seconds-long OFFLINE CUDA-contract probe (write_file -> exec -> read_file of a torch.cuda
# check) that shows the same contrast across local / agentcore / agentcore-runtime without training.
#
# GPU split on this 8xL40S box: 4 train (Megatron) + 4 inference (SGLang) + 0 env. The KernelBench
# checker's GPU is REMOTE (the custom GPU Runtime's g6), so no local GPUs are held back for it.
#
# Async/decoupled: agentic rollouts have long, variable latency (multi-turn kernel repair on a
# GPU). slime's async driver (train_async.py) runs trainer GPUs and SGLang inference GPUs DISJOINT
# and generates rollout N+1 while training on rollout N. Async hard-requires no --colocate.
#
# NOTE step count: slime resumes at loaded_iter+1; a fresh torch_dist ckpt is iter 0 => the loop
# runs range(1, NR) = (NR-1) real training steps. NR=11 => 10 train (SMOKE; no more needed).
set -euo pipefail

# ---- ON-BOX validation path (LOCAL_GPU=1) --------------------------------------------------------
# The custom GPU Runtime this demo trains on is normally a REMOTE g6 Instance (SANDBOX=agentcore-runtime,
# needs an ARN). To smoke the SAME trainer on THIS 8xL40S box with no remote g6, set LOCAL_GPU=1: it
# serves the custom GPU Runtime LOCALLY as a whole-GPU Ray farm on reserved GPUs 6,7 (SANDBOX=
# agentcore-ray-farm, local endpoint :8080) and keeps slime on GPUs 0-5 — exactly like demo_gpu_local.
# The local farm is still the SAME sandbox_server_agent GPU image, just run via docker on the host, so
# it validates the custom-Runtime CUDA path end-to-end. Start the farm on the HOST first:
#     bash code/deploy/run_local_gpu_runtime.sh local
LOCAL_GPU="${LOCAL_GPU:-0}"
if [ "$LOCAL_GPU" = "1" ]; then
  SANDBOX="${SANDBOX:-agentcore-ray-farm}"                 # local Ray-farm backend (RayFarmRuntimeSandbox)
  ACTOR_GPUS="${ACTOR_GPUS:-4}"; ROLLOUT_GPUS="${ROLLOUT_GPUS:-2}"   # 4 train + 2 infer = 6; GPUs 6,7 -> farm
  SLIME_NUM_GPUS="${SLIME_NUM_GPUS:-6}"; SLIME_CUDA_VISIBLE="${SLIME_CUDA_VISIBLE:-0,1,2,3,4,5}"
  AGENTCORE_LOCAL_ENDPOINT="${AGENTCORE_LOCAL_ENDPOINT:-http://127.0.0.1:8080}"
  AGENTCORE_SANDBOX_RUNTIME_ARN="${AGENTCORE_SANDBOX_RUNTIME_ARN:-}"  # local endpoint, no remote ARN
  CONC="${CONC:-2}"                                        # <= whole-GPU farm slots (2)
  B4_SHARED_RUNTIME_SESSION="${B4_SHARED_RUNTIME_SESSION:-b4farm-$(python3 -c 'import secrets;print(secrets.token_hex(16))')}"
  export B4_FARM_MAX_CONCURRENT="${B4_FARM_MAX_CONCURRENT:-2}"   # client open-gate == server whole-GPU slots
  INVOKE_MAX_RETRIES="${INVOKE_MAX_RETRIES:-20}"
fi

SANDBOX="${SANDBOX:-agentcore-runtime}"   # the sandbox is YOUR custom GPU-backed AgentCore Runtime on
                                          # a g6 Instance (sandbox_server_agent.py). The KernelBench
                                          # torch/CUDA checker runs THERE (remote) — the ONLY backend
                                          # with a GPU. SANDBOX=agentcore (Code Interpreter) is CPU-only
                                          # and cannot serve this training.
# ---- Task set: the REAL KernelBench (ScalingIntelligence/KernelBench) ----------------------------
# Train = level_1 (100 single-op CUDA problems); held-out eval = 20 level_2 fusion problems (disjoint,
# no leakage). Build with build_dataset.py --dataset kernelbench --level level_1 --limit 100 (train)
# and --level level_2 --limit 20 (eval). Save-only-at-end holds regardless of dataset size: run.sh
# passes an explicit --num-rollout, so slime leaves num_rollout_per_epoch=None and its epoch-boundary
# save never fires; with the huge SAVE_INTERVAL below the periodic save never fires either => the
# ONLY checkpoint is slime's forced final-rollout one. (Keep NR*RB <= 100 only so a run stays within
# one pass of the 100-prompt level_1 pool, not for checkpoint reasons.)
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
DEMO_DIR=/work/code/demos/demo_gpu_customruntime
# Base Megatron checkpoint is SHARED (7.5 GB) — referenced, not copied. Override via $CKPT_BASE.
CKPT="${CKPT_BASE:-/work/code/p4_slime_train/ckpt/Qwen3-4B_torch_dist}"
HF=/root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c

# ---- GPU split + where the sandbox Runtime lives -------------------------------------------------
# ALL 8 GPUs to slime (4 train + 4 infer); the KernelBench checker's GPU is REMOTE (the custom GPU
# Runtime's g6), reached over invoke_agent_runtime. For an on-box GPU run with no remote g6, use
# demo_gpu_local (a LOCAL whole-GPU Ray farm on this box's reserved GPUs 6,7) instead.
ACTOR_GPUS="${ACTOR_GPUS:-4}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-4}"                       # 4 train + 4 infer = 8; checker GPU is remote
SLIME_NUM_GPUS="${SLIME_NUM_GPUS:-8}"
SLIME_CUDA_VISIBLE="${SLIME_CUDA_VISIBLE:-}"
AGENTCORE_LOCAL_ENDPOINT="${AGENTCORE_LOCAL_ENDPOINT:-}"
# Remote path requires the g6 ARN; the LOCAL_GPU on-box path uses the local endpoint instead (no ARN).
if [ "$LOCAL_GPU" != "1" ]; then
  AGENTCORE_SANDBOX_RUNTIME_ARN="${AGENTCORE_SANDBOX_RUNTIME_ARN:?set to your custom GPU sandbox Runtime ARN (a g6 Instance with CUDA — deploy_two_runtime.py / redeploy_gpu_sandbox.py). The managed Code Interpreter has no GPU; for an on-box GPU run with no remote g6, use LOCAL_GPU=1 (or demo_gpu_local)}"
fi
DATA="${DATA:-$DEMO_DIR/data/kernelbench_train.jsonl}"
EVAL_DATA="${EVAL_DATA:-$DEMO_DIR/data/kernelbench_eval.jsonl}"
EVAL_NAME="${EVAL_NAME:-kernelbench}"
SAVE="${SAVE:-$DEMO_DIR/ckpt/Qwen3-4B_gpu_customruntime_async}"

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
)
# ---- Checkpoint-save mode -----------------------------------------------------------------------
# REMOTE (g6) path uses ASYNC persistent-ckpt save (two flags that MUST go together): Megatron's
# checkpoint writer otherwise forks off the MAIN training process; forking while holding
# CUDA/NCCL/inductor threads can inherit a locked mutex with no thread to release it -> the writer
# hangs forever AFTER the .distcp files are written. The safe backend is the SPAWN-based
# PersistentAsyncCaller, selected only when --async-save is set AND init_persistent_async_worker()
# is called; slime forces use_persistent_ckpt_worker=True but never makes that call, so we add it via
# --custom-megatron-init-path (see agentcore_slime_demo/megatron_init.py).
# LOCAL_GPU on-box path saves SYNCHRONOUSLY instead: the local Ray-farm runtime container's seccomp
# blocks pidfd_getfd (which the persistent async-ckpt child uses to grab CUDA storage fds), so async
# save hangs on-box — matching demo_gpu_local's sync-save fix.
if [ "$LOCAL_GPU" != "1" ]; then
   CKPT_ARGS+=(
      --async-save
      --custom-megatron-init-path agentcore_slime_demo.megatron_init.enable_persistent_async_ckpt
   )
fi

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
# Ray gets SLIME_NUM_GPUS (8; the checker GPU is the remote g6's). SLIME_CUDA_VISIBLE is empty here,
# so ray discovers all 8 local GPUs. (Do NOT put CUDA_VISIBLE_DEVICES in the ray job runtime-env: ray
# sets it per-actor from the GPUs it assigns.)
[ -n "$SLIME_CUDA_VISIBLE" ] && export CUDA_VISIBLE_DEVICES="$SLIME_CUDA_VISIBLE"
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "$SLIME_NUM_GPUS" \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM:${DEMO_DIR}:/root/slime\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"SANDBOX\": \"${SANDBOX}\",
    \"AWS_REGION\": \"${AWS_REGION:-us-west-2}\",
    \"AGENTCORE_REGION\": \"${AGENTCORE_REGION:-${AWS_REGION:-us-west-2}}\",
    \"AWS_BEARER_TOKEN_BEDROCK\": \"${AWS_BEARER_TOKEN_BEDROCK:-}\",
    \"AGENTCORE_SANDBOX_RUNTIME_ARN\": \"${AGENTCORE_SANDBOX_RUNTIME_ARN:-}\",
    \"AGENTCORE_LOCAL_ENDPOINT\": \"${AGENTCORE_LOCAL_ENDPOINT}\",
    \"B4_SHARED_RUNTIME_SESSION\": \"${B4_SHARED_RUNTIME_SESSION:-}\",
    \"INVOKE_MAX_RETRIES\": \"${INVOKE_MAX_RETRIES:-8}\"
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
