#!/bin/bash
# Demo B4 — KernelBench, REMOTE g6 sandbox, with SEVERAL episodes SHARING that g6's ONE GPU via Ray.
# Runner (this dir owns data/ + ckpt/; agentcore_slime_demo is the vendored local copy in this dir). Runs INSIDE the
# persistent slime container (agentcore_p4), async/decoupled (train_async.py, NO --colocate).
#
# Demo B4 == Demo B (SAME task + SAME in-process run_episode harness), but a DIFFERENT sandbox GPU
# story. B4 is now IDENTICAL to Demo B except for ONE line — the SANDBOX backend:
#   * Demo B  (SANDBOX=agentcore-runtime):  one AgentCore session -> one g6 instance, so N concurrent
#     rollouts want N g6 instances. On a capacity-starved g6 fleet that is exactly what stalls it.
#   * Demo B4 (SANDBOX=agentcore-ray-farm): pack SEVERAL concurrent KernelBench sandboxes onto ONE g6
#     instance and let RAY SPLIT THAT g6's single GPU between them. Because AgentCore maps
#     1 runtimeSessionId -> 1 instance, every B4 episode invokes the g6 under ONE shared
#     runtimeSessionId (B4_SHARED_RUNTIME_SESSION, so they reuse the same g6) and carries a
#     per-episode sub_session; a Ray farm running INSIDE the g6 (sandbox_ray_farm.RayGpuFarm, driven
#     by sandbox_server_agent) leases one FRACTIONAL-GPU actor per sub_session. Ray's fraction is
#     scheduling-only (no memory guard; L4/L40S have NO MIG), so each sandbox ALSO caps its GPU memory
#     in software (an injected sitecustomize.py). That server-side farm is the WHOLE difference.
#
# Same harness as Demo B: the run_episode loop (agentcore_slime_demo.harness) runs IN-PROCESS on this
# box inside generate.generate, driving SGLang DIRECTLY (SGLangPolicy) — no AnthropicAdapter proxy,
# no MiniClaudeHarness, no harness Runtime #1. This box gives ALL 8 of its GPUs to slime (4 train +
# 4 infer); B4 reserves NO local GPU. The only remote piece is the shared-GPU sandbox Runtime #2 (its
# g6 image bakes `ray` + the B4_* farm config). So B4 adds exactly ONE thing over Demo B: the Ray
# GPU-sharing farm inside the g6 sandbox, selected purely by SANDBOX=agentcore-ray-farm.
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

SANDBOX="${SANDBOX:-agentcore-ray-farm}"  # THE one difference from Demo B: sub_session slots on ONE
                                          # shared g6 whose GPU a Ray farm splits (vs B's dedicated g6)
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
DEMO_DIR=/work/code/demos/demo_gpu_shared
CKPT="${CKPT_BASE:-/work/code/p4_slime_train/ckpt/Qwen3-4B_torch_dist}"   # shared base ckpt (referenced, not copied)
HF=/root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c

DATA="${DATA:-$DEMO_DIR/data/kernelbench_train.jsonl}"
EVAL_DATA="${EVAL_DATA:-$DEMO_DIR/data/kernelbench_eval.jsonl}"
EVAL_NAME="${EVAL_NAME:-kernelbench}"
SAVE="${SAVE:-$DEMO_DIR/ckpt/Qwen3-4B_kernelbench_rayfarm_async}"

# ---- GPU split + where the shared-GPU sandbox Runtime lives --------------------------------------
# DEFAULT (remote): ALL 8 GPUs to slime (4 train + 4 infer); the shared-GPU checker is REMOTE (the
# shared g6 whose one GPU a Ray farm splits) — identical to B2/B3.
# LOCAL_GPU=1 fallback (remote g6 unavailable): run the SAME Ray-farm sandbox Runtime LOCALLY on THIS
# box. We reserve GPUs 6,7 for the local farm container (a `docker run` of sandbox_server_agent with
# the ray farm + B4_FARM_GPUS=2 — start it on the HOST first: `code/deploy/run_local_gpu_runtime.sh
# shared`) and give slime 6 GPUs. Still one AgentCore runtime session (shared runtimeSessionId +
# per-episode sub_session), reached over http://127.0.0.1:8080 (agentcore_p4 is --network host); the
# farm now splits the LOCAL reserved GPUs instead of the g6's.
LOCAL_GPU="${LOCAL_GPU:-0}"
if [ "$LOCAL_GPU" = "1" ]; then
   ACTOR_GPUS="${ACTOR_GPUS:-4}"
   ROLLOUT_GPUS="${ROLLOUT_GPUS:-2}"                       # 4 train + 2 infer = 6; GPUs 6,7 reserved
   SLIME_NUM_GPUS="${SLIME_NUM_GPUS:-6}"
   SLIME_CUDA_VISIBLE="${SLIME_CUDA_VISIBLE:-0,1,2,3,4,5}" # keep slime OFF the reserved farm GPUs
   AGENTCORE_LOCAL_ENDPOINT="${AGENTCORE_LOCAL_ENDPOINT:-http://127.0.0.1:8080}"
   AGENTCORE_SANDBOX_RUNTIME_ARN="${AGENTCORE_SANDBOX_RUNTIME_ARN:-}"   # no remote ARN needed
else
   ACTOR_GPUS="${ACTOR_GPUS:-4}"
   ROLLOUT_GPUS="${ROLLOUT_GPUS:-4}"                       # 4 train + 4 infer = 8; farm GPU is remote
   SLIME_NUM_GPUS="${SLIME_NUM_GPUS:-8}"
   SLIME_CUDA_VISIBLE="${SLIME_CUDA_VISIBLE:-}"
   AGENTCORE_LOCAL_ENDPOINT="${AGENTCORE_LOCAL_ENDPOINT:-}"
   AGENTCORE_SANDBOX_RUNTIME_ARN="${AGENTCORE_SANDBOX_RUNTIME_ARN:?set to the B4 GPU-shared sandbox Runtime #2 ARN (g6 with ray farm baked in), or set LOCAL_GPU=1 to serve the farm locally on GPUs 6,7}"
fi
# ONE shared runtimeSessionId for the WHOLE job: every episode (across every rollout worker) invokes
# the g6 under this id, so they all land on the SAME g6 instance whose one GPU the farm splits. Minted
# once here if unset (AgentCore requires length in [33,128]: "b4farm-" + 32 hex = 39). Pin it to reuse
# a warm instance across runs.
B4_SHARED_RUNTIME_SESSION="${B4_SHARED_RUNTIME_SESSION:-b4farm-$(python3 -c 'import secrets;print(secrets.token_hex(16))')}"
# Client-side OPEN-concurrency gate (agentcore_runtime_sandbox._farm_open_gate): cap how many episodes
# hold a farm slot at once to the g6 farm's actor count. The async rollout submits RB*NS episodes that
# each open their own sandbox, and neither --sglang-server-concurrency (gates GENERATION) nor NS caps
# `op=open`; without this gate the overflow opens block on the SERVER farm gate until the invoke times
# out (~min) and stall the step. Must EQUAL the server's B4_MAX_SANDBOXES = floor(B4_FARM_GPUS/
# B4_GPU_FRACTION) (set on the runtime by run_local_gpu_runtime.sh; default 1 GPU / 0.25 = 4 slots).
export B4_FARM_MAX_CONCURRENT="${B4_FARM_MAX_CONCURRENT:-$(python3 -c "import os,math;print(max(1,math.floor(float(os.environ.get('B4_FARM_GPUS','1'))/float(os.environ.get('B4_GPU_FRACTION','0.25')))))")}"
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
   # SYNC save (no --async-save): this local sandbox-runtime container's seccomp profile blocks
   # pidfd_getfd, which slime's persistent async-ckpt child uses to grab the CUDA storage fds
   # ("RuntimeError: pidfd_getfd: Operation not permitted") — the async save hangs the job even at
   # PP=1. Same class of failure the PP=2 demos hit, so we take the same fix: save synchronously.
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
   # SAME hook as Demo B — the in-process run_episode harness (SGLangPolicy direct). The GPU-sharing
   # is entirely in the sandbox backend (SANDBOX=agentcore-ray-farm -> RayFarmRuntimeSandbox), which
   # make_sandbox() selects per episode, so no B4-specific generate function is needed.
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
   # Keep rollout concurrency <= the g6 farm's sandbox count (B4_MAX_SANDBOXES, baked on the g6) so
   # episodes don't block on the farm's checkout gate. Default 4; raise together with the farm.
   --sglang-server-concurrency "${CONC:-4}"
   # NOTE: no tool-call/reasoning parser here (unlike the old MiniClaudeHarness path). run_episode
   # extracts a ```python code block from the assistant text (task.extract_submission) and suppresses
   # <think> via SGLangPolicy's enable_thinking=False — exactly like Demo B, which sets neither flag.
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
# Ray gets SLIME_NUM_GPUS (8 remote-mode; 6 in LOCAL_GPU mode). In LOCAL_GPU mode we pin
# CUDA_VISIBLE_DEVICES to 0-5 BEFORE `ray start` so ray only discovers/schedules those 6 — GPUs 6,7
# stay free for the local Ray-farm sandbox Runtime container. (Do NOT set CUDA_VISIBLE_DEVICES in the
# ray job runtime-env: ray assigns it per-actor.)
[ -n "$SLIME_CUDA_VISIBLE" ] && export CUDA_VISIBLE_DEVICES="$SLIME_CUDA_VISIBLE"
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "$SLIME_NUM_GPUS" \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM:${DEMO_DIR}:/root/slime\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\",
    \"SANDBOX\": \"${SANDBOX}\",
    \"AWS_REGION\": \"${AWS_REGION:-us-west-2}\",
    \"AGENTCORE_REGION\": \"${AGENTCORE_REGION:-${AWS_REGION:-us-west-2}}\",
    \"AWS_BEARER_TOKEN_BEDROCK\": \"${AWS_BEARER_TOKEN_BEDROCK:-}\",
    \"AGENTCORE_SANDBOX_RUNTIME_ARN\": \"${AGENTCORE_SANDBOX_RUNTIME_ARN}\",
    \"AGENTCORE_LOCAL_ENDPOINT\": \"${AGENTCORE_LOCAL_ENDPOINT}\",
    \"B4_SHARED_RUNTIME_SESSION\": \"${B4_SHARED_RUNTIME_SESSION}\",
    \"INVOKE_MAX_RETRIES\": \"${INVOKE_MAX_RETRIES}\"
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
