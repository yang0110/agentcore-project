#!/bin/bash
# demo_gpu_farm_whole — KernelBench on a LOCAL Ray-farm GPU sandbox: several GPUs, ONE WHOLE GPU per episode.
# Runner (this dir owns data/ + ckpt/; agentcore_slime_demo is the vendored local copy in this dir). Runs INSIDE the
# persistent slime container (agentcore_p4), async/decoupled (train_async.py, NO --colocate).
#
# The whole-GPU case (was demo_gpu_farm FARM=whole, before that demo_gpu_shared_multi == demo_gpu_local):
# B4_FARM_GPUS=2, B4_GPU_FRACTION=1.0 -> floor(2/1.0)=2 Ray actors, one WHOLE GPU each (host GPUs 6,7).
# No two episodes share a GPU, so no software memory guard is needed and the non-MIG nvcc wedge of
# fractional sharing is sidestepped; the host (RAM, CPU, kernel, driver) is still shared.
# Related: demo_gpu_farm_fractional (two episodes on ONE GPU) and demo_gpu_single (the remote g6 Runtime).
#
# AgentCore reality: 1 runtimeSessionId -> 1 instance. So this is ONE shared runtimeSessionId
# (B4_SHARED_RUNTIME_SESSION) + a per-episode sub_session, and the Ray farm INSIDE that one local
# runtime container leases one whole-GPU actor per sub_session. Isolation details: README.md.
#
# PREREQ — start the LOCAL runtime container on the HOST first (NOT inside agentcore_p4); it reserves
# GPUs 6,7 and serves the farm at http://127.0.0.1:8080 (agentcore_p4 is --network host):
#     bash 4_deploy_runtime.sh        # or: bash code/deploy/run_local_gpu_runtime.sh whole
# Then, inside agentcore_p4:
#     bash code/demos/demo_gpu_farm_whole/run.sh
#
# Same harness as demo_gpu_single: the run_episode loop (agentcore_slime_demo.harness) runs
# IN-PROCESS on this box inside generate.generate, driving SGLang DIRECTLY (SGLangPolicy). slime gets
# GPUs 0-5 (4 train + 2 infer = 6); GPUs 6,7 belong to the local farm container.
#
# NOTE step count: slime resumes at loaded_iter+1; a fresh torch_dist ckpt is iter 0 => the loop
# runs range(1, NR) = (NR-1) real training steps. NR=11 => 10 train (SMOKE; no more needed).
set -euo pipefail

# ---- Farm split (hard-set; MUST match run_local_gpu_runtime.sh whole) -----------------------------
FARM=whole                                 # this demo IS the whole-GPU farm; not a selector
_FARM_GPUS=2; _FRACTION=1.0                # GPUs 6,7: floor(2/1.0)=2 slots, one whole GPU each

SANDBOX="${SANDBOX:-agentcore-ray-farm}"  # the local Ray farm backend (RayFarmRuntimeSandbox): a
                                          # per-episode sub_session slot on the ONE local runtime:
                                          # a whole-GPU actor.
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
DEMO_DIR=/work/code/demos/demo_gpu_farm_whole
CKPT="${CKPT_BASE:-/work/code/p4_slime_train/ckpt/Qwen3-4B_torch_dist}"   # shared base ckpt (referenced, not copied)
HF=/root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c

DATA="${DATA:-$DEMO_DIR/data/kernelbench_train.jsonl}"
EVAL_DATA="${EVAL_DATA:-$DEMO_DIR/data/kernelbench_eval.jsonl}"
EVAL_NAME="${EVAL_NAME:-kernelbench}"
SAVE="${SAVE:-$DEMO_DIR/ckpt/Qwen3-4B_kernelbench_farm_whole_async}"

# ---- GPU split: slime on 0-5, the local farm on GPUs 6,7 ------------------------------------------
# Reserve GPUs 6,7 for the local Ray-farm sandbox runtime container (started on the HOST via
# run_local_gpu_runtime.sh whole) and give slime GPUs 0-5 (4 train + 2 infer). The client reaches the farm over http://127.0.0.1:8080; no remote ARN.
ACTOR_GPUS="${ACTOR_GPUS:-4}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-2}"                       # 4 train + 2 infer = 6; farm GPU(s) reserved
SLIME_NUM_GPUS="${SLIME_NUM_GPUS:-6}"
SLIME_CUDA_VISIBLE="${SLIME_CUDA_VISIBLE:-0,1,2,3,4,5}" # keep slime OFF the reserved farm GPU(s) (6 / 6,7)
AGENTCORE_LOCAL_ENDPOINT="${AGENTCORE_LOCAL_ENDPOINT:-http://127.0.0.1:8080}"
AGENTCORE_SANDBOX_RUNTIME_ARN="${AGENTCORE_SANDBOX_RUNTIME_ARN:-}"   # local endpoint, no remote ARN

# Farm math: 2 GPUs / 1.0 => 2 whole-GPU slots (6,7). These MUST match the farm knobs baked on the local runtime container by
# run_local_gpu_runtime.sh whole. They are read here only to size the client OPEN gate below.
export B4_FARM_GPUS="${B4_FARM_GPUS:-$_FARM_GPUS}"
export B4_GPU_FRACTION="${B4_GPU_FRACTION:-$_FRACTION}"

# ONE shared runtimeSessionId for the WHOLE job: every episode (across every rollout worker) invokes
# the local runtime under this id, so they all land on the SAME farm whose GPUs it splits. Minted
# once here if unset (AgentCore requires length in [33,128]: "b4farm-" + 32 hex = 39). Pin it to reuse
# a warm instance across runs.
B4_SHARED_RUNTIME_SESSION="${B4_SHARED_RUNTIME_SESSION:-b4farm-$(python3 -c 'import secrets;print(secrets.token_hex(16))')}"
# Client-side OPEN-concurrency gate (agentcore_runtime_sandbox._farm_open_gate): cap how many episodes
# hold a farm slot at once to the farm's actor count. The async rollout submits RB*NS episodes that
# each open their own sandbox, and neither --sglang-server-concurrency (gates GENERATION) nor NS caps
# `op=open`; without this gate the overflow opens block on the SERVER farm gate until the invoke times
# out (~min) and stall the step. Must EQUAL the server's B4_MAX_SANDBOXES = floor(B4_FARM_GPUS/
# B4_GPU_FRACTION) (2 slots here).
export B4_FARM_MAX_CONCURRENT="${B4_FARM_MAX_CONCURRENT:-$(python3 -c "import os,math;print(max(1,math.floor(float(os.environ['B4_FARM_GPUS'])/float(os.environ['B4_GPU_FRACTION']))))")}"
# Runtime-invoke transient-retry budget (agentcore_runtime_sandbox._retry_invoke). We invoke the farm
# once per tool op (open + writes + exec + close), so a transient 502/blip on ANY of them would kill
# the whole job at the default 8 attempts (~150s). 20 attempts ≈ 10 min patience.
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
   # SAME hook as demo_gpu_single — the in-process run_episode harness (SGLangPolicy direct).
   # The GPU packing is entirely in the sandbox backend (SANDBOX=agentcore-ray-farm ->
   # RayFarmRuntimeSandbox), which make_sandbox() selects per episode, so no custom generate is needed.
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
   # Keep rollout concurrency <= the local farm's slot count (B4_MAX_SANDBOXES = 2 here
   # defaults) so episodes don't block on the farm's checkout gate. Raise together with the farm.
   --sglang-server-concurrency "${CONC:-2}"
   # NOTE: no tool-call/reasoning parser here (unlike the old MiniClaudeHarness path). run_episode
   # extracts a ```python code block from the assistant text (task.extract_submission) and suppresses
   # <think> via SGLangPolicy's enable_thinking=False — exactly like demo_gpu_single.
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
# Ray gets SLIME_NUM_GPUS (6 here). We pin CUDA_VISIBLE_DEVICES to 0-5 BEFORE `ray start` so ray only
# discovers/schedules those 6 — GPUs 6,7 stay free for the local Ray-farm sandbox runtime container.
# (Do NOT set CUDA_VISIBLE_DEVICES in the ray job runtime-env: ray assigns it per-actor.)
[ -n "$SLIME_CUDA_VISIBLE" ] && export CUDA_VISIBLE_DEVICES="$SLIME_CUDA_VISIBLE"
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "$SLIME_NUM_GPUS" \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# HEARTBEAT (default ON for our own remote Runtimes, SANDBOX=agentcore-* with no local endpoint; off
# for the Code Interpreter and the local farm, which has no idle timeout): a process-wide
# SessionKeeper heartbeats every open sandbox session that goes idle past
# HEARTBEAT_PING_FRACTION*HEARTBEAT_IDLE, so a trainer stall can't idle-reclaim it — a reclaimed
# session id silently comes back EMPTY (agentcore_slime_demo/session_keeper.py). HEARTBEAT=0 is the
# A/B baseline. Watch the [demo] heartbeat_session=… lines (0 with HEARTBEAT=0).
HEARTBEAT="${HEARTBEAT:-${HEARTBEAT_KEEP_WARM:-auto}}"
if [ "$HEARTBEAT" = auto ]; then
  case "$SANDBOX" in agentcore-*) HEARTBEAT=1 ;; *) HEARTBEAT=0 ;; esac
  if [ -n "${AGENTCORE_LOCAL_ENDPOINT:-}" ]; then HEARTBEAT=0; fi
fi
# HEARTBEAT_IDLE must be <= the Runtime's idleRuntimeSessionTimeout. 600 s is safe for every Runtime:
# older ones were created with 600 s, new ones get RUNTIME_IDLE_TIMEOUT (3600 s, deploy/agentcore_deploy.py).
HEARTBEAT_IDLE="${HEARTBEAT_IDLE:-600}"
HEARTBEAT_PING_FRACTION="${HEARTBEAT_PING_FRACTION:-0.5}"
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM:${DEMO_DIR}:/root/slime\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\",
    \"SANDBOX\": \"${SANDBOX}\",
    \"HEARTBEAT\": \"${HEARTBEAT}\",
    \"HEARTBEAT_IDLE\": \"${HEARTBEAT_IDLE}\",
    \"HEARTBEAT_PING_FRACTION\": \"${HEARTBEAT_PING_FRACTION}\",
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
