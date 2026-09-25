#!/bin/bash
# demo_gpu_single (KernelBench) — REAL slime training, ASYNC (decoupled), on ONE warm custom GPU Runtime (g6).
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
# ONE DEMO, THREE ANGLES (merged 2026-09-25 from demo_gpu_single + demo_gpu_snapshot + demo_gpu_customruntime):
#   * default               warm-shared g6 custom Runtime, KernelBench CUDA checker remote.
#   * SNAPSHOT_EVERY=1      + APPLICATION-LEVEL snapshot/resume (was demo_gpu_snapshot): the harness
#                           restores each task's workspace from a store you own at episode ENTER and
#                           snapshots it at EXIT. AgentCore has no VM/GPU snapshot (docs/gpu-snapshot.md);
#                           see agentcore_slime_demo/app_snapshot.py, offline animation: snapshot_demo.py.
#   * WHY A CUSTOM RUNTIME  (was demo_gpu_customruntime): the managed Code Interpreter (SANDBOX=agentcore)
#                           is a CPU-only microVM (torch.cuda.is_available() is False), so this training
#                           needs YOUR custom Runtime on a g6 Instance. Offline probe: gpu_torch_smoke.py.
#
# NOTE step count: slime resumes at loaded_iter+1; a fresh torch_dist ckpt is iter 0 => the loop
# runs range(1, NR) = (NR-1) real training steps. NR=11 => 10 train (SMOKE; no more needed).
set -euo pipefail

# ON-BOX SMOKE (no remote g6): LOCAL_GPU=1 serves the SAME sandbox_server_agent GPU image as a LOCAL
# whole-GPU Ray farm on this box's reserved GPUs 6,7 (identical to demo_gpu_farm FARM=whole), with slime on GPUs
# 0-5. Two-step, exactly like demo_gpu_farm FARM=whole:
#     bash code/deploy/run_local_gpu_runtime.sh whole     # start the farm on 6,7 (host; or: bash 4_deploy_runtime.sh with LOCAL_GPU=1)
#     LOCAL_GPU=1 bash code/demos/demo_gpu_single/run.sh   # then run this (inside agentcore_p4)
#     bash code/deploy/run_local_gpu_runtime.sh stop       # tear down after
# The farm speaks the SAME warm-shared contract this demo already defaults to (agentcore-ray-farm +
# B4_SHARED_RUNTIME_SESSION), just over the local endpoint instead of a remote ARN.
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

SANDBOX="${SANDBOX:-agentcore-ray-farm}"  # the sandbox is its OWN GPU-backed AgentCore Runtime #2
                                          # (sandbox_server_agent.py on a g6). The KernelBench torch/CUDA
                                          # checker runs THERE (remote), so no local env GPUs are used.
# WARM-INSTANCE (shared-g6) default: all episodes invoke the g6 under ONE shared runtimeSessionId
# (B4_SHARED_RUNTIME_SESSION) and carry a per-episode sub_session, so AgentCore keeps them on the SAME
# warm g6 (the ~min capacity-provider cold-boot is paid ONCE per run, not once per episode) and the
# server's in-process Ray farm (sandbox_ray_farm, engaged automatically by the sub_session) gives each
# episode a FRESH isolated env on a WHOLE GPU (B4_GPU_FRACTION=1.0). This replaces the old one-session-
# =one-dedicated-g6 default, whose instance count scaled with concurrency (not cost-feasible). The SAME
# g6 image serves both paths; no redeploy. (Set SANDBOX=agentcore-runtime for the legacy per-episode
# dedicated-instance path.)
B4_SHARED_RUNTIME_SESSION="${B4_SHARED_RUNTIME_SESSION:-b4farm-$(python3 -c 'import secrets;print(secrets.token_hex(16))')}"
# Client OPEN gate; must EQUAL the server's whole-GPU farm slots. Whole-GPU per episode => one slot per
# g6 GPU; default 1 (one warm g6, episodes serialized). Raise to the g6's GPU count for intra-instance
# parallelism (still ONE warm instance).
export B4_FARM_MAX_CONCURRENT="${B4_FARM_MAX_CONCURRENT:-1}"
CONC="${CONC:-1}"                         # <= whole-GPU farm slots (one warm g6, whole GPU per episode)
# ---- Task set: the REAL KernelBench (ScalingIntelligence/KernelBench) ----------------------------
# Train = level_1 (100 single-op CUDA problems); held-out eval = 20 level_2 fusion problems (disjoint,
# no leakage). Build with build_dataset.py --dataset kernelbench --level level_1 --limit 100 (train)
# and --level level_2 --limit 20 (eval). This replaced the old hand-written 3-op smoke stub, so real
# RB/NR now work. Save-only-at-end holds regardless of dataset size: run.sh passes an explicit
# --num-rollout, so slime leaves num_rollout_per_epoch=None and its epoch-boundary save never fires;
# with the huge SAVE_INTERVAL below the periodic save never fires either => the ONLY checkpoint is
# slime's forced final-rollout one. (Keep NR*RB <= 100 only so a run stays within one pass of the
# 100-prompt level_1 pool, not for checkpoint reasons.)
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

# ---- Application-snapshot knobs (SNAPSHOT_EVERY=1 turns on the snapshot/resume angle) -----------
# Every episode opens a fresh sandbox sub_session, so each episode boundary is effectively a recycle;
# the store, not the session, is the durable source of truth. Off by default.
SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-0}"            # snapshot cadence in episodes (0 = off)
SNAPSHOT_DIR="${SNAPSHOT_DIR:-/work/code/demos/demo_gpu_single/snapshots}"  # LocalSnapshotStore root
SNAPSHOT_PATHS="${SNAPSHOT_PATHS:-progress.json,work.log,solution.py}"      # workspace files to keep
RECYCLE_AFTER="${RECYCLE_AFTER:-0}"              # >0 => log a simulated instance recycle after N episodes
# SNAPSHOT_S3_BUCKET (optional): if set, use S3SnapshotStore (production; no mount, AZ-agnostic).

SLIME=/root/slime
DEMO_DIR=/work/code/demos/demo_gpu_single
# Base Megatron checkpoint is SHARED (7.5 GB) — referenced, not copied. Override via $CKPT_BASE.
CKPT="${CKPT_BASE:-/work/code/p4_slime_train/ckpt/Qwen3-4B_torch_dist}"
HF=/root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c

# ---- GPU split + where the sandbox Runtime lives -------------------------------------------------
# ALL 8 GPUs to slime (4 train + 4 infer); the KernelBench checker's GPU is REMOTE (the sandbox
# Runtime #2 g6), reached over invoke_agent_runtime. For an on-box GPU run with no remote g6, use
# demo_gpu_farm (a LOCAL whole-GPU Ray farm on this box's reserved GPUs 6,7) instead.
ACTOR_GPUS="${ACTOR_GPUS:-4}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-4}"                       # 4 train + 4 infer = 8; checker GPU is remote
SLIME_NUM_GPUS="${SLIME_NUM_GPUS:-8}"
SLIME_CUDA_VISIBLE="${SLIME_CUDA_VISIBLE:-}"
AGENTCORE_LOCAL_ENDPOINT="${AGENTCORE_LOCAL_ENDPOINT:-}"
# Remote path requires the g6 ARN; the LOCAL_GPU on-box path uses the local endpoint instead (no ARN).
if [ "$LOCAL_GPU" != "1" ]; then
  AGENTCORE_SANDBOX_RUNTIME_ARN="${AGENTCORE_SANDBOX_RUNTIME_ARN:?set to your custom GPU sandbox Runtime ARN (a g6 Instance with CUDA — bash 4_deploy_runtime.sh). The managed Code Interpreter has no GPU; for an on-box GPU run with no remote g6, use LOCAL_GPU=1 (or demo_gpu_farm)}"
fi
DATA="${DATA:-$DEMO_DIR/data/kernelbench_train.jsonl}"
EVAL_DATA="${EVAL_DATA:-$DEMO_DIR/data/kernelbench_eval.jsonl}"
EVAL_NAME="${EVAL_NAME:-kernelbench}"
_SNAP=""; [ "$SNAPSHOT_EVERY" != "0" ] && _SNAP=_snapshot
SAVE="${SAVE:-$DEMO_DIR/ckpt/Qwen3-4B_kernelbench${_SNAP}_async}"

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
# REMOTE (g6) path: ASYNC persistent-ckpt save (two flags that MUST go together). The end-of-training
# save intermittently DEADLOCKED: Megatron's checkpoint writer forks off the MAIN training process;
# forking while holding CUDA/NCCL/inductor threads can inherit a locked mutex with no thread to release
# it -> the writer hangs forever AFTER the .distcp files are written. The safe backend is the
# SPAWN-based PersistentAsyncCaller, selected only when --async-save is set AND
# init_persistent_async_worker() is called; slime forces use_persistent_ckpt_worker=True but never
# makes that call, so we add it via --custom-megatron-init-path (agentcore_slime_demo/megatron_init.py).
# LOCAL_GPU on-box path saves SYNCHRONOUSLY instead: the local Ray-farm runtime container's seccomp
# blocks pidfd_getfd (which the persistent async-ckpt child uses to grab CUDA storage fds), so async
# save hangs on-box — matching demo_gpu_farm's sync-save fix.
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

# HEARTBEAT=1 (opt-in; was the standalone demo_cpu_heartbeat): a process-wide SessionKeeper
# heartbeats every open sandbox session that goes idle past HEARTBEAT_PING_FRACTION*HEARTBEAT_IDLE,
# so the weight-sync stall can't idle-reclaim it (agentcore_slime_demo/session_keeper.py).
# Watch the [demo] heartbeat_session=… lines (0 with HEARTBEAT=0).
HEARTBEAT="${HEARTBEAT:-${HEARTBEAT_KEEP_WARM:-0}}"
HEARTBEAT_IDLE="${HEARTBEAT_IDLE:-600}"
HEARTBEAT_PING_FRACTION="${HEARTBEAT_PING_FRACTION:-0.5}"
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM:${DEMO_DIR}:/root/slime\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"SANDBOX\": \"${SANDBOX}\",
    \"HEARTBEAT\": \"${HEARTBEAT}\",
    \"HEARTBEAT_IDLE\": \"${HEARTBEAT_IDLE}\",
    \"HEARTBEAT_PING_FRACTION\": \"${HEARTBEAT_PING_FRACTION}\",
    \"AWS_REGION\": \"${AWS_REGION:-us-west-2}\",
    \"AGENTCORE_REGION\": \"${AGENTCORE_REGION:-${AWS_REGION:-us-west-2}}\",
    \"AWS_BEARER_TOKEN_BEDROCK\": \"${AWS_BEARER_TOKEN_BEDROCK:-}\",
    \"AGENTCORE_SANDBOX_RUNTIME_ARN\": \"${AGENTCORE_SANDBOX_RUNTIME_ARN:-}\",
    \"AGENTCORE_LOCAL_ENDPOINT\": \"${AGENTCORE_LOCAL_ENDPOINT}\",
    \"B4_SHARED_RUNTIME_SESSION\": \"${B4_SHARED_RUNTIME_SESSION:-}\",
    \"INVOKE_MAX_RETRIES\": \"${INVOKE_MAX_RETRIES:-8}\",
    \"SNAPSHOT_EVERY\": \"${SNAPSHOT_EVERY}\",
    \"SNAPSHOT_DIR\": \"${SNAPSHOT_DIR}\",
    \"SNAPSHOT_PATHS\": \"${SNAPSHOT_PATHS}\",
    \"RECYCLE_AFTER\": \"${RECYCLE_AFTER}\",
    \"SNAPSHOT_S3_BUCKET\": \"${SNAPSHOT_S3_BUCKET:-}\"
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
