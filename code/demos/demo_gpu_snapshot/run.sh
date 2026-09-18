#!/bin/bash
# demo_gpu_snapshot — REAL slime KernelBench GPU trainer WITH application-level sandbox snapshot/resume.
# Runs INSIDE the persistent container (agentcore_p4). Self-contained: owns its data/ + ckpt/ +
# snapshots/ and imports its OWN vendored agentcore_slime_demo copy (PYTHONPATH). Same decoupled/async
# shape as demo_gpu_single (KernelBench torch/CUDA checker on a REMOTE g6 sandbox Runtime #2), PLUS:
#
# APPLICATION-LEVEL SNAPSHOT: AgentCore exposes NO VM/EC2 snapshot (you don't own the managed
# instance; an EBS snapshot is disk-only, not the paused VRAM/CUDA context; a recycled GPU instance
# comes back blank — docs/gpu-snapshot.md). The only feasible snapshot is application-level: serialize
# the sandbox WORKSPACE to a store you own and restore it onto a fresh session. This trainer wires
# that into the LIVE rollout: every episode opens a fresh sandbox session (so every episode boundary
# is effectively a recycle); the harness RESTORES each task's workspace from the store at episode
# ENTER and SNAPSHOTS it back at episode EXIT — the quiescent point, right before the trainer's
# weight-sync barrier. The store, not the session, is the durable source of truth. See
# agentcore_slime_demo/app_snapshot.py + snapshot_store.py; snapshot_demo.py is the seconds-long
# OFFLINE animation of the same mechanism.
#
# NOTE step count: slime resumes at loaded_iter+1; a fresh torch_dist ckpt is iter 0 => the loop runs
# range(1, NR) = (NR-1) real training steps. NR=11 => 10 train + 1 eval (SMOKE; no more needed).
#
# HARDWARE STATUS: the snapshot glue is offline-validated (import + ast + the offline snapshot_demo.py
# sim); the end-to-end REAL trainer path (g6 sandbox + store round-trip during a live run) is
# UNVERIFIED on hardware.
set -euo pipefail

SANDBOX="${SANDBOX:-agentcore-runtime}"   # the sandbox is its OWN GPU-backed AgentCore Runtime #2
                                          # (sandbox_server_agent.py on a g6). The KernelBench torch/CUDA
                                          # checker runs THERE (remote), so no local env GPUs are used.
# ---- Task set: the REAL KernelBench (ScalingIntelligence/KernelBench) ----------------------------
# Train = level_1 (100 single-op CUDA problems); held-out eval = 20 level_2 fusion problems (disjoint).
# Save-only-at-end holds regardless of dataset size (explicit --num-rollout => no epoch-boundary save;
# huge SAVE_INTERVAL => no periodic save; the ONLY checkpoint is slime's forced final-rollout one).
RB="${RB:-4}"          # prompts per rollout step
NS="${NS:-8}"          # GRPO group size (samples per prompt)
NR="${NR:-11}"         # num-rollout; real training steps = NR-1 => 10 train (SMOKE). Keep NR*RB <= 100.
TEMP="${TEMP:-1.0}"    # rollout sampling temperature (higher => more intra-group diversity)
GBS=$(( RB * NS ))     # global batch = all samples, 1 optimizer step per rollout
EVAL_INTERVAL="${EVAL_INTERVAL:-10}"  # run held-out eval every N training steps (0 => off)
EVAL_NS="${EVAL_NS:-2}"              # samples per eval prompt (eval solve-rate resolution)
SAVE_INTERVAL="${SAVE_INTERVAL:-1000000}"

# ---- Application-snapshot knobs (this demo's distinctive property) -------------------------------
# SNAPSHOT_EVERY=0 disables snapshot/resume entirely => this run behaves EXACTLY like demo_gpu_single.
SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-1}"            # snapshot cadence in episodes (0 = off)
SNAPSHOT_DIR="${SNAPSHOT_DIR:-/work/code/demos/demo_gpu_snapshot/snapshots}"  # LocalSnapshotStore root
SNAPSHOT_PATHS="${SNAPSHOT_PATHS:-progress.json,work.log,solution.py}"        # workspace files to keep
RECYCLE_AFTER="${RECYCLE_AFTER:-0}"              # >0 => log a simulated instance recycle after N episodes
# SNAPSHOT_S3_BUCKET (optional): if set, use S3SnapshotStore (production; no mount, AZ-agnostic).

SLIME=/root/slime
DEMO_DIR=/work/code/demos/demo_gpu_snapshot
# Base Megatron checkpoint is SHARED (7.5 GB) — referenced, not copied. Override via $CKPT_BASE.
CKPT="${CKPT_BASE:-/work/code/p4_slime_train/ckpt/Qwen3-4B_torch_dist}"
HF=/root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c

# ---- GPU split + where the sandbox Runtime lives -------------------------------------------------
# ALL 8 GPUs to slime (4 train + 4 infer); the KernelBench checker's GPU is REMOTE (the sandbox
# Runtime #2 g6). For an on-box GPU run with no remote g6, use demo_gpu_local instead.
ACTOR_GPUS="${ACTOR_GPUS:-4}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-4}"                       # 4 train + 4 infer = 8; checker GPU is remote
SLIME_NUM_GPUS="${SLIME_NUM_GPUS:-8}"
SLIME_CUDA_VISIBLE="${SLIME_CUDA_VISIBLE:-}"
AGENTCORE_LOCAL_ENDPOINT="${AGENTCORE_LOCAL_ENDPOINT:-}"
AGENTCORE_SANDBOX_RUNTIME_ARN="${AGENTCORE_SANDBOX_RUNTIME_ARN:?set to the GPU sandbox Runtime #2 ARN — for an on-box GPU run with no remote g6, use demo_gpu_local instead}"
DATA="${DATA:-$DEMO_DIR/data/kernelbench_train.jsonl}"
EVAL_DATA="${EVAL_DATA:-$DEMO_DIR/data/kernelbench_eval.jsonl}"
EVAL_NAME="${EVAL_NAME:-kernelbench}"
SAVE="${SAVE:-$DEMO_DIR/ckpt/Qwen3-4B_kernelbench_gpu_snapshot_async}"

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
   # ---- Checkpoint-save deadlock fix (two flags that MUST go together; see demo_gpu_single) ------
   --async-save
   --custom-megatron-init-path agentcore_slime_demo.megatron_init.enable_persistent_async_ckpt
)

# Held-out EVAL every EVAL_INTERVAL steps (20 KernelBench level_2 fusions, disjoint from train).
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
    \"AGENTCORE_SANDBOX_RUNTIME_ARN\": \"${AGENTCORE_SANDBOX_RUNTIME_ARN}\",
    \"AGENTCORE_LOCAL_ENDPOINT\": \"${AGENTCORE_LOCAL_ENDPOINT}\",
    \"SNAPSHOT_EVERY\": \"${SNAPSHOT_EVERY}\",
    \"SNAPSHOT_DIR\": \"${SNAPSHOT_DIR}\",
    \"SNAPSHOT_PATHS\": \"${SNAPSHOT_PATHS}\",
    \"RECYCLE_AFTER\": \"${RECYCLE_AFTER}\",
    \"SNAPSHOT_S3_BUCKET\": \"${SNAPSHOT_S3_BUCKET:-}\"
  }
}"

cd "$SLIME"
# ASYNC driver + DECOUPLED placement: train_async.py, NO --colocate. ABSOLUTE path (the ray job
# entrypoint's cwd is the ray-head cwd (/work), not this script's `cd "$SLIME"`).
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
