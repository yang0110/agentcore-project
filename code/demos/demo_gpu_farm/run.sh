#!/bin/bash
# demo_gpu_farm — KernelBench (real CUDA-kernel RL) on a Ray GPU-farm sandbox, LOCAL or on a remote g6.
# Runner (this dir owns data/ + ckpt/; agentcore_slime_demo is the vendored local copy in this dir). Runs INSIDE the
# persistent slime container (agentcore_p4), async/decoupled (train_async.py, NO --colocate) with slime's fully-async rollout driver.
#
# The task: the policy writes solution.py, a GPU checker (run_checks.py) compiles/runs it on CUDA and
# compares against a torch reference, the agent reads the failure and fixes it. The mini_claude loop runs
# INSIDE the episode's sandbox session (in_session_harness.py), next to that checker, and dials the
# trainer's per-engine AnthropicAdapter back; the trainer keeps only the harness tier (harness_pool.py:
# HARNESS_WORKERS dispatch workers, default RB x NS, least-loaded engine per episode).
#
# ONE FOLDER, TWO KNOBS (merged 2026-09-26 from demo_gpu_farm_fractional + _whole, and 2026-09-28 with
# demo_gpu_single, the remote g6 path; all ran the same harness, trainer, image and farm):
#   LOCAL_GPU=1 (default)  the farm is a LOCAL runtime container on this host's reserved GPU(s), served
#                          at http://127.0.0.1:8080; slime gets GPUs 0-5 (4 train + 2 infer).
#   LOCAL_GPU=0            the farm runs inside ONE warm g6 capacity-provider Instance (your custom GPU
#                          Runtime, AGENTCORE_SANDBOX_RUNTIME_ARN); slime gets all 8 GPUs (4 + 4).
#                          The managed Code Interpreter is CPU-only (torch.cuda.is_available() is False),
#                          which is why this needs your own Runtime; offline probe: gpu_torch_smoke.py.
#   FARM=fractional (default)  0.5-GPU Ray actors, two episodes per card (host GPU 6, or the g6's L4),
#                          each checker capped to ~half the VRAM in software (B4_PER_SANDBOX_MEM_FRACTION;
#                          L40S/L4 have no MIG). 2 slots either site.
#   FARM=whole             one WHOLE GPU per episode: host GPUs 6,7 (2 slots) or the g6's L4 (1 slot).
#                          No GPU memory/fault sharing (sidesteps the non-MIG nvcc wedge); host still shared.
#   SNAPSHOT_EVERY=1       + APPLICATION-LEVEL snapshot/resume: restore each task's workspace from a store
#                          you own at episode ENTER, snapshot it at EXIT. AgentCore has no VM/GPU snapshot
#                          (docs/gpu-snapshot.md); agentcore_slime_demo/app_snapshot.py, offline: snapshot_demo.py.
#
# ISOLATION — neither FARM mode is full isolation: all episodes share one kernel, network and host, and
# (fractional) one physical GPU. Trusted checkers only; full isolation = one session per episode
# (one g6 each). Details: README.md.
#
# AgentCore reality: 1 runtimeSessionId -> 1 instance. So this is ONE shared runtimeSessionId
# (B4_SHARED_RUNTIME_SESSION) + a per-episode sub_session, and the Ray farm INSIDE that one runtime
# leases one GPU actor (fractional or whole, per FARM) per sub_session. Remotely that keeps every
# episode on the SAME warm g6, so its cold boot is paid once per run, not once per episode.
#
# LOCAL (default) — start the local runtime container on the HOST first (NOT inside agentcore_p4), in the
# SAME mode as FARM; it reserves the farm GPU(s):
#     FARM=fractional bash 4_deploy_runtime.sh   # or: bash deploy/local_gpu_runtime.sh fractional
#     FARM=whole      bash 4_deploy_runtime.sh   # or: bash deploy/local_gpu_runtime.sh whole
# then, inside agentcore_p4:   FARM=fractional bash code/demos/demo_gpu_farm/run.sh
# REMOTE — LOCAL_GPU=0 on every step (the FARM split is baked into the Runtime at deploy):
#     LOCAL_GPU=0 FARM=fractional bash 4_deploy_runtime.sh && LOCAL_GPU=0 FARM=fractional bash 5_run.sh
#
# NOTE step count: slime resumes at loaded_iter+1; a fresh torch_dist ckpt is iter 0 => the loop
# runs range(1, NR) = (NR-1) real training steps. NR=11 => 10 train (SMOKE; no more needed).
set -euo pipefail
if [ -n "${MODE:-}" ]; then
  echo "MODE is gone: the fully-async driver is the only rollout driver. Unset MODE." >&2
  exit 1
fi

LOCAL_GPU="${LOCAL_GPU:-1}"
# ---- FARM: how the Ray farm partitions its GPUs (MUST match the farm: local_gpu_runtime.sh <mode>,
# or the B4_* env baked into the remote Runtime by 4_deploy_runtime.sh) ----------------------------
FARM="${FARM:-fractional}"
case "$LOCAL_GPU:$FARM" in
   1:fractional) _FARM_GPUS=1; _FRACTION=0.5 ;;   # host GPU 6: floor(1/0.5)=2 slots, 0.5 + 0.5
   1:whole)      _FARM_GPUS=2; _FRACTION=1.0 ;;   # host GPUs 6,7: 2 slots, one whole GPU each
   0:fractional) _FARM_GPUS=1; _FRACTION=0.5 ;;   # the g6's L4: 2 slots, 0.5 + 0.5
   0:whole)      _FARM_GPUS=1; _FRACTION=1.0 ;;   # the g6's L4: 1 slot (raise B4_FARM_GPUS on a bigger g6)
   *) echo "LOCAL_GPU must be 0 or 1 and FARM fractional or whole (got LOCAL_GPU='$LOCAL_GPU' FARM='$FARM')" >&2; exit 1 ;;
esac

SANDBOX="${SANDBOX:-agentcore-ray-farm}"  # the Ray-farm backend (RayFarmRuntimeSandbox): a per-episode
                                          # sub_session slot on the ONE farm runtime (local or the g6):
                                          # a 0.5-GPU actor (fractional) or a whole GPU (whole).
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

# ---- Application-snapshot knobs (SNAPSHOT_EVERY=1 turns on the snapshot/resume angle) -----------
# Every episode opens a fresh sandbox sub_session, so each episode boundary is effectively a recycle;
# the store, not the session, is the durable source of truth. Off by default.
SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-0}"            # snapshot cadence in episodes (0 = off)
SNAPSHOT_DIR="${SNAPSHOT_DIR:-/work/code/demos/demo_gpu_farm/snapshots}"   # LocalSnapshotStore root
SNAPSHOT_PATHS="${SNAPSHOT_PATHS:-progress.json,work.log,solution.py}"     # workspace files to keep
RECYCLE_AFTER="${RECYCLE_AFTER:-0}"              # >0 => log a simulated instance recycle after N episodes
# SNAPSHOT_S3_BUCKET (optional): if set, use S3SnapshotStore (production; no mount, AZ-agnostic).

SLIME=/root/slime
DEMO_DIR=/work/code/demos/demo_gpu_farm
CKPT="${CKPT_BASE:-/work/code/p4_slime_train/ckpt/Qwen3-4B_torch_dist}"   # shared base ckpt (referenced, not copied)
HF=/root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c

DATA="${DATA:-$DEMO_DIR/data/kernelbench_train.jsonl}"
EVAL_DATA="${EVAL_DATA:-$DEMO_DIR/data/kernelbench_eval.jsonl}"
EVAL_NAME="${EVAL_NAME:-kernelbench}"
_SITE=farm; [ "$LOCAL_GPU" = 1 ] || _SITE=remote
_SNAP=""; [ "$SNAPSHOT_EVERY" != "0" ] && _SNAP=_snapshot
SAVE="${SAVE:-$DEMO_DIR/ckpt/Qwen3-4B_kernelbench_${_SITE}_${FARM}${_SNAP}_async}"

# ---- GPU split + where the farm lives -------------------------------------------------------------
if [ "$LOCAL_GPU" = 1 ]; then
   # Reserve the farm GPU(s) for the local Ray-farm runtime container (started on the HOST by
   # local_gpu_runtime.sh $FARM) and give slime GPUs 0-5 (4 train + 2 infer). GPU 7 is idle with
   # FARM=fractional. The client reaches the farm over http://127.0.0.1:8080; no remote ARN (a
   # leftover one from a LOCAL_GPU=0 deploy in deploy/runtime.env is ignored).
   ACTOR_GPUS="${ACTOR_GPUS:-4}"
   ROLLOUT_GPUS="${ROLLOUT_GPUS:-2}"                       # 4 train + 2 infer = 6; farm GPU(s) reserved
   SLIME_NUM_GPUS="${SLIME_NUM_GPUS:-6}"
   SLIME_CUDA_VISIBLE="${SLIME_CUDA_VISIBLE:-0,1,2,3,4,5}" # keep slime OFF the reserved farm GPU(s) (6 / 6,7)
   AGENTCORE_LOCAL_ENDPOINT="${AGENTCORE_LOCAL_ENDPOINT:-http://127.0.0.1:8080}"
   AGENTCORE_SANDBOX_RUNTIME_ARN=""
else
   # ALL 8 GPUs to slime (4 train + 4 infer); the checker's GPU is the remote g6's, reached over
   # invoke_agent_runtime.
   ACTOR_GPUS="${ACTOR_GPUS:-4}"
   ROLLOUT_GPUS="${ROLLOUT_GPUS:-4}"                       # 4 train + 4 infer = 8; checker GPU is remote
   SLIME_NUM_GPUS="${SLIME_NUM_GPUS:-8}"
   SLIME_CUDA_VISIBLE="${SLIME_CUDA_VISIBLE:-}"
   AGENTCORE_LOCAL_ENDPOINT=""
   AGENTCORE_SANDBOX_RUNTIME_ARN="${AGENTCORE_SANDBOX_RUNTIME_ARN:?LOCAL_GPU=0 needs your custom GPU sandbox Runtime ARN (a g6 Instance with CUDA — LOCAL_GPU=0 bash 4_deploy_runtime.sh). The managed Code Interpreter has no GPU; for an on-box run use LOCAL_GPU=1}"
fi

# Farm math: slots = floor(B4_FARM_GPUS / B4_GPU_FRACTION) (table above). These MUST match the farm
# (local_gpu_runtime.sh $FARM, or the Runtime's deploy-time env); here they only size the open gate.
export B4_FARM_GPUS="${B4_FARM_GPUS:-$_FARM_GPUS}"
export B4_GPU_FRACTION="${B4_GPU_FRACTION:-$_FRACTION}"

# ONE shared runtimeSessionId for the WHOLE job: every episode (across every rollout worker) invokes
# the farm runtime under this id, so they all land on the SAME farm whose GPUs it splits. Minted
# once here if unset (AgentCore requires length in [33,128]: "b4farm-" + 32 hex = 39). Pin it to reuse
# a warm instance across runs.
B4_SHARED_RUNTIME_SESSION="${B4_SHARED_RUNTIME_SESSION:-b4farm-$(python3 -c 'import secrets;print(secrets.token_hex(16))')}"
# Client-side OPEN-concurrency gate (agentcore_runtime_sandbox._farm_open_gate): cap how many episodes
# hold a farm slot at once to the farm's actor count. The async rollout submits RB*NS episodes that
# each open their own sandbox, and neither --sglang-server-concurrency (gates GENERATION) nor NS caps
# `op=open`; without this gate the overflow opens block on the SERVER farm gate until the invoke times
# out (~min) and stall the step. Must EQUAL the server's B4_MAX_SANDBOXES = floor(B4_FARM_GPUS/
# B4_GPU_FRACTION).
export B4_FARM_MAX_CONCURRENT="${B4_FARM_MAX_CONCURRENT:-$(python3 -c "import os,math;print(max(1,math.floor(float(os.environ['B4_FARM_GPUS'])/float(os.environ['B4_GPU_FRACTION']))))")}"
CONC="${CONC:-$B4_FARM_MAX_CONCURRENT}"   # rollout concurrency <= farm slots; raise together with the farm
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
)
# ---- Checkpoint-save mode -----------------------------------------------------------------------
# LOCAL: SYNC save (no --async-save). The local farm container's seccomp profile blocks pidfd_getfd,
# which slime's persistent async-ckpt child uses to grab the CUDA storage fds ("RuntimeError:
# pidfd_getfd: Operation not permitted"), so the async save hangs the job even at PP=1.
# REMOTE: ASYNC persistent-ckpt save (two flags that MUST go together). Megatron's default writer forks
# off the main training process while it holds CUDA/NCCL/inductor threads and can inherit a locked mutex
# (the end-of-training save deadlocked AFTER the .distcp files were written). The safe SPAWN-based
# PersistentAsyncCaller needs --async-save AND init_persistent_async_worker(), which slime never calls,
# so agentcore_slime_demo/megatron_init.py does it via --custom-megatron-init-path.
if [ "$LOCAL_GPU" != 1 ]; then
   CKPT_ARGS+=(
      --async-save
      --custom-megatron-init-path agentcore_slime_demo.megatron_init.enable_persistent_async_ckpt
   )
fi

EVAL_ARGS=()
if [ "$EVAL_INTERVAL" != "0" ] && [ -f "$EVAL_DATA" ]; then
   EVAL_ARGS=(
      --eval-interval "$EVAL_INTERVAL"
      --eval-prompt-data "$EVAL_NAME" "$EVAL_DATA"
      --n-samples-per-eval-prompt "$EVAL_NS"
      --eval-temperature "${EVAL_TEMP:-0.2}"
      # The fully-async worker owns slime's GenerateState loop; this quiesces it for eval.
      --eval-function-path agentcore_slime_demo.mode_b_eval.generate_rollout
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
   # The fully-async driver (the only one): a background worker keeps CONC x engines groups in flight
   # across step boundaries, so the rollout never waits on a step barrier.
   --rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async
   # Harness tier + mini_claude in the sandbox session (+ the SNAPSHOT_EVERY restore/snapshot hooks).
   # The GPU packing is entirely in the sandbox backend (SANDBOX=agentcore-ray-farm ->
   # RayFarmRuntimeSandbox), which make_sandbox() selects per episode.
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
   # CONC defaults to the farm's slot count (above) so episodes don't block on its checkout gate.
   --sglang-server-concurrency "$CONC"
)
# HARNESS=mini_claude (the only harness; HARNESS=direct and HARNESS_WHERE=trainer were removed): the
# tool-calling loop runs in the sandbox session and calls slime's AnthropicAdapter (mini_claude_episode.py,
# hidden grader). It needs SGLang's qwen25 tool parser + qwen3 reasoning parser, or every tool call is
# dead text (the A3/B3 zero-reward bug).
export HARNESS="${HARNESS:-mini_claude}"
[ "$HARNESS" = mini_claude ] || { echo "HARNESS=$HARNESS is not supported (mini_claude only)" >&2; exit 1; }
case "${HARNESS_WHERE:-session}" in session) ;; *) echo "HARNESS_WHERE=$HARNESS_WHERE is not supported: the agent loop always runs in the sandbox session" >&2; exit 1 ;; esac
SGLANG_ARGS+=(--sglang-tool-call-parser qwen25 --sglang-reasoning-parser qwen3)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
# Ray gets SLIME_NUM_GPUS (6 local, 8 remote). Locally we pin CUDA_VISIBLE_DEVICES to 0-5 BEFORE
# `ray start` so ray only discovers/schedules those 6 — GPUs 6,7 stay free for the local farm container.
# (Do NOT set CUDA_VISIBLE_DEVICES in the ray job runtime-env: ray assigns it per-actor.)
[ -n "$SLIME_CUDA_VISIBLE" ] && export CUDA_VISIBLE_DEVICES="$SLIME_CUDA_VISIBLE"
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "$SLIME_NUM_GPUS" \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# HEARTBEAT (default ON, every backend; HEARTBEAT=0 is the A/B baseline): a process-wide
# SessionKeeper heartbeats every open sandbox session that goes idle past
# HEARTBEAT_PING_FRACTION*HEARTBEAT_IDLE, so a trainer stall (weight sync, a long train step) can't
# idle-reclaim it — a reclaimed session id silently comes back EMPTY
# (agentcore_slime_demo/session_keeper.py). The beat is a no-op exec, so it is harmless where there is
# no idle timeout (the local farm). Watch the [demo] heartbeat_session=… lines (0 with HEARTBEAT=0).
HEARTBEAT="${HEARTBEAT:-${HEARTBEAT_KEEP_WARM:-1}}"
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
    \"ADAPTER_PUBLIC_HOST\": \"${ADAPTER_PUBLIC_HOST:-}\",
    \"ADAPTER_PORT_BASE\": \"${ADAPTER_PORT_BASE:-18100}\",
    \"MINI_SESSION_POLL_SEC\": \"${MINI_SESSION_POLL_SEC:-25}\",
    \"HARNESS\": \"${HARNESS}\",
    \"PREOPEN\": \"${PREOPEN:-0}\",
    \"HARNESS_WORKERS\": \"${HARNESS_WORKERS:-}\",
    \"HEARTBEAT\": \"${HEARTBEAT}\",
    \"HEARTBEAT_IDLE\": \"${HEARTBEAT_IDLE}\",
    \"HEARTBEAT_PING_FRACTION\": \"${HEARTBEAT_PING_FRACTION}\",
    \"AWS_REGION\": \"${AWS_REGION:-us-west-2}\",
    \"AGENTCORE_REGION\": \"${AGENTCORE_REGION:-${AWS_REGION:-us-west-2}}\",
    \"AWS_BEARER_TOKEN_BEDROCK\": \"${AWS_BEARER_TOKEN_BEDROCK:-}\",
    \"AGENTCORE_SANDBOX_RUNTIME_ARN\": \"${AGENTCORE_SANDBOX_RUNTIME_ARN}\",
    \"AGENTCORE_LOCAL_ENDPOINT\": \"${AGENTCORE_LOCAL_ENDPOINT}\",
    \"B4_SHARED_RUNTIME_SESSION\": \"${B4_SHARED_RUNTIME_SESSION}\",
    \"INVOKE_MAX_RETRIES\": \"${INVOKE_MAX_RETRIES}\",
    \"SNAPSHOT_EVERY\": \"${SNAPSHOT_EVERY}\",
    \"SNAPSHOT_DIR\": \"${SNAPSHOT_DIR}\",
    \"SNAPSHOT_PATHS\": \"${SNAPSHOT_PATHS}\",
    \"RECYCLE_AFTER\": \"${RECYCLE_AFTER}\",
    \"SNAPSHOT_S3_BUCKET\": \"${SNAPSHOT_S3_BUCKET:-}\"
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
