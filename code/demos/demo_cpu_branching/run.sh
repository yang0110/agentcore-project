#!/bin/bash
# demo_cpu_branching — ONE EPISODE → MANY TRAINING SAMPLES ("fan-out"). The REAL Claude Code CLI is
# the harness (co-located with slime, wired to slime's AnthropicAdapter); its TOOLS run in a REMOTE
# CPU AgentCore Runtime microVM. The demo deliberately provokes the two harness behaviors that split
# one rollout into several trainable branches — sub-agents (Task tool) and context summarization /
# compaction — so a single episode yields a LIST of Samples, and we reconcile that list with a
# rollout-level GRPO advantage + rollout-level token-mean loss. Runs INSIDE agentcore_p4 (slimerl/slime).
#
# READ THIS FIRST — how the fan-out is wired end-to-end (see also README.md and the two source files):
#   * agentcore_slime_demo/claude_code_generate.py — the custom generate hook. Runs the CLI in the
#     sandbox against the slime-served Anthropic endpoint, then `finish_session` linearizes the
#     per-session trajectory TREE into one Sample per leaf, all sharing rollout_id + the episode's
#     outcome reward. Returns list[Sample].
#   * agentcore_slime_demo/rollout_advantage.py — the custom advantage. GRPO baseline over DISTINCT
#     ROLLOUTS (episodes), not over fanned-out leaves. Broadcast per-episode advantage to every leaf.
#
# THREE FLAGS MAKE THE RECONCILIATION CORRECT (all set below; do not drop any):
#   1. --custom-advantage-function-path ...rollout_advantage.rollout_level_grpo_advantage
#        The built-in GRPO path baselines over SAMPLES; under fan-out its _post_process_rewards guard
#        (rewards.shape[-1] == n_samples_per_prompt*rollout_batch_size) FAILS and it collapses the
#        whole batch into one wrong group. So we compute the baseline ourselves, per episode.
#   2. --disable-rewards-normalization
#        Keep RAW rewards flowing into the actor; the custom advantage does the (only) normalization.
#   3. NO --calculate-per-token-loss (left OFF on purpose)
#        Default sum_of_sample_mean reducer + shared rollout_id => slime keys the loss denominator on
#        `rollout_mask_sums` (rollout.py) = per-EPISODE token-weighted mean. So each episode counts
#        once in the loss too, matching the per-episode advantage. Passing the flag would switch to a
#        global per-token mean and let prolific (many-leaf) episodes dominate — do NOT pass it.
#
# MODE=A ONLY (enforced below). Returning list[Sample] puts slime on the nested list[list[Sample]]
# shape. train_async.py's stock 1-step-async driver tolerates it (isinstance(group[0], list) guards
# everywhere); the FULLY-ASYNC driver (generate_rollout_fully_async) CRASHES on it (its _key sort).
# So this demo forbids MODE=B.
#
# dp_size == 1 REQUIRED. The custom advantage runs in the Megatron actor on each rank's partition and
# assumes a whole prompt-group (all its rollouts + their leaves) is co-resident on the rank. That
# holds only at data-parallel size 1. With 4 actor GPUs we get dp=1 via TP=2 * PP=2 = 4 (see
# PERF_ARGS). PP=2 breaks BOTH of Megatron's async-checkpoint backends (the a2/b2 pidfd_getfd EPERM),
# so this demo uses SYNC checkpoint save (no --async-save, no persistent-async-ckpt init hook).
#
# REMOTE-ONLY (CPU demo): SANDBOX=local is rejected. The sandbox here must host the Claude Code CLI,
# so its image BAKES node 22 + the claude CLI (deploy/Dockerfile.sandbox.cli) — the prebaked path;
# streaming those binaries through write_file corrupts them. Point AGENTCORE_SANDBOX_RUNTIME_ARN at a
# Runtime built from that image (NOT the plain CPU checker image the other CPU demos use).
#
# ADAPTER DIAL-BACK: the CLI inside the microVM must reach slime's AnthropicAdapter on THIS box.
# ADAPTER_PUBLIC_HOST (required) = a private VPC IP the sandbox can dial; ADAPTER_PORT (default 18021)
# = a PINNED port so one security-group ingress rule stays stable across the whole run.
#
# Tunables: RB (rollout-batch-size = prompts/step), NS (n-samples-per-prompt = distinct rollouts per
# prompt), NR (num-rollout; real steps = NR-1), TEMP, CONC (episode concurrency),
# BRANCHING_FORK_THRESHOLD (lower => more compaction forks), AGENT_TIME_BUDGET_SEC.
set -euo pipefail

SANDBOX="${SANDBOX:-agentcore-runtime}"   # agentcore-runtime -> remote AgentCore Runtime CLI sandbox (default);
                                          # agentcore -> Code Interpreter (dev). CPU demo => REMOTE-ONLY.
if [ "$SANDBOX" = "local" ]; then
   echo "SANDBOX=local is not supported for demo_cpu_branching (the sandbox must host the real Claude Code CLI) — use agentcore-runtime (default)." >&2
   exit 1
fi
# The remote sandbox Runtime ARN. For THIS demo it must be a CLI-capable image (node+claude baked;
# deploy/Dockerfile.sandbox.cli), NOT the plain CPU checker image other CPU demos use.
if [ "$SANDBOX" = "agentcore-runtime" ]; then
   AGENTCORE_SANDBOX_RUNTIME_ARN="${AGENTCORE_SANDBOX_RUNTIME_ARN:?set AGENTCORE_SANDBOX_RUNTIME_ARN to a CLI-capable sandbox Runtime ARN (image from deploy/Dockerfile.sandbox.cli — node 22 + claude baked in)}"
fi
# The CLI in the microVM dials the AnthropicAdapter back on this box. REQUIRED — no sane default.
ADAPTER_PUBLIC_HOST="${ADAPTER_PUBLIC_HOST:?set ADAPTER_PUBLIC_HOST to a private VPC IP of this training box that the sandbox microVM can reach}"
ADAPTER_PORT="${ADAPTER_PORT:-18021}"          # PINNED port (stable security-group ingress)
ADAPTER_BIND_HOST="${ADAPTER_BIND_HOST:-0.0.0.0}"
BRANCHING_FORK_THRESHOLD="${BRANCHING_FORK_THRESHOLD:-512}"   # lower => record more compactions as forks
AGENT_TIME_BUDGET_SEC="${AGENT_TIME_BUDGET_SEC:-900}"

RB="${RB:-4}"          # prompts per rollout step
NS="${NS:-8}"          # GRPO group size = DISTINCT ROLLOUTS per prompt (each may fan out into many leaves)
NR="${NR:-11}"         # num-rollout; real training steps = NR-1 => 10 train + 1 eval (SMOKE)
TEMP="${TEMP:-1.0}"    # rollout sampling temperature
GBS=$(( RB * NS ))     # global batch counts DISTINCT ROLLOUTS (leaves inflate the actual sample count)

# MODE=A ONLY — see the header. Fully-async (Mode B) crashes on the nested fan-out shape.
MODE="${MODE:-A}"
case "$MODE" in A|a) MODE=A;; B|b) echo "MODE=B (fully-async) is unsupported for demo_cpu_branching: its rollout worker crashes on the nested list[list[Sample]] fan-out shape. Use MODE=A." >&2; exit 1;;
   *) echo "MODE must be A for demo_cpu_branching (got '$MODE')" >&2; exit 1;; esac

EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
EVAL_NS="${EVAL_NS:-2}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000000}"   # save-only-at-end (final rollout is force-saved)

SLIME=/root/slime
DEMO_DIR=/work/code/demos/demo_cpu_branching
CKPT="${CKPT_BASE:-/work/code/p4_slime_train/ckpt/Qwen3-4B_torch_dist}"  # SHARED base Megatron ckpt
HF=/root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c

ACTOR_GPUS="${ACTOR_GPUS:-4}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-4}"

# Single-family SWE branching dataset (built by prepare_data.sh). Each row is a SWE bug-fix task;
# every episode runs the real CLI and fans out into several leaves.
DATA="${DATA:-$DEMO_DIR/data/swe_train.jsonl}"
EVAL_DATA="${EVAL_DATA:-$DEMO_DIR/data/swe_eval.jsonl}"
EVAL_NAME="${EVAL_NAME:-swe}"
SAVE="${SAVE:-$DEMO_DIR/ckpt/Qwen3-4B_branching}"
if [ ! -f "$DATA" ]; then
   echo "missing $DATA — build it first: bash $DEMO_DIR/prepare_data.sh (inside agentcore_p4)" >&2
   exit 1
fi

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
   # SYNC save: PP=2 (needed for dp_size=1) breaks BOTH Megatron async-ckpt backends
   # (pidfd_getfd EPERM under the fork caller; the spawn caller is not initialized here). So we do
   # NOT pass --async-save / the persistent-async-ckpt init hook — slime saves synchronously.
)

# Held-out EVAL uses slime's stock sync rollout (Mode A has no background worker to quiesce). Eval
# also fans out; the metric of interest is per-episode solve rate.
EVAL_ARGS=()
if [ "$EVAL_INTERVAL" != "0" ] && [ -f "$EVAL_DATA" ]; then
   EVAL_ARGS=(
      --eval-interval "$EVAL_INTERVAL"
      --eval-prompt-data "$EVAL_NAME" "$EVAL_DATA"
      --n-samples-per-eval-prompt "$EVAL_NS"
      --eval-temperature "$TEMP"
      --eval-function-path slime.rollout.sglang_rollout.generate_rollout
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
   # Per-TURN cap (the adapter clamps the CLI's max_tokens to it). A turn that hits it comes back as
   # stop_reason=max_tokens, which the CLI treats as fatal ("exceeded the output token maximum",
   # exit=1). Qwen3's <think> alone often fills 1024, hence 2048.
   --rollout-max-response-len "${MAX_RESPONSE_LEN:-2048}"
   --rollout-temperature "$TEMP"
   --global-batch-size "$GBS"
   # The real Claude Code CLI harness. Returns list[Sample] (the fan-out) per episode.
   --custom-generate-function-path agentcore_slime_demo.claude_code_generate.generate
)

# NO MODE_ARGS: MODE=A uses train_async.py's stock 1-step-async driver (the only one that tolerates
# the nested fan-out shape). Do not add --rollout-function-path here.

PERF_ARGS=(
   # TP=2 * PP=2 == 4 == ACTOR_GPUS => data-parallel size 1. The custom rollout-level advantage
   # assumes each prompt group is co-resident on the rank, which requires dp_size == 1.
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 2
   --context-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

GRPO_ARGS=(
   --advantage-estimator grpo
   # Rollout-level advantage: baseline over DISTINCT rollouts (episodes), not fanned-out leaves.
   --custom-advantage-function-path agentcore_slime_demo.rollout_advantage.rollout_level_grpo_advantage
   # Raw rewards into the actor; the custom advantage does the only normalization. WITHOUT this the
   # built-in _post_process_rewards mis-groups the inflated (fanned-out) sample count.
   --disable-rewards-normalization
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   # NOTE: --calculate-per-token-loss is intentionally OMITTED (rollout-level token-mean loss).
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
   # Episode concurrency S = CONC * num_engines (num_engines = ROLLOUT_GPUS/per-engine = 2). Each
   # episode opens its OWN remote sandbox session AND its own adapter session (multiplexed by sid).
   --sglang-server-concurrency "${CONC:-4}"
   # The A3/B3 dead-reward fix, same as terminalbench/scale_lanes: without a tool-call parser the
   # AnthropicAdapter returns Qwen3's <tool_call> text as a plain end_turn reply, the CLI treats the
   # task as done after ONE turn (exit=0, fan_out=1), never writes solution.py, and reward is 0.
   --sglang-tool-call-parser qwen25
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
# Shared box only: SLIME_CUDA_VISIBLE=2,3,4,5,6,7 SLIME_NUM_GPUS=6 pins ray to a subset (default: all 8).
[ -n "${SLIME_CUDA_VISIBLE:-}" ] && export CUDA_VISIBLE_DEVICES="$SLIME_CUDA_VISIBLE"
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${SLIME_NUM_GPUS:-8}" \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM:${DEMO_DIR}:/root/slime\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"SANDBOX\": \"${SANDBOX}\",
    \"AGENTCORE_SANDBOX_RUNTIME_ARN\": \"${AGENTCORE_SANDBOX_RUNTIME_ARN:-}\",
    \"SLIME_AGENT_CC_EXTRA_ENVS\": \"{\\\"IS_SANDBOX\\\":\\\"1\\\"}\",
    \"ADAPTER_PUBLIC_HOST\": \"${ADAPTER_PUBLIC_HOST}\",
    \"ADAPTER_PORT\": \"${ADAPTER_PORT}\",
    \"ADAPTER_BIND_HOST\": \"${ADAPTER_BIND_HOST}\",
    \"BRANCHING_FORK_THRESHOLD\": \"${BRANCHING_FORK_THRESHOLD}\",
    \"AGENT_TIME_BUDGET_SEC\": \"${AGENT_TIME_BUDGET_SEC}\",
    \"AWS_REGION\": \"${AWS_REGION:-us-west-2}\",
    \"AWS_BEARER_TOKEN_BEDROCK\": \"${AWS_BEARER_TOKEN_BEDROCK:-}\"
  }
}"

cd "$SLIME"
# ASYNC driver + DECOUPLED placement: train_async.py, NO --colocate. ABSOLUTE path (ray job cwd is
# the ray-head cwd, /work here).
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
