#!/bin/bash
# Demo A2 — MBPP (CPU sandbox) with the REAL CLAUDE CODE agent harness, ASYNC/decoupled slime.
# Runner: this dir owns its data/ + ckpt/; agentcore_slime_demo is the SHARED lib from /work/code (PYTHONPATH).
# Runs INSIDE the persistent container (agentcore_p4). In-container path: /work/code/demos/demo_a2.
#
# Demo A2 == Demo A + Claude Code:
#   Same task (MBPP), same sandbox family (CPU microVM / Code Interpreter), same slime training
#   path and same Qwen3-4B policy. The ONE difference vs Demo A is the ROLLOUT HARNESS: instead
#   of our lightweight run_episode loop (agentcore_slime_demo.generate.generate), the agent loop
#   is Claude Code's OWN CLI running inside the sandbox
#   (agentcore_slime_demo.claude_code_generate.generate). Claude Code's "Anthropic" traffic is
#   pointed at slime's AnthropicAdapter (ANTHROPIC_BASE_URL -> adapter -> SGLang policy), so every
#   completion is served by the policy under training and its tokens are captured for GRPO.
#   Agent Lightning's "train with real harnesses, zero changes": the ONLY edit to slime's shipped
#   coding_agent_rl generate is swapping E2BSandbox -> AgentCoreSandbox.
#
# THIS PATH DOES NOT RUN OFFLINE. It requires:
#   * a CUSTOM AgentCore Runtime SANDBOX (multi-user: useradd/setsid/npm + Node 22 + claude baked
#     in) — build+deploy it with code/deploy/deploy_cli_sandbox.py (from demo_a2/Dockerfile), then
#     export the printed AGENTCORE_SANDBOX_RUNTIME_ARN. Node/claude are BAKED into that image, so
#     install_cli is skipped (no host tarballs needed) — see claude_code_generate._boot_sandbox.
#   * ADAPTER_PUBLIC_HOST — this box's PRIVATE IP; the Claude Code CLI running INSIDE the sandbox
#     dials the AnthropicAdapter proxy back over the VPC (SG-gated, no public port).
# The trainer-side ClaudeCodeHarness runs IN-PROCESS here (claude_code_generate.generate); only the
# sandbox is remote. So A2 needs ONE new Runtime (the sandbox), unlike A3's two-Runtime layout.
#
# GPU split (same as Demo A): 4 train + 4 inference + 0 env (CPU envs => no GpuPool).
# Tunables: RB (rollout-batch-size), NS (n-samples-per-prompt), NR (num-rollout),
#           TEMP (sampling temp), CONC (sglang-server-concurrency).
#
# NOTE step count: slime resumes at loaded_iter+1; a fresh torch_dist ckpt is iter 0 => the loop
# runs range(1, NR) = (NR-1) real training steps. NR=2 => 1 step.
set -euo pipefail

SANDBOX="${SANDBOX:-agentcore-runtime}"  # sandbox = its OWN custom AgentCore Runtime (hosts the CLI)
RB="${RB:-4}"          # prompts per rollout step
NS="${NS:-8}"          # GRPO group size (samples per prompt)
NR="${NR:-11}"        # num-rollout; real training steps = NR-1 => 10 train + 1 eval (SMOKE; no more needed).
                       # Keep the train set >= NR*RB prompts so one epoch spans the whole run and
                       # the ONLY checkpoint is the final one (slime force-saves at epoch
                       # boundaries too). The MBPP train pool here is built at 500 prompts
                       # (>= 100*4), so NR up to 125 stays save-only-at-end. Override per run.
TEMP="${TEMP:-1.0}"    # rollout sampling temperature (higher => more intra-group diversity)
GBS=$(( RB * NS ))     # global batch = all samples, 1 optimizer step per rollout
EVAL_INTERVAL="${EVAL_INTERVAL:-10}"  # run held-out eval every N training steps (0 => off)
EVAL_NS="${EVAL_NS:-2}"               # samples per eval prompt
# Eval sampling temperature. The prior run reused TEMP=1.0 for eval, which (with the verbose real
# Claude Code CLI) made completions ramble and saturate the response cap → the final tool call was
# truncated → no valid solution → eval/mbpp collapsed to 0.085. A low eval temp keeps eval crisp.
EVAL_TEMP="${EVAL_TEMP:-0.2}"
# Checkpoint policy (disk-saving): no mid-training snapshots. slime ALWAYS force-saves at the final
# rollout AND at every EPOCH boundary (step % (dataset_size/RB) == 0). To get EXACTLY ONE
# checkpoint (at the end), the train set must span one epoch: dataset_size >= NR * RB. The MBPP
# train pool is built at 500 prompts (=> 125 steps/epoch >= NR=100), so this holds.
SAVE_INTERVAL="${SAVE_INTERVAL:-1000000}"

SLIME=/root/slime
DEMO_DIR=/work/code/demos/demo_a2
CKPT_BASE="${CKPT_BASE:-/work/code/p4_slime_train/ckpt/Qwen3-4B_torch_dist}"  # shared base ckpt (referenced, not copied)
HF=/root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c

# Demo A2: 4 train + 4 inference + 0 env. CPU envs => no GpuPool.
ACTOR_GPUS="${ACTOR_GPUS:-4}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-4}"
DATA="${DATA:-$DEMO_DIR/data/mbpp_train.jsonl}"
EVAL_DATA="${EVAL_DATA:-$DEMO_DIR/data/mbpp_eval.jsonl}"
EVAL_NAME="${EVAL_NAME:-mbpp}"
SAVE="${SAVE:-$DEMO_DIR/ckpt/Qwen3-4B_mbpp_claudecode_async}"

# ---- Claude Code harness env (REQUIRED for a real run) ----
# The rollout is Claude Code running inside the custom sandbox Runtime, talking back to slime's adapter.
AGENTCORE_SANDBOX_RUNTIME_ARN="${AGENTCORE_SANDBOX_RUNTIME_ARN:?set to the A2 sandbox Runtime ARN (deploy_cli_sandbox.py a2)}"
ADAPTER_PUBLIC_HOST="${ADAPTER_PUBLIC_HOST:?set to this box PRIVATE IP that the in-sandbox CLI dials back to}"
ADAPTER_PORT="${ADAPTER_PORT:-18021}"                # pinned; the in-sandbox CLI dials back ONLY to this port (one narrow SG rule)
ANTHROPIC_MODEL="${ANTHROPIC_MODEL:-}"               # model name Claude Code requests (served by policy)
# Node 22 + claude are BAKED into the sandbox image, so install_cli is skipped — no host tarballs needed.
# Sandbox-invoke transient-retry budget (agentcore_runtime_sandbox._retry_invoke). The first rerun
# DIED at step ~2 when the a2 sandbox Runtime returned a burst of 502 / "error when starting the
# runtime" for ~3 min; at the default 8 attempts (~150s) the retry budget was exhausted and the
# exception propagated out of generate() and killed the WHOLE 17h job over ONE bad episode. Raise it
# to ride out a multi-minute runtime-start blip: 20 attempts at the 30s cap ≈ 10 min of patience.
INVOKE_MAX_RETRIES="${INVOKE_MAX_RETRIES:-20}"

# clean any prior ray/sglang from a previous run
pkill -9 sglang 2>/dev/null || true; ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true; sleep 2

set -ex
export PYTHONUNBUFFERED=1
source "$SLIME/scripts/models/qwen3-4B.sh"

CKPT_ARGS=(
   --hf-checkpoint "$HF"
   --ref-load "$CKPT_BASE"
   --load "$CKPT_BASE"
   --save "$SAVE"
   --save-interval "$SAVE_INTERVAL"   # save-only-at-end (see SAVE_INTERVAL note above)
   # ---- Checkpoint save: SYNCHRONOUS for A2 (do NOT enable --async-save here) ---------------
   # A2 saves exactly ONCE, at the end (save-only-at-end), so async buys nothing — and both async
   # backends FAIL under A2's PP=2 config:
   #   * TemporalAsyncCaller (default): forks a writer off the CUDA/NCCL main process -> inherits a
   #     locked mutex -> deadlocks after .distcp are written (the original bug).
   #   * PersistentAsyncCaller (spawn, via megatron_init.enable_persistent_async_ckpt): the spawn
   #     worker receives the state dict over a queue. At PP=1 (demos a3/b3) those tensors are staged
   #     to host and this works; at PP=2 they are STILL GPU-resident, so the worker's
   #     rebuild_cuda_tensor -> _new_shared_cuda uses CUDA IPC via the pidfd_getfd(2) syscall, which
   #     is gated behind CAP_SYS_PTRACE in the default docker seccomp profile. The trainer container
   #     (agentcore_p4) has no such cap -> EPERM ("pidfd_getfd: Operation not permitted"), the worker
   #     dies, and only common.pt lands (0 of the 53 GB .distcp shards written). Observed 2026-09-08:
   #     A2 trained 10 steps + eval fine, then this killed the final save.
   # Sync save has NEITHER failure mode: no fork (no deadlock), no spawn worker (no CUDA IPC / no
   # pidfd_getfd) — rank 0 does the D2H copy and writes .distcp inline. It blocks ~a few min for the
   # final 53 GB write, which is irrelevant here (training + eval are already done). The a3/b3 demos
   # keep --async-save because PP=1 makes the spawn path safe there. See docs/demos-guide.md §7.
)

# Held-out EVAL every EVAL_INTERVAL steps, routed through the SAME Claude Code generate with
# evaluation=True (one full agentic episode per eval prompt, no gradient). eval metrics logged as
# eval/<name>; metrics.py writes them to a separate eval CSV.
EVAL_ARGS=()
if [ "$EVAL_INTERVAL" != "0" ] && [ -f "$EVAL_DATA" ]; then
   EVAL_ARGS=(
      --eval-interval "$EVAL_INTERVAL"
      --eval-prompt-data "$EVAL_NAME" "$EVAL_DATA"
      --n-samples-per-eval-prompt "$EVAL_NS"
      --eval-temperature "$EVAL_TEMP"
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
   # Raised 1024 -> 2048: the real Claude Code CLI is verbose (large system prompt, <think> traces,
   # tool-call scaffolding), so at 1024 the final tool call/solution was truncated mid-stream — eval
   # completions saturated the cap (median response len = 1024) and eval/mbpp fell to 0.085. 2048
   # gives the solution room to land. (max-tokens-per-gpu comfortably exceeds this.)
   --rollout-max-response-len 2048
   --rollout-temperature "$TEMP"
   --global-batch-size "$GBS"
   # THE Demo-A2 difference: the real Claude Code harness, not our lightweight run_episode.
   --custom-generate-function-path agentcore_slime_demo.claude_code_generate.generate
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   # PP=2 (not 1) is REQUIRED here, and it is a correctness fix, not a perf knob. slime's dynamic-batch
   # DP scheduler (dp_schedule.build_dp_schedule) forces the per-step microbatch count to be a multiple
   # of align_to = dp_size (= actor_gpus / (TP*PP) = 4/(2*PP)). At PP=1 that is dp_size=2, so the mbs
   # count must be EVEN. Real Claude Code trajectories are long, so at --max-tokens-per-gpu 9216 almost
   # every sample lands in its OWN single-sample microbatch, and a single-sample bin CANNOT be split;
   # when the resulting bin count comes out ODD, expand_bins_by_splitting can't reach the even target
   # and it asserts "could only produce N mbs ... need N+1" (killed the run at step 3, intermittently).
   # PP=2 makes dp_size = 4/(2*2) = 1 => align_to=1 => ANY microbatch count is valid (bug gone), keeps
   # all 4 actor GPUs busy, keeps TP=2 (matches rollout TP, no weight-resharding mismatch), and splits
   # the 36 layers 18/stage so per-GPU memory DROPS. Do NOT set it back to 1 for the real-CLI demos.
   --pipeline-model-parallel-size 2
   --context-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   # Keep this >= the LONGEST training sequence: real Claude Code trajectories are long multi-turn
   # sequences and the dynamic batcher CANNOT split a single sample below its own length — at 4096 it
   # failed "could only produce 139 mbs ... need 140" (a sequence exceeded 4096). The OOM we first hit
   # here at 9216 was FRAGMENTATION (7.39 GiB reserved-but-unallocated), fixed by
   # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (env below), NOT by lowering the token budget.
   --max-tokens-per-gpu "${MAX_TOK_PER_GPU:-9216}"
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
   # Cap rollout concurrency. Each episode boots a sandbox + a Claude Code CLI session, so keep small.
   --sglang-server-concurrency "${CONC:-4}"
   # REQUIRED: the policy is Qwen3-4B, which emits Hermes-style <tool_call>{json}</tool_call>. The
   # Claude Code CLI drives its Bash/Edit tools off real Anthropic tool_use blocks, which slime's
   # AnthropicAdapter can only produce from that raw output via SGLang's qwen25 parser. Left unset,
   # the CLI sees no tool calls, never writes solution.py, and reward is 0 for every rollout (the
   # same all-zeros failure demo_a3 hit on 2026-09-08).
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
# Give ray ALL 8 physical GPUs. slime's placement reserves (ACTOR_GPUS + ROLLOUT_GPUS) of them.
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
    \"AGENTCORE_GPU_GATE\": \"off\",
    \"AGENTCORE_SANDBOX_RUNTIME_ARN\": \"${AGENTCORE_SANDBOX_RUNTIME_ARN}\",
    \"ADAPTER_PUBLIC_HOST\": \"${ADAPTER_PUBLIC_HOST}\",
    \"ADAPTER_PORT\": \"${ADAPTER_PORT}\",
    \"ANTHROPIC_MODEL\": \"${ANTHROPIC_MODEL}\",
    \"INVOKE_MAX_RETRIES\": \"${INVOKE_MAX_RETRIES}\",
    \"SLIME_AGENT_CC_EXTRA_ARGS\": \"--max-turns ${EPISODE_MAX_TURNS:-3}\"
  }
}"

cd "$SLIME"
# ASYNC driver + DECOUPLED placement: train_async.py, NO --colocate. Absolute path required (the
# ray job entrypoint's cwd is the ray-head cwd (/work), not this script's `cd "$SLIME"`).
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
