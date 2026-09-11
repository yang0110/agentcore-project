#!/bin/bash
# Demo B2 — KernelBench, GPU sandbox, REAL CLAUDE CODE harness, the TWO-RUNTIME target.
# Runner (this dir owns data/ + ckpt/; agentcore_slime_demo is the SHARED lib from /work/code). Runs INSIDE the
# persistent slime container (agentcore_p4), async/decoupled (train_async.py, NO --colocate).
#
# Demo B2 == Demo B3 (two-Runtime target) but with the REAL Claude Code CLI as the harness instead
# of our mini agent. Same task as Demo B (KernelBench CUDA ops; the checker imports torch and runs
# the candidate kernel on cuda). The Qwen3-4B policy is still trained by slime; Claude Code talks to
# it via slime's AnthropicAdapter proxy (ANTHROPIC_BASE_URL -> proxy -> SGLang).
#
# DEPLOYMENT (the two-Runtime target — docs/harness-sandbox-decoupling.md §7): the real `claude -p`
# runs on its OWN AgentCore Runtime #1 (harness_runtime_agent.py, HARNESS_KIND=claude; built-in
# tools DISABLED, its only tools are the mcp__sandbox__* bridge). That harness drives a SEPARATE
# GPU-backed sandbox Runtime #2 (sandbox_server_agent.py on a g6 capacity provider) where the
# KernelBench torch/CUDA checker runs. Deploy the claude harness with
# `code/deploy/deploy_two_runtime.py harness-claude` and reuse the B3 GPU sandbox Runtime #2 (from
# `deploy_two_runtime.py all`). This demo drives agentcore_slime_demo.agentcore_runtime_generate.
# generate (SAME hook as B3), so the harness Runtime #1 dials the AnthropicAdapter proxy back over
# the private VPC (SG-gated, no public port). The env-checker GPU is REMOTE (Runtime #2), NOT local:
# no local GpuPool here, this box runs just train + inference. (The old in-sandbox-CLI shortcut that
# used claude_code_generate.generate has been replaced by this two-Runtime path.)
#
# NOTE step count: slime resumes at loaded_iter+1; a fresh torch_dist ckpt is iter 0 => the loop
# runs range(1, NR) = (NR-1) real training steps. NR=20 => 10 train (SMOKE; no more needed).
set -euo pipefail

SANDBOX="${SANDBOX:-agentcore-runtime}"   # sandbox = its OWN custom g6 AgentCore Runtime (hosts the CLI)
# ---- Task set: the REAL KernelBench (ScalingIntelligence/KernelBench) ----------------------------
# Train = level_1 (100 single-op CUDA problems); held-out eval = 20 level_2 fusion problems (disjoint).
# Build with build_dataset.py --dataset kernelbench --level level_1 --limit 100 (train) and
# --level level_2 --limit 20 (eval). This replaced the old 3-op smoke stub, so real RB/NR now work.
# Save-only-at-end holds while dataset_size (100) >= NR * RB: slime force-saves at every epoch boundary
# (step % (dataset_size/RB) == 0), so with RB=4 the epoch is 25 steps and NR=20 (19 real) never crosses it.
RB="${RB:-4}"          # prompts per rollout step
NS="${NS:-8}"          # GRPO group size (samples per prompt)
NR="${NR:-11}"         # num-rollout; real training steps = NR-1 => 10 train (SMOKE; no more needed). Keep NR*RB <= 100.
TEMP="${TEMP:-1.0}"    # rollout sampling temperature
GBS=$(( RB * NS ))     # global batch = all samples, 1 optimizer step per rollout
EVAL_INTERVAL="${EVAL_INTERVAL:-10}"   # run held-out eval every N training steps (0 => off)
EVAL_NS="${EVAL_NS:-2}"               # samples per eval prompt
SAVE_INTERVAL="${SAVE_INTERVAL:-1000000}"  # save-only-at-end (final rollout force-saves anyway)

SLIME=/root/slime
DEMO_DIR=/work/code/demos/demo_b2
CKPT="${CKPT_BASE:-/work/code/p4_slime_train/ckpt/Qwen3-4B_torch_dist}"   # shared base ckpt (referenced, not copied)
HF=/root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c

DATA="${DATA:-$DEMO_DIR/data/kernelbench_train.jsonl}"
EVAL_DATA="${EVAL_DATA:-$DEMO_DIR/data/kernelbench_eval.jsonl}"
EVAL_NAME="${EVAL_NAME:-kernelbench}"
SAVE="${SAVE:-$DEMO_DIR/ckpt/Qwen3-4B_kernelbench_cc_async}"

# GPU split: 4 train + 4 inference + 0 env. The KernelBench checker's GPU is REMOTE (on the sandbox
# g6 Runtime), so no local GpuPool and no reserved env GPUs here.
ACTOR_GPUS="${ACTOR_GPUS:-4}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-4}"

# ---- Two-Runtime target env (REQUIRED for a real run) ----
# The real `claude -p` runs on its OWN AgentCore Runtime #1 and dials the AnthropicAdapter proxy back
# over the private VPC; the sandbox is the GPU-backed Runtime #2 where the KernelBench checker runs:
#   * AGENTCORE_HARNESS_RUNTIME_ARN — Runtime #1 (harness_runtime_agent.py, HARNESS_KIND=claude);
#                                     deploy with `deploy_two_runtime.py harness-claude`.
#   * AGENTCORE_SANDBOX_RUNTIME_ARN — Runtime #2 (sandbox_server_agent.py, GPU g6; SAME as B3's).
#   * ADAPTER_PUBLIC_HOST — this box's PRIVATE IP (e.g. PRIVATE-IP-f99864bc) that Runtime #1 dials back to
#   * ADAPTER_PORT — pinned so a single narrow security-group ingress rule stays stable (no public port)
AGENTCORE_HARNESS_RUNTIME_ARN="${AGENTCORE_HARNESS_RUNTIME_ARN:?set to the CLAUDE harness Runtime #1 ARN (deploy_two_runtime.py harness-claude)}"
AGENTCORE_SANDBOX_RUNTIME_ARN="${AGENTCORE_SANDBOX_RUNTIME_ARN:?set to the GPU sandbox Runtime #2 ARN (deploy_two_runtime.py all)}"
ADAPTER_PUBLIC_HOST="${ADAPTER_PUBLIC_HOST:?set to this box PRIVATE IP that harness Runtime 1 dials back to}"
ADAPTER_PORT="${ADAPTER_PORT:-18021}"    # pinned; harness Runtime #1 dials back ONLY to this port (one narrow SG rule)
HARNESS_KIND="${HARNESS_KIND:-claude}"   # B2 => the real Claude Code CLI on Runtime #1 (vs B3's mini)
# Per-episode agentic-turn cap. Raised 3 -> 10 for B2: the first run scored zero because the weak
# policy wasted turns on hallucinated `Explore`/`Task` calls; with those tools now disabled + a
# steering system prompt (harness_runtime_agent.py), give it room to write the kernel then compile/run
# it. Read trainer-side by generate.py from the mounted /work/code, so it takes effect WITHOUT a
# Runtime #1 redeploy (the harness-image changes DO need a redeploy).
EPISODE_MAX_TURNS="${EPISODE_MAX_TURNS:-10}"
# Runtime-invoke transient-retry budget (agentcore_runtime_sandbox._retry_invoke). a2's rerun DIED
# at step ~2 when its sandbox Runtime returned a ~3-min burst of 502 / "error when starting the
# runtime"; at the default 8 attempts (~150s) the budget was exhausted and the exception propagated
# out of generate() and killed the WHOLE job over ONE bad episode. B2 is MORE exposed (two runtimes
# per episode: harness #1 + sandbox #2, ~5 invokes each), so raise it too: 20 attempts ≈ 10 min.
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
   --save-interval "$SAVE_INTERVAL"   # save-only-at-end (final rollout force-saves anyway)
   # ---- Checkpoint save: SYNCHRONOUS for B2 (do NOT enable --async-save here) ---------------
   # B2 saves exactly ONCE, at the end (save-only-at-end), so async buys nothing — and BOTH async
   # backends FAIL under B2's PP=2 config (B2 is the real-Claude-Code case, so PP=2 is a required
   # dynamic-batch correctness fix, see PERF_ARGS below), exactly like Demo A2:
   #   * TemporalAsyncCaller (default): forks a writer off the CUDA/NCCL main process -> inherits a
   #     locked mutex -> deadlocks after .distcp are written (the original bug).
   #   * PersistentAsyncCaller (spawn, via megatron_init.enable_persistent_async_ckpt): the spawn
   #     worker receives the state dict over a queue. At PP=1 (demos a3/b3/b4) those tensors are
   #     staged to host and this works; at PP=2 they are STILL GPU-resident, so the worker's
   #     rebuild_cuda_tensor -> _new_shared_cuda uses CUDA IPC via the pidfd_getfd(2) syscall, which
   #     is gated behind CAP_SYS_PTRACE in the default docker seccomp profile. The agentcore_p4
   #     container has no such cap -> EPERM ("pidfd_getfd: Operation not permitted"), the worker
   #     dies, and only common.pt lands (0 of the ~53 GB .distcp shards). Observed on A2 2026-09-08.
   # Sync save has NEITHER failure mode: no fork (no deadlock), no spawn worker (no CUDA IPC / no
   # pidfd_getfd) — rank 0 does the D2H copy and writes .distcp inline. It blocks ~a few min for the
   # final write, which is irrelevant here (training + eval are already done). b3/b4 keep --async-save
   # because PP=1 makes the spawn path safe there. See demo_a2/run.sh and docs/demos-guide.md §7.
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
   # Raised 1024 -> 2048 (same reason as a2): the real Claude Code CLI is verbose, so at 1024 the
   # final mcp__sandbox__* tool call was truncated mid-stream. With the steering fix now pointing the
   # model at the sandbox tools, give the kernel-write + compile/run calls room to land. max-tokens-
   # per-gpu (9216) comfortably exceeds this; PP=2 dynamic-batch handling is unaffected.
   --rollout-max-response-len 2048
   --rollout-temperature "$TEMP"
   --global-batch-size "$GBS"
   # THE KEY DIFFERENCE from Demo B: the two-Runtime target — the real Claude Code CLI on its OWN
   # AgentCore Runtime #1, tools + checker bound to a separate GPU sandbox Runtime #2. SAME hook as
   # B3 (agentcore_runtime_generate.generate); HARNESS_KIND=claude selects the CLI over the mini agent.
   --custom-generate-function-path agentcore_slime_demo.agentcore_runtime_generate.generate
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
   # and it asserts "could only produce N mbs ... need N+1" (killed a2 at step 3, intermittently).
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
   # failed "could only produce 139 mbs ... need 140" (a sequence exceeded 4096). The actor-GPU OOM in
   # ppo_utils vocab_parallel_softmax is FRAGMENTATION, fixed by
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
   # Cap rollout concurrency (Semaphore(sglang_server_concurrency * num_engines)). Each episode
   # spins a Claude Code agent + a sandbox, so keep this small.
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
# Give ray ALL 8 physical GPUs. slime reserves (ACTOR_GPUS + ROLLOUT_GPUS) = 8; the KernelBench
# checker's GPU is on the remote sandbox Runtime #2, so no local GPUs are held back for env work.
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
    \"AGENTCORE_HARNESS_RUNTIME_ARN\": \"${AGENTCORE_HARNESS_RUNTIME_ARN}\",
    \"AGENTCORE_SANDBOX_RUNTIME_ARN\": \"${AGENTCORE_SANDBOX_RUNTIME_ARN}\",
    \"ADAPTER_PUBLIC_HOST\": \"${ADAPTER_PUBLIC_HOST}\",
    \"ADAPTER_PORT\": \"${ADAPTER_PORT}\",
    \"HARNESS_KIND\": \"${HARNESS_KIND}\",
    \"INVOKE_MAX_RETRIES\": \"${INVOKE_MAX_RETRIES}\",
    \"EPISODE_MAX_TURNS\": \"${EPISODE_MAX_TURNS:-10}\"
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
