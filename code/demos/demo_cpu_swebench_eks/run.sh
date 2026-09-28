#!/bin/bash
# demo_cpu_swebench_eks — REAL slime CPU trainer on a SWE-bench Verified subset, every episode ISOLATED in
# its own fresh gVisor POD on EKS. The twin of demo_cpu_swebench (AgentCore Runtime per image): same
# dataset rows, same wrapped images (same ECR repos), same in-session agent loop, same grader, same
# metrics — only the sandbox tier differs, so the two runs compare substrates and nothing else.
#
# Each episode routes to ITS instance's image (pool map EKS_SANDBOX_IMAGE_BY_TASK = {instance -> ECR
# URI@digest}, written by deploy/deploy_swebench.py to swebench/image_by_task.json; an unmapped instance is
# an error, never routed elsewhere). The pool (TerminalBenchPool, sandbox tier swapped to
# eks_pod_sandbox.EksPodSandbox) creates one pod per episode with runtimeClassName=gvisor on the
# acs/role=sandbox node group, talks to the in-image server on the pod IP (:8080), runs the MiniClaudeHarness
# loop in /testbed (LLM via the adapter under training, which the pod dials back on the trainer's private
# IP), grades with swebench.grade_swebench (hidden test_patch, FAIL_TO_PASS + PASS_TO_PASS), then deletes
# the pod. Runs INSIDE the persistent container agentcore_p4 (slimerl/slime); needs the cluster from
# `EKS_CREATE=1 bash 4_deploy_runtime.sh`.
#
# WHY A FRESH POD PER EPISODE: the agent edits the repo and may pip-install into the testbed env; grading
# must see exactly the environment SWE-bench validated, so no pod is ever reused. The pod cold start
# (schedule + image pull on a node that has not cached it + runsc boot) is hidden the same way as on
# AgentCore: opened at submit time and, via the look-ahead data source, TB_PREOPEN_BATCHES rollouts ahead.
# No heartbeat: pods are not idle-reclaimed.
#
# STOCK (Mode A) DRIVER ONLY, as in demo_cpu_swebench (the look-ahead pre-open data source feeds it).
#
# ADAPTER TOOL PARSER (the A3/B3 dead-reward fix): SGLANG_ARGS bakes in --sglang-tool-call-parser
# qwen25 --sglang-reasoning-parser qwen3. Without them the tool loop is dead and reward is structurally 0.
set -euo pipefail
case "${HARNESS_WHERE:-session}" in session) ;; *) echo "HARNESS_WHERE=$HARNESS_WHERE is not supported: the agent loop always runs in the sandbox session" >&2; exit 1 ;; esac

SANDBOX="${SANDBOX:-eks-gvisor}"   # fresh gVisor pod per episode (the pool creates it).
if [ "$SANDBOX" != "eks-gvisor" ]; then
   echo "SANDBOX=$SANDBOX is not supported here — this demo's only sandbox is eks-gvisor (AgentCore: demo_cpu_swebench)." >&2
   exit 1
fi

DEMO_DIR=/work/code/demos/demo_cpu_swebench_eks
POOL_MAP="${POOL_MAP:-$DEMO_DIR/swebench/image_by_task.json}"   # {instance -> ECR image URI}, from deploy
# The image pool map is REQUIRED: every episode routes to its instance's image through it.
if [ ! -f "$POOL_MAP" ]; then
   echo "pool map $POOL_MAP not found — run deploy/deploy_swebench.py first (writes swebench/image_by_task.json)." >&2
   exit 1
fi
EKS_SANDBOX_IMAGE_BY_TASK="$(cat "$POOL_MAP")"
EKS_CLUSTER="${EKS_CLUSTER:-acs-swe-eks}"
EKS_NAMESPACE="${EKS_NAMESPACE:-acs-swe}"

RB="${RB:-4}"          # prompts per rollout step
NS="${NS:-8}"          # GRPO group size (samples per prompt)
NR="${NR:-11}"         # num-rollout; real training steps = NR-1 => 10 train + 1 eval (SMOKE)
TEMP="${TEMP:-1.0}"
GBS=$(( RB * NS ))

# ---- isolation + pre-open + heartbeat knobs (read by TerminalBenchPool inside the rollout process) --------
HARNESS_WORKERS="${HARNESS_WORKERS:-$GBS}"    # harness tier: dispatch workers (default RB x NS = one step)
EPISODE_MAX_TURNS="${EPISODE_MAX_TURNS:-12}"  # agentic turns per SWE-bench episode (= SWE_MAX_TURNS in the data rows)
EPISODE_MAX_TOKENS="${EPISODE_MAX_TOKENS:-1024}"
MINI_AGENT_MAX_TOOL_CHARS="${MINI_AGENT_MAX_TOOL_CHARS:-2000}"  # clip each tool result fed back (bounds trajectory length)
TB_EXEC_TIMEOUT="${TB_EXEC_TIMEOUT:-180}"     # per bash-tool command timeout (test runs / pip are routine)
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-24576}"  # >= one full trajectory: 12 x (1024 gen + ~600 tool) + prompt
# THE pre-open knob: batches kept booted ahead (x RB x NS = 64 pods; 0 = off). Pods that do not fit on
# the node group stay Pending and are counted in open_ms — size EKS_NODES for RB x NS x (1 + this).
TB_PREOPEN_BATCHES="${TB_PREOPEN_BATCHES:-2}"

MODE="${MODE:-A}"
case "$MODE" in A|a) MODE=A;;
   B|b) echo "MODE=B (fully-async) unsupported here: the look-ahead pre-open data source feeds the stock driver. Use MODE=A." >&2; exit 1;;
   *) echo "MODE must be A (got '$MODE')" >&2; exit 1;; esac

EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
EVAL_NS="${EVAL_NS:-2}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000000}"

SLIME=/root/slime
CKPT="${CKPT_BASE:-/work/code/p4_slime_train/ckpt/Qwen3-4B_torch_dist}"
HF=/root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c

ACTOR_GPUS="${ACTOR_GPUS:-4}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-4}"
NENG=$(( ROLLOUT_GPUS / 2 ))
# Admit W running + W queued episodes (S = CONC*NENG) so a free worker never idles. Pre-opening no longer
# needs a deep queue: the look-ahead data source announces upcoming episodes, not the admission gate.
CONC="${CONC:-$(( (2 * HARNESS_WORKERS + NENG - 1) / NENG ))}"

DATA="${DATA:-$DEMO_DIR/data/swebench_train.jsonl}"
EVAL_DATA="${EVAL_DATA:-$DEMO_DIR/data/swebench_eval.jsonl}"
EVAL_NAME="${EVAL_NAME:-swebench}"
SAVE="${SAVE:-$DEMO_DIR/ckpt/Qwen3-4B_swebench_async}"

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
   --async-save
   --custom-megatron-init-path agentcore_slime_demo.megatron_init.enable_persistent_async_ckpt
)

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
   --rollout-max-response-len 1024
   --rollout-temperature "$TEMP"
   --global-batch-size "$GBS"
   # THE POOL HOOK: routes each episode to its instance's Runtime on its own fresh, pre-opened session.
   --custom-generate-function-path agentcore_slime_demo.terminalbench_pool_generate.generate
   # THE LOOK-AHEAD: peeks the next TB_PREOPEN_BATCHES rollouts' prompts so their sessions boot early.
   --data-source-path agentcore_slime_demo.tb_lookahead_source.LookaheadDataSource
)

MODE_ARGS=()

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu "$MAX_TOKENS_PER_GPU"
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
   --sglang-server-concurrency "$CONC"
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

# The isolation/pre-open/heartbeat knobs and the image-pool map are read by TerminalBenchPool INSIDE the
# rollout process, so they MUST be forwarded through the ray runtime env (not just the launching shell).
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM:${DEMO_DIR}:/root/slime\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"SANDBOX\": \"${SANDBOX}\",
    \"ADAPTER_PUBLIC_HOST\": \"${ADAPTER_PUBLIC_HOST:-}\",
    \"ADAPTER_PORT_BASE\": \"${ADAPTER_PORT_BASE:-18100}\",
    \"MINI_SESSION_POLL_SEC\": \"${MINI_SESSION_POLL_SEC:-25}\",
    \"EKS_SANDBOX_IMAGE_BY_TASK\": $(python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))' <<< "${EKS_SANDBOX_IMAGE_BY_TASK}"),
    \"EKS_CLUSTER\": \"${EKS_CLUSTER}\",
    \"EKS_NAMESPACE\": \"${EKS_NAMESPACE}\",
    \"EKS_RUNTIME_CLASS\": \"${EKS_RUNTIME_CLASS-gvisor}\",
    \"EKS_POD_CPU\": \"${EKS_POD_CPU:-1500m}\",
    \"EKS_POD_MEMORY\": \"${EKS_POD_MEMORY:-6Gi}\",
    \"EKS_POD_MEMORY_LIMIT\": \"${EKS_POD_MEMORY_LIMIT:-8Gi}\",
    \"EKS_OPEN_TIMEOUT\": \"${EKS_OPEN_TIMEOUT:-1200}\",
    \"AWS_REGION\": \"${AWS_REGION:-us-west-2}\",
    \"AWS_BEARER_TOKEN_BEDROCK\": \"${AWS_BEARER_TOKEN_BEDROCK:-}\",
    \"HARNESS_WORKERS\": \"${HARNESS_WORKERS}\",
    \"EPISODE_MAX_TURNS\": \"${EPISODE_MAX_TURNS}\",
    \"EPISODE_MAX_TOKENS\": \"${EPISODE_MAX_TOKENS}\",
    \"TB_PREOPEN_BATCHES\": \"${TB_PREOPEN_BATCHES}\",
    \"MINI_AGENT_MAX_TOOL_CHARS\": \"${MINI_AGENT_MAX_TOOL_CHARS}\",
    \"TB_EXEC_TIMEOUT\": \"${TB_EXEC_TIMEOUT}\"
  }
}"

cd "$SLIME"
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 "$SLIME/train_async.py" \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "$ACTOR_GPUS" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${MODE_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"
