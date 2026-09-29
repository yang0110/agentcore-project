#!/bin/bash
# demo_cpu_swebench_ecs — REAL slime CPU trainer on a SWE-bench Verified subset, every episode ISOLATED in
# its own fresh ECS TASK on EC2 (awsvpc: own ENI + security group). The twin of demo_cpu_swebench
# (AgentCore arm64 microVM per episode) and demo_cpu_swebench_eks (gVisor pod per episode): same dataset
# rows, same in-session agent loop, same grader, same metrics — only the sandbox tier differs, so the
# runs compare substrates and nothing else. Images: the amd64 wrappers shared with the EKS twin.
#
# Each episode routes to ITS instance's task definition (pool map ECS_SANDBOX_TASKDEF_BY_TASK = {instance
# -> task definition ARN}, written by deploy/ecs_deploy.py to swebench/taskdef_by_task.json; an unmapped
# instance is an error, never routed elsewhere). The pool (TerminalBenchPool, sandbox tier swapped to
# ecs_task_sandbox.EcsTaskSandbox) RunTasks one task per episode on the capacity provider, talks to the
# in-image server on the task's ENI IP (:8080), runs the MiniClaudeHarness loop in /testbed (LLM via the
# adapter under training, which the task dials back on the trainer's private IP — the only egress its SG
# allows), grades with swebench.grade_swebench (hidden test_patch, FAIL_TO_PASS + PASS_TO_PASS), then
# StopTasks it. Runs INSIDE the persistent container agentcore_p4 (slimerl/slime); needs the substrate
# from `ECS_CREATE=1 bash 4_deploy_runtime.sh`.
#
# WHY A FRESH TASK PER EPISODE: the agent edits the repo and may pip-install into the testbed env; grading
# must see exactly the environment SWE-bench validated, so no task is ever reused. The task cold start
# (placement + image pull on a node that has not cached it + container start) is hidden the same way as on
# AgentCore: opened at submit time and, via the look-ahead data source, TB_PREOPEN_BATCHES rollouts ahead.
# No heartbeat: tasks are not idle-reclaimed.
#
# FULLY-ASYNC DRIVER, as in demo_cpu_swebench (the look-ahead pre-open data source slides with its draws).
#
# ADAPTER TOOL PARSER (the A3/B3 dead-reward fix): SGLANG_ARGS bakes in --sglang-tool-call-parser
# qwen25 --sglang-reasoning-parser qwen3. Without them the tool loop is dead and reward is structurally 0.
set -euo pipefail
case "${HARNESS_WHERE:-session}" in session) ;; *) echo "HARNESS_WHERE=$HARNESS_WHERE is not supported: the agent loop always runs in the sandbox session" >&2; exit 1 ;; esac

SANDBOX="${SANDBOX:-ecs-ec2}"   # fresh ECS task per episode (the pool RunTasks it).
if [ "$SANDBOX" != "ecs-ec2" ]; then
   echo "SANDBOX=$SANDBOX is not supported here — this demo's only sandbox is ecs-ec2 (AgentCore: demo_cpu_swebench, EKS: demo_cpu_swebench_eks)." >&2
   exit 1
fi

DEMO_DIR=/work/code/demos/demo_cpu_swebench_ecs
POOL_MAP="${POOL_MAP:-$DEMO_DIR/swebench/taskdef_by_task.json}"   # {instance -> task definition ARN}
ECS_NETWORK="${ECS_NETWORK:-$DEMO_DIR/swebench/ecs_network.json}"  # cluster / CP / subnets / task SG
# Both are REQUIRED: every episode routes to its instance's task definition, in the task SG.
for f in "$POOL_MAP" "$ECS_NETWORK"; do
   if [ ! -f "$f" ]; then
      echo "$f not found — run \`ECS_CREATE=1 python3 deploy/ecs_deploy.py up\` first." >&2
      exit 1
   fi
done
ECS_SANDBOX_TASKDEF_BY_TASK="$(cat "$POOL_MAP")"
_net() { python3 -c "import json,sys; v=json.load(open(sys.argv[1]))[sys.argv[2]]; print(','.join(v) if isinstance(v,list) else v)" "$ECS_NETWORK" "$1"; }
ECS_CLUSTER="${ECS_CLUSTER:-$(_net cluster)}"
ECS_CAPACITY_PROVIDER="${ECS_CAPACITY_PROVIDER:-$(_net capacity_provider)}"
ECS_SUBNETS="${ECS_SUBNETS:-$(_net subnets)}"
ECS_SECURITY_GROUPS="${ECS_SECURITY_GROUPS:-$(_net security_groups)}"

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
# THE pre-open knob: batches kept booted ahead (x RB x NS = 64 tasks; 0 = off). Tasks that do not fit on
# the nodes wait in PROVISIONING (managed scaling adds nodes) and are counted in open_ms — size
# ECS_NODES_MAX for RB x NS x (1 + this) tasks (~4 per m5.2xlarge at the default 1.5 vCPU / 8 GiB).
# HEARTBEAT (default 1; 0 = A/B): the keeper pings every open sandbox idle past WARM_PING_FRACTION x
# WARM_IDLE_TIMEOUT (240 x 0.5 = every ~2 min). There is no idle reclaim here, so it is a liveness check:
# a pre-opened task that fails it (or is too close to ECS_TASK_MAX_LIFETIME) is replaced before a worker
# claims it ([demo] preopen_replaced=…); a running episode's task is only logged ([demo] heartbeat_failed).
WARM_IDLE_TIMEOUT="${WARM_IDLE_TIMEOUT:-240}"
WARM_PING_FRACTION="${WARM_PING_FRACTION:-0.5}"
HEARTBEAT="${HEARTBEAT:-1}"
TB_PREOPEN_BATCHES="${TB_PREOPEN_BATCHES:-2}"
# Pre-open is capped to the fleet: SANDBOX_SLOTS (ECS_NODES_MAX x ECS_TASKS_PER_NODE, ~4 per m5.2xlarge) minus the
# running episodes (HARNESS_WORKERS). Set SANDBOX_SLOTS=0 for no cap. ECS_TASK_MAX_LIFETIME must match the deploy.
SANDBOX_SLOTS="${SANDBOX_SLOTS:-$(( ${ECS_NODES_MAX:-16} * ${ECS_TASKS_PER_NODE:-4} ))}"
ECS_TASK_MAX_LIFETIME="${ECS_TASK_MAX_LIFETIME:-7200}"
TB_EPISODE_BUDGET_SEC="${TB_EPISODE_BUDGET_SEC:-0}"   # 0 = EPISODE_MAX_TURNS x TB_EXEC_TIMEOUT + 660 s

if [ -n "${MODE:-}" ]; then
  echo "MODE is gone: the fully-async driver is the only rollout driver. Unset MODE." >&2
  exit 1
fi

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
   # THE DRIVER: a background worker keeps CONC x engines groups in flight across step boundaries.
   --rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async
   # THE POOL HOOK: routes each episode to its instance's Runtime on its own fresh, pre-opened session.
   --custom-generate-function-path agentcore_slime_demo.terminalbench_pool_generate.generate
   # THE LOOK-AHEAD: peeks the next TB_PREOPEN_BATCHES x RB prompts so their sessions boot early.
   --data-source-path agentcore_slime_demo.tb_lookahead_source.LookaheadDataSource
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
    \"ECS_SANDBOX_TASKDEF_BY_TASK\": $(python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))' <<< "${ECS_SANDBOX_TASKDEF_BY_TASK}"),
    \"ECS_CLUSTER\": \"${ECS_CLUSTER}\",
    \"ECS_CAPACITY_PROVIDER\": \"${ECS_CAPACITY_PROVIDER}\",
    \"ECS_SUBNETS\": \"${ECS_SUBNETS}\",
    \"ECS_SECURITY_GROUPS\": \"${ECS_SECURITY_GROUPS}\",
    \"ECS_OPEN_TIMEOUT\": \"${ECS_OPEN_TIMEOUT:-1200}\",
    \"ECS_POLL_SEC\": \"${ECS_POLL_SEC:-1.0}\",
    \"AWS_REGION\": \"${AWS_REGION:-us-west-2}\",
    \"AWS_BEARER_TOKEN_BEDROCK\": \"${AWS_BEARER_TOKEN_BEDROCK:-}\",
    \"HARNESS_WORKERS\": \"${HARNESS_WORKERS}\",
    \"EPISODE_MAX_TURNS\": \"${EPISODE_MAX_TURNS}\",
    \"EPISODE_MAX_TOKENS\": \"${EPISODE_MAX_TOKENS}\",
    \"TB_PREOPEN_BATCHES\": \"${TB_PREOPEN_BATCHES}\",
    \"WARM_IDLE_TIMEOUT\": \"${WARM_IDLE_TIMEOUT}\",
    \"WARM_PING_FRACTION\": \"${WARM_PING_FRACTION}\",
    \"HEARTBEAT\": \"${HEARTBEAT}\",
    \"SANDBOX_SLOTS\": \"${SANDBOX_SLOTS}\",
    \"ECS_TASK_MAX_LIFETIME\": \"${ECS_TASK_MAX_LIFETIME}\",
    \"TB_EPISODE_BUDGET_SEC\": \"${TB_EPISODE_BUDGET_SEC}\",
    \"TB_PREOPEN_RECHECK_SEC\": \"${TB_PREOPEN_RECHECK_SEC:-2}\",
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
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"
