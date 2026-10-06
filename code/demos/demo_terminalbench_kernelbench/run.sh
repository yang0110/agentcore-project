#!/bin/bash
# demo_terminalbench_kernelbench — agentic RL on ALL of Terminal-Bench, on two sandbox substrates: ONE image +
# AgentCore microVM Runtime PER single-container task that fits, and the m5 Docker host (one compose project
# per episode) for everything else (compose, x86-only, > 2 GB, microVM oracle failures). A fresh sandbox per
# episode on either, pre-opened by the look-ahead data source; microVM sessions heartbeated across the
# weight-sync stall.
#
# ONE ROW = ONE TASK = ONE IMAGE = ONE ROUTE. Every row names its task's image (metadata.task.image);
# image_routing.py maps it to that task's route — an AgentCore Runtime ARN or docker://<m5>/<task> (strict:
# an unmapped image stops the run). Each of the RB x NS samples of a step gets its OWN fresh sandbox.
# REAL slime training with the MiniClaudeHarness, on OUR OWN AgentCore Runtimes (SANDBOX=agentcore-runtime,
# the only backend; networkMode=VPC) with the agent loop INSIDE each episode's session (dialling this
# box's adapter back SG->SG); the session is stopped when the episode ends.
# ASYNC (decoupled) mode. This dir owns its data/ + ckpt/, but imports the SHARED agentcore_slime_demo
# library from its own vendored copy in this dir (on PYTHONPATH). Runs INSIDE the persistent
# container agentcore_p4 (slimerl/slime).
#
# Harness/sandbox placement (DSec §6.2: one agent loop per sandbox, sharing its lifetime): each episode
# gets ONE fresh AgentCore session; mini_claude_episode.py writes the stdlib agent loop into it
# (in_session_harness.py + session_runner.py), the loop calls the policy through this box's per-engine
# AnthropicAdapter (tokens recorded here), and a hidden grader scores the workdir after it stops.
# Only the adapter + token record stay in the trainer; there is no in-trainer harness in any demo.
#
# Async/decoupled: agentic rollouts have long, variable latency (multi-turn code repair in a
# sandbox). slime's async driver (train_async.py) runs trainer GPUs and SGLang inference GPUs on
# DISJOINT devices. Hard-requires no --colocate (train_async.py asserts it).
#
# FULLY-ASYNC rollout driver (the only one): on top of train_async.py's disjoint train/infer GPUs,
# --rollout-function-path generate_rollout_fully_async runs a background worker that keeps
# CONC x num_engines groups in flight and COMPLETED groups queued ahead of the trainer, across step
# boundaries — the rollout never waits on a step barrier, and staleness grows with CONC (see
# docs/sandbox-concurrency-ablation.md). Eval goes through agentcore_slime_demo.mode_b_eval (below).
# The custom generate hook + Sample contract (tokens/logprobs/loss-mask/reward) is the stock one.
#
# GPU split (8xL40S): 4 train + 4 inference + 0 env. MBPP checker is plain Python/IO => CPU envs,
# so all 8 GPUs serve the policy (train + infer).
#
# Sandbox backend (SANDBOX): agentcore-runtime (the only one) -> our own AgentCore Runtimes, one per
# task (steps 2-4 deploy them, VPC mode). SANDBOX=local and SANDBOX=agentcore
# (Code Interpreter) are rejected (see the guard below). Tunables: RB (rollout-batch-size), NS (n-samples-per-prompt), NR (num-rollout), TEMP
# (sampling temp), CONC (sglang-server-concurrency), EVAL_INTERVAL/EVAL_NS, HARNESS_WORKERS (harness
# tier, default RB x NS), SANDBOX_SESSIONS (opt-in pooled sessions per image, default 0 = fresh per
# episode).
#
# CONCURRENCY KNOBS (decoupled, for the sandbox-concurrency ablation — see
# docs/sandbox-concurrency-ablation.md):
#   * CONC       -> SANDBOX/EPISODE concurrency. #episodes (= #concurrent remote sandboxes) in
#                   flight is S = CONC * num_engines = CONC * 2. This is the knob to SWEEP. It
#                   also sets the fully-async BUFFER DEPTH: the worker keeps up to
#                   S in-flight AND up to S completed groups queued ahead of the trainer (which
#                   drains RB/step), so raising CONC raises staleness. One knob moves both, which is
#                   exactly the trade-off the ablation measures.
#
# NOTE step count: slime resumes at loaded_iter+1; a fresh torch_dist ckpt is iter 0 => the loop
# runs range(1, NR) = (NR-1) real training steps. NR=11 => 10 train + 1 eval (SMOKE; no more needed).
# NOTE GRPO signal: group-relative advantage is 0 when every sample in a prompt's group gets the
# same reward. Reward variance within groups (harder tasks / higher temp) yields nonzero gradient.
set -euo pipefail
if [ -n "${MODE:-}" ]; then
  echo "MODE is gone: the fully-async driver (formerly MODE=B) is the only rollout driver. Unset MODE." >&2
  exit 1
fi

SANDBOX="${SANDBOX:-agentcore-runtime}"   # our own VPC Runtime(s) — the only backend.
# CPU demos run ONLY on remote AgentCore sandbox sessions that can host the agent loop — never local
# (in-container LocalSandbox), and no longer the managed Code Interpreter (SANDBOX=agentcore): it has no
# route back to the adapter, so its agent loop would have had to run in the trainer. Rejected here so a
# run can never silently move the loop or the sandbox off the substrate the demo exercises.
case "$SANDBOX" in agentcore-runtime|runtime) ;;
   local) echo "SANDBOX=local is not supported for CPU demos (remote-only) — use agentcore-runtime." >&2; exit 1 ;;
   agentcore|aws) echo "SANDBOX=$SANDBOX (Code Interpreter) was removed: the agent loop runs in the sandbox session and" \
                       "the Code Interpreter cannot reach the adapter. Use SANDBOX=agentcore-runtime (steps 2-4)." >&2; exit 1 ;;
   *) echo "unknown SANDBOX=$SANDBOX (agentcore-runtime)" >&2; exit 1 ;; esac
# No generic sandbox Runtime: every Terminal-Bench row is routed strictly to its own task Runtime via
# AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE (IMAGE ROUTING below; terminalbench_pool refuses any other).
RB="${RB:-4}"          # prompts per rollout step
NS="${NS:-8}"          # GRPO group size (samples per prompt)
NR="${NR:-11}"        # num-rollout; real training steps = NR-1 (see NOTE) => 10 train + 1 eval (SMOKE; no more needed).
TEMP="${TEMP:-1.0}"    # rollout sampling temperature (higher => more intra-group diversity)
GBS=$(( RB * NS ))     # global batch = all samples, 1 optimizer step per rollout
# Held-out eval every N steps (0 => off). Held-out solve-rate is a KEY metric, so eval stays ON
# under fully-async. It does NOT work through the stock rollout path, though: slime's
# GenerateState is a process singleton whose asyncio.Semaphore binds to the FIRST event loop that
# acquires it — the background fully-async worker's loop — while eval runs generation on the MAIN
# loop, which would crash with "Semaphore is bound to a different event loop". We route eval
# through agentcore_slime_demo.mode_b_eval.generate_rollout (see EVAL_ARGS), which quiesces the
# worker and rebuilds the singleton on the eval loop so stock eval runs cleanly. NOTE: each eval
# discards the worker's warm buffer (it re-warms over the next few rollouts), so keep EVAL_INTERVAL
# well above the re-warm time (default 10 is fine for the smoke; use a larger value for long runs).
EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
EVAL_NS="${EVAL_NS:-2}"               # samples per eval prompt
# Checkpoint policy (disk-saving): we do NOT snapshot mid-training — the ONLY checkpoint written is
# the final one. slime saves whenever should_run_periodic_action() fires; here that is ONLY the last
# rollout (rollout_id == num_rollout-1, always forced). Its two other triggers are both inactive:
# (a) step % save-interval == 0 is disabled by the huge SAVE_INTERVAL below, and (b) the
# epoch-boundary trigger (step % num_rollout_per_epoch) is OFF because we pass an explicit
# --num-rollout (NR) => slime leaves num_rollout_per_epoch = None (it is computed only under
# --num-epoch). Net: exactly one checkpoint, at the end of training, regardless of dataset size.
SAVE_INTERVAL="${SAVE_INTERVAL:-1000000}"

SLIME=/root/slime

DEMO_DIR=/work/code/demos/demo_terminalbench_kernelbench
CKPT="${CKPT_BASE:-/work/code/p4_slime_train/ckpt/Qwen3-4B_torch_dist}"  # SHARED base Megatron ckpt
HF=/root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c

# ---- Demo A config (baked in) --------------------------------------------------------------
ACTOR_GPUS="${ACTOR_GPUS:-4}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-4}"
GPUS_PER_ENGINE="${GPUS_PER_ENGINE:-2}"   # TP per SGLang engine; num_engines = ROLLOUT_GPUS / GPUS_PER_ENGINE
NENG=$(( ROLLOUT_GPUS / GPUS_PER_ENGINE ))

# DATA: data/terminalbench_{train,eval}.jsonl from step 1 = the deployed, oracle-verified tasks only.
DATA="${DATA:-$DEMO_DIR/data/terminalbench_train.jsonl}"
EVAL_DATA="${EVAL_DATA:-$DEMO_DIR/data/terminalbench_eval.jsonl}"                 # microVM tasks
EVAL_DOCKER_DATA="${EVAL_DOCKER_DATA:-$DEMO_DIR/data/terminalbench_eval_docker.jsonl}"  # m5 tasks
EVAL_GPU_DATA="${EVAL_GPU_DATA:-$DEMO_DIR/data/terminalbench_eval_gpu.jsonl}"           # g6 tasks (TB_GPU_TASKS)
EVAL_KB_DATA="${EVAL_KB_DATA:-$DEMO_DIR/data/kernelbench_eval.jsonl}"                   # KernelBench on g6
EVAL_NAME="${EVAL_NAME:-terminalbench}"
[ -f "$DATA" ] || { echo "no $DATA — run 1_prepare_data.sh (after steps 2-4)" >&2; exit 2; }
SHUFFLE_ARGS=(--rollout-shuffle)
# IMAGE ROUTING: task image -> its route (terminalbench/tasks/route_by_image.json, via image_routing.py).
# A row whose image has no route is an error here, before any GPU work.
TB_ROUTE_BY_IMAGE="$(python3 "$DEMO_DIR/image_routing.py" "$DATA" "$EVAL_DATA" "$EVAL_DOCKER_DATA" "$EVAL_GPU_DATA" "$EVAL_KB_DATA")"
export TB_ROUTE_BY_IMAGE
# The m5 compose agent's shared secret (deploy/m5_host.json from `deploy/m5_host.py up`), for docker:// routes.
M5_STATE="$DEMO_DIR/deploy/m5_host.json"
if [ -z "${TB_DOCKER_TOKEN:-}" ] && grep -q 'docker://' <<<"$TB_ROUTE_BY_IMAGE"; then
  [ -f "$M5_STATE" ] || { echo "the route map has m5 tasks but $M5_STATE is missing (deploy/m5_host.py up)" >&2; exit 2; }
  TB_DOCKER_TOKEN="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['token'])" "$M5_STATE")"
fi
TB_DOCKER_MAX_PROJECTS="${TB_DOCKER_MAX_PROJECTS:-16}"   # shared mode: m5 episode projects at once (pre-opened + running)
# m5 tier mode: per_episode (default) = ONE fresh EC2 instance per episode, from the AMI baked by
# `deploy/m5_host.py bake-ami` (deploy/m5_host.json "episode"), terminated after grading; shared = one
# compose project per episode on the one shared m5.
TB_M5_MODE="${TB_M5_MODE:-warm}"
TB_M5_MAX_INSTANCES="${TB_M5_MAX_INSTANCES:-24}"          # warm/per_episode: live m5 instances at once (~87% of moments need <= 24; see README)
TB_M5_IDLE_SHUTDOWN_S="${TB_M5_IDLE_SHUTDOWN_S:-1800}"   # m5 dead-man switch: no trainer call this long -> instance terminates itself (0 = off)
TB_PREOPEN_BATCHES_SLOW="${TB_PREOPEN_BATCHES_SLOW:-5}"   # m5 VMs / g6 Instances open this many batches ahead (microVMs: TB_PREOPEN_BATCHES); a g6 needs ~2.5 min
TB_M5_EPISODE="${TB_M5_EPISODE:-}"
if [ "$TB_M5_MODE" != shared ] && [ -z "$TB_M5_EPISODE" ] && grep -q 'docker://' <<<"$TB_ROUTE_BY_IMAGE"; then
  TB_M5_EPISODE="$(python3 -c "import json,sys; print(json.dumps(json.load(open(sys.argv[1])).get('episode') or ''))" "$M5_STATE")"
  [ "$TB_M5_EPISODE" != '""' ] || { echo "TB_M5_MODE=per_episode but no baked AMI: python3 deploy/m5_host.py roles && python3 deploy/m5_host.py bake-ami (or TB_M5_MODE=shared)" >&2; exit 2; }
fi
# Each task changes the whole OS, so it runs in the isolated per-task pool (terminalbench_pool.py): a
# FRESH session per episode opened ahead of time — the look-ahead data source (tb_lookahead_source.py)
# boots the next TB_PREOPEN_BATCHES rollouts' sessions (2 = 2 x RB x NS = 64) — a SessionKeeper heartbeat
# across the weight-sync stall, graded by the task's hidden pytest oracle.
HARNESS_WORKERS="${HARNESS_WORKERS:-$GBS}"
# Admit W running + W queued episodes per step (S = CONC x NENG) so a free worker never idles.
CONC="${CONC:-$(( (2 * HARNESS_WORKERS + NENG - 1) / NENG ))}"
EPISODE_MAX_TURNS="${EPISODE_MAX_TURNS:-8}"     # agentic turns per episode
EPISODE_MAX_TOKENS="${EPISODE_MAX_TOKENS:-1024}"
MINI_AGENT_MAX_TOOL_CHARS="${MINI_AGENT_MAX_TOOL_CHARS:-2000}"  # clip each tool result (bounds the trajectory)
TB_EXEC_TIMEOUT="${TB_EXEC_TIMEOUT:-180}"       # per bash-tool command (apt/pip/builds are routine)
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-16384}"  # >= one full trajectory: 8 x (1024 gen + ~600 tool) + prompt
WARM_IDLE_TIMEOUT="${WARM_IDLE_TIMEOUT:-900}"   # serverless microVM idle timeout (~15 min); heartbeat < this
WARM_PING_FRACTION="${WARM_PING_FRACTION:-0.5}"
TB_PREOPEN_BATCHES="${TB_PREOPEN_BATCHES:-2}"   # batches kept booted ahead (0 = off)
TB_ARGS=()
if [ "$TB_PREOPEN_BATCHES" != 0 ]; then
  TB_ARGS=(--data-source-path agentcore_slime_demo.tb_lookahead_source.LookaheadDataSource)
fi
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
_HB=""; [ "$HEARTBEAT" = "0" ] && [ -z "${AGENTCORE_LOCAL_ENDPOINT:-}" ] && case "$SANDBOX" in agentcore-*) _HB="_noheartbeat" ;; esac
SAVE="${SAVE:-$DEMO_DIR/ckpt/Qwen3-4B_terminalbench${_HB}_async}"

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
   --save-interval "$SAVE_INTERVAL"   # save-only-at-end (see SAVE_INTERVAL note above)
   # ---- Checkpoint-save deadlock fix (two flags that MUST go together) ---------------------
   # Megatron's checkpoint writer forks a writer process off the MAIN training process
   # (mp.get_context("fork")); forking a process holding CUDA/NCCL/inductor threads can inherit a
   # locked mutex with no thread to release it -> the writer hangs forever AFTER the .distcp files
   # are written. The safe backend is Megatron's SPAWN-based PersistentAsyncCaller, selected only
   # when BOTH --async-save is set AND init_persistent_async_worker() is called. slime forces
   # use_persistent_ckpt_worker=True but its own init() bypasses megatron.training.initialize, so
   # it NEVER makes that call -> --async-save alone falls back to the fork-based caller and can
   # still deadlock. So we supply the missing call via slime's supported init hook:
   --async-save
   --custom-megatron-init-path agentcore_slime_demo.megatron_init.enable_persistent_async_ckpt
)

# Held-out EVAL: the stock rollout path can't run eval (the fully-async worker's event loop owns
# slime's GenerateState singleton semaphore, so in-process eval would crash with "Semaphore is bound
# to a different event loop"). We route eval through agentcore_slime_demo.mode_b_eval.generate_rollout,
# which quiesces the fully-async worker (waits for its loop thread to fully die), drops the
# GenerateState singleton so the semaphore rebuilds on THIS loop, runs slime's stock eval unchanged,
# then drops the singleton again so the next training rollout's worker rebuilds it cleanly on the
# worker loop. (generate_rollout_fully_async itself raises on evaluation=True, which is exactly why
# eval needs its own function-path.)
EVAL_FN=agentcore_slime_demo.mode_b_eval.generate_rollout
EVAL_ARGS=()
if [ "$EVAL_INTERVAL" != "0" ] && [ -f "$EVAL_DATA" ]; then
   # Two held-out sets: eval/terminalbench = the microVM tasks (the same set the microVM-only demo used),
   # eval/terminalbench_docker = the m5 tasks.
   EVAL_SETS=("$EVAL_NAME" "$EVAL_DATA")
   [ -s "$EVAL_DOCKER_DATA" ] && EVAL_SETS+=("${EVAL_NAME}_docker" "$EVAL_DOCKER_DATA")
   [ -s "$EVAL_GPU_DATA" ] && EVAL_SETS+=("${EVAL_NAME}_gpu" "$EVAL_GPU_DATA")
   [ -s "$EVAL_KB_DATA" ] && EVAL_SETS+=("kernelbench" "$EVAL_KB_DATA")
   EVAL_ARGS=(
      --eval-interval "$EVAL_INTERVAL"
      --eval-prompt-data "${EVAL_SETS[@]}"
      --n-samples-per-eval-prompt "$EVAL_NS"
      --eval-temperature "$TEMP"
      --eval-function-path "$EVAL_FN"
   )
fi

ROLLOUT_ARGS=(
   --prompt-data "$DATA"
   --input-key prompt
   --label-key label
   --metadata-key metadata
   "${SHUFFLE_ARGS[@]}"
   --num-rollout "$NR"
   --rollout-batch-size "$RB"
   --n-samples-per-prompt "$NS"
   --rollout-max-response-len 1024
   --rollout-temperature "$TEMP"
   --global-batch-size "$GBS"
   # The fully-async driver (see the note near the top). Its worker calls slime's
   # generate_and_rm_group, which dispatches to our --custom-generate-function-path, so our token-ids / logprobs / loss-mask /
   # reward Sample contract is preserved UNCHANGED (reward set on the Sample => no --custom-rm-path).
   # ...wrapped by last_rollout: after the LAST rollout it stops the worker + Terminal-Bench look-ahead
   # (cancels pre-opens, stops their sessions) instead of letting it run on through the final save.
   --rollout-function-path agentcore_slime_demo.last_rollout.generate_rollout
   --custom-generate-function-path agentcore_slime_demo.generate.generate
   # Per-step task mix of the batch actually trained on ([batch] line + mix/n_tasks, mix/{episodes,reward}_<family>)
   # and each task's held-out reward (eval/<name>_<family>).
   --custom-rollout-log-function-path agentcore_slime_demo.batch_mix_log.log_rollout_data
   --custom-eval-rollout-log-function-path agentcore_slime_demo.batch_mix_log.log_eval_rollout_data
   # Terminal-Bench rows only: peek the next TB_PREOPEN_BATCHES rollouts so their sessions boot early.
   "${TB_ARGS[@]}"
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
   --rollout-num-gpus-per-engine "$GPUS_PER_ENGINE"
   --sglang-mem-fraction-static 0.7
   # SANDBOX/EPISODE concurrency S. slime gates custom-generate with
   # Semaphore(sglang_server_concurrency * num_engines), so with num_engines = ROLLOUT_GPUS /
   # per-engine = 4/2 = 2, the number of episodes (= concurrent remote sandboxes) in flight is
   # S = CONC * 2. This is the knob to SWEEP for the sandbox-concurrency ablation
   # (docs/sandbox-concurrency-ablation.md). Each episode opens an HTTP session AND a remote
   # sandbox session, so raising it also raises client-side load (asyncio thread pool, boto3
   # clients) — see the ablation doc's "can the in-process harness handle large C" section.
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
# Give ray ALL 8 physical GPUs. slime's placement reserves (ACTOR_GPUS + ROLLOUT_GPUS) of them.
# Shared box only: SLIME_CUDA_VISIBLE=2,3,4,5 SLIME_NUM_GPUS=4 pins ray to a subset (default: all 8).
[ -n "${SLIME_CUDA_VISIBLE:-}" ] && export CUDA_VISIBLE_DEVICES="$SLIME_CUDA_VISIBLE"
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${SLIME_NUM_GPUS:-8}" \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# PYTORCH_CUDA_ALLOC_CONF is opt-in: with expandable_segments:True, CUDA IPC to Megatron's async-save worker
# goes through pidfd_getfd, which agentcore_p4's default seccomp profile forbids (no CAP_SYS_PTRACE) — the
# final checkpoint save then hangs with "pidfd_getfd: Operation not permitted" (2026-10-04 smoke).
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM:${DEMO_DIR}:/root/slime\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"${PYTORCH_CUDA_ALLOC_CONF:-}\",
    \"SANDBOX\": \"${SANDBOX}\",
    \"HEARTBEAT\": \"${HEARTBEAT}\",
    \"HARNESS\": \"${HARNESS}\",
    \"ADAPTER_PUBLIC_HOST\": \"${ADAPTER_PUBLIC_HOST:-}\",
    \"ADAPTER_PORT_BASE\": \"${ADAPTER_PORT_BASE:-18100}\",
    \"HARNESS_WORKERS\": \"${HARNESS_WORKERS:-}\",
    \"SANDBOX_SESSIONS\": \"${SANDBOX_SESSIONS:-0}\",
    \"SANDBOX_OPEN_RATE\": \"${SANDBOX_OPEN_RATE:-20}\",
    \"MINI_SESSION_POLL_SEC\": \"${MINI_SESSION_POLL_SEC:-25}\",
    \"TB_ROUTE_BY_IMAGE\": \"$(printf '%s' "${TB_ROUTE_BY_IMAGE:-}" | sed 's/\\/\\\\/g; s/\"/\\\"/g')\",
    \"TB_DOCKER_TOKEN\": \"${TB_DOCKER_TOKEN:-}\",
    \"TB_DOCKER_MAX_PROJECTS\": \"${TB_DOCKER_MAX_PROJECTS}\",
    \"TB_DOCKER_OPEN_WAIT_S\": \"${TB_DOCKER_OPEN_WAIT_S:-1800}\",
    \"TB_GPU_MAX_SESSIONS\": \"${TB_GPU_MAX_SESSIONS:-32}\",
    \"TB_KB_SPEEDUP_MIN\": \"${TB_KB_SPEEDUP_MIN:-1.10}\",
    \"MINI_AGENT_EXEC_TIMEOUT\": \"${MINI_AGENT_EXEC_TIMEOUT:-300}\",
    \"MINI_SESSION_WALL_SEC\": \"${MINI_SESSION_WALL_SEC:-3840}\",
    \"MINI_GRADE_TIMEOUT\": \"${MINI_GRADE_TIMEOUT:-600}\",
    \"TB_M5_MODE\": \"${TB_M5_MODE}\",
    \"TB_M5_MAX_INSTANCES\": \"${TB_M5_MAX_INSTANCES}\",
    \"TB_M5_IDLE_SHUTDOWN_S\": \"${TB_M5_IDLE_SHUTDOWN_S}\",
    \"TB_M5_EPISODE\": \"$(printf '%s' "${TB_M5_EPISODE}" | sed 's/\\/\\\\/g; s/\"/\\\"/g')\",
    \"HEARTBEAT_IDLE\": \"${HEARTBEAT_IDLE}\",
    \"HEARTBEAT_PING_FRACTION\": \"${HEARTBEAT_PING_FRACTION}\",
    \"AGENTCORE_SANDBOX_RUNTIME_ARN\": \"${AGENTCORE_SANDBOX_RUNTIME_ARN:-}\",
    \"AGENTCORE_SANDBOX_RUNTIME_ARNS\": \"${AGENTCORE_SANDBOX_RUNTIME_ARNS:-}\",
    \"AWS_REGION\": \"${AWS_REGION:-us-west-2}\",
    \"AWS_BEARER_TOKEN_BEDROCK\": \"${AWS_BEARER_TOKEN_BEDROCK:-}\",
    \"EPISODE_MAX_TURNS\": \"${EPISODE_MAX_TURNS:-4}\",
    \"EPISODE_MAX_TOKENS\": \"${EPISODE_MAX_TOKENS:-1024}\",
    \"MINI_AGENT_MAX_TOOL_CHARS\": \"${MINI_AGENT_MAX_TOOL_CHARS:-4000}\",
    \"TB_EXEC_TIMEOUT\": \"${TB_EXEC_TIMEOUT:-180}\",
    \"WARM_IDLE_TIMEOUT\": \"${WARM_IDLE_TIMEOUT:-900}\",
    \"WARM_PING_FRACTION\": \"${WARM_PING_FRACTION:-0.5}\",
    \"TB_PREOPEN_BATCHES\": \"${TB_PREOPEN_BATCHES:-2}\",
    \"TB_PREOPEN_BATCHES_SLOW\": \"${TB_PREOPEN_BATCHES_SLOW}\"
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
