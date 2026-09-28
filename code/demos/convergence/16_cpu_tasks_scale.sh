#!/bin/bash
# ============================================================================
# CONVERGENCE RUN 16 — demo_cpu_tasks SCALE=1  (deepscaler; 4 engines : 256 workers : 256 sessions; was demo_cpu_scale_lanes)
# ============================================================================
# TASK / DATASET : DeepScaleR competition-math (same task as 05/15). Multi-turn
#                  solve-in-sandbox, RLVR final-answer check. Train ~400 prompts,
#                  disjoint held-out eval.
# SUBSTRATE      : our own VPC CPU sandbox Runtime (SANDBOX=agentcore-runtime); a pool of
#                  SANDBOX_SESSIONS PERSISTENT microVM sessions shared by all workers.
# TOPOLOGY       : 4 SGLang engines (TP1, GPUS_PER_ENGINE=1) : HARNESS_WORKERS=256 dispatch
#                  workers : SANDBOX_SESSIONS=256 sessions, sized independently. Each episode
#                  takes a free worker, the least-loaded engine and any free session (fresh workdir).
# PROPERTY (LIVE): scales the HARNESS TIER with the sandbox. The decoupled harness tier
#                  (harness_pool.py) is every demo's default; SCALE=1 only turns up its sizes.
#                  The mini_claude loop runs in each episode's sandbox session, so 256 sessions
#                  carry 256 loops; the trainer keeps only the W dispatch workers + adapters.
# MODE           : A (the SCALE preset's default); MODE=B also works (one bare Sample per episode).
# PARSERS        : the run.sh adds --sglang-tool-call-parser qwen25 / --reasoning-parser
#                  qwen3 so the mini-harness tool loop actually fires (the a3/b3 fix).
# BOTTLENECK     : throughput = min(inference, sandbox, harness); the [demo] records show
#                  which binds (qdepth > 0 harness, sandbox_wait_ms > 0 sandbox, engine_inflight).
# CONVERGENCE    : eval/deepscaler mean reward rises then plateaus.
# REWARD         : RLVR binary; GRPO group-relative.
# PREREQS        : run INSIDE agentcore_p4; AWS creds for the remote sandbox.
# RUNTIME/CKPT   : ~200 steps; final ckpt ~56 GB (disk caveat).
# ----------------------------------------------------------------------------
# Overridable, e.g.:  HARNESS_WORKERS=64 NR=401 bash 16_cpu_tasks_scale.sh
#   HARNESS_WORKERS / SANDBOX_SESSIONS / ROLLOUT_GPUS each move one tier alone.
#   RB*NS = 256 episodes per step keeps all 256 workers busy (stock driver: one batch in flight).
# ============================================================================
set -euo pipefail

CONV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO="$(cd "$CONV_DIR/.." && pwd)/demo_cpu_tasks"

: "${NR:=201}"; : "${RB:=32}"; : "${NS:=8}"; : "${TEMP:=1.0}"
: "${EVAL_INTERVAL:=20}"; : "${EVAL_NS:=4}"; : "${SAVE_INTERVAL:=100}"
: "${HARNESS_WORKERS:=256}"; : "${SANDBOX_SESSIONS:=256}"   # the decoupled tiers (the demo's point)
export SCALE=1 NR RB NS TEMP EVAL_INTERVAL EVAL_NS SAVE_INTERVAL HARNESS_WORKERS SANDBOX_SESSIONS

LOG="$DEMO/train.convergence_scale_$(date +%Y%m%d_%H%M%S).log"
echo ">>> converging demo_cpu_tasks SCALE=1 (deepscaler, W=$HARNESS_WORKERS M=$SANDBOX_SESSIONS)  NR=$NR RB=$RB NS=$NS steps=$((NR-1)) MODE=${MODE:-A}  log=$LOG"
bash "$DEMO/run.sh" 2>&1 | tee "$LOG"
