#!/bin/bash
# ============================================================================
# CONVERGENCE RUN 01 — demo_cpu_tasks  (DATASET=mbpp / MBPP)
# ============================================================================
# TASK / DATASET : MBPP-style Python programming. The policy writes a function,
#                  runs it against hidden unit tests in the sandbox, and repairs
#                  it over up to a few turns (our multi-turn run_episode harness).
#                  Train pool ~500 prompts; disjoint held-out eval set.
# SUBSTRATE      : our own VPC CPU sandbox Runtime (SANDBOX=agentcore-runtime, steps 2-4): one
#                  fresh microVM session per episode, loop in the session. Pure-Python /
#                  IO checker => CPU sandbox (no GPU).
# BOTTLENECK     : The runtime-count ablation found the sandbox is NOT the
#                  bottleneck for a pure-Python checker — the 2 SGLang inference
#                  engines + the trainer are. Sandbox count K flat across 1/2/4.
# CONVERGENCE    : eval/mbpp mean reward (unit-test pass rate) rises then plateaus.
# REWARD         : RLVR binary (tests pass = 1.0 else 0.0); GRPO group-relative.
# PREREQS        : run INSIDE agentcore_p4; AWS creds (AWS_BEARER_TOKEN_BEDROCK,
#                  region) for the remote sandbox. Override endpoints with
#                  AGENTCORE_SANDBOX_RUNTIME_ARNS if yours differ.
# RUNTIME/CKPT   : ~200 optimizer steps; final ckpt ~56 GB (see disk caveat below).
# TEARDOWN       : remote sandbox sessions close per-episode; nothing host-side.
# ----------------------------------------------------------------------------
# All knobs below are overridable from the environment, e.g.:
#   NR=401 RB=16 SAVE_INTERVAL=1000000 bash 01_cpu_mbpp.sh
# SAVE_INTERVAL=100 keeps periodic ckpts (~56 GB each!). Set it to 1000000 to
# save only at the end if you only care about the eval curve. See README.
# ============================================================================
set -euo pipefail

CONV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO="$(cd "$CONV_DIR/.." && pwd)/demo_cpu_tasks"

: "${NR:=201}"            # real train steps = NR-1 = 200
: "${RB:=8}"             # prompts per step
: "${NS:=8}"             # GRPO group size
: "${TEMP:=1.0}"         # keep intra-group reward variance (GRPO signal)
: "${EVAL_INTERVAL:=20}" # held-out eval every 20 steps
: "${EVAL_NS:=4}"        # samples per eval prompt
: "${SAVE_INTERVAL:=100}"# periodic checkpoints — DISK CAVEAT: ~56 GB each
: "${CONC:=4}"           # sandbox/episode concurrency
: "${DATASET:=mbpp}"   # which dataset of demo_cpu_tasks to train
export DATASET NR RB NS TEMP EVAL_INTERVAL EVAL_NS SAVE_INTERVAL CONC

LOG="$DEMO/train.convergence_${DATASET}_$(date +%Y%m%d_%H%M%S).log"
echo ">>> converging demo_cpu_tasks DATASET=$DATASET (coding/MBPP)  NR=$NR RB=$RB NS=$NS steps=$((NR-1))  log=$LOG"
bash "$DEMO/run.sh" 2>&1 | tee "$LOG"
