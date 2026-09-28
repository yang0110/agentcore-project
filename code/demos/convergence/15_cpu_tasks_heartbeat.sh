#!/bin/bash
# ============================================================================
# CONVERGENCE RUN 15 — demo_cpu_tasks HEARTBEAT=1  (DATASET=deepscaler; keep-warm across the barrier)
# (was demo_cpu_heartbeat; heartbeat is now an opt-in knob of every demo, merged 2026-09-25)
# ============================================================================
# TASK / DATASET : DeepScaleR competition-math (same task pipeline as 05). The
#                  policy solves a hard math problem in the sandbox over up to a
#                  few turns; RLVR checks the final answer. Train pool ~400 prompts,
#                  disjoint held-out eval.
# SUBSTRATE      : YOUR OWN AgentCore sandbox Runtime (SANDBOX=agentcore-runtime, set below),
#                  K=2 CPU sandbox ARNs (deploy/runtime.env or the run.sh defaults). The heartbeat
#                  only helps our own Runtimes (the managed Code Interpreter was removed).
# PROPERTY (LIVE): this is the plain CPU trainer PLUS a session keep-warm layer.
#                  HEARTBEAT=1 wraps every episode's sandbox in
#                  KeptSandbox + a process-wide SessionKeeper heartbeat, so the
#                  remote sandbox session survives the WEIGHT-SYNC BARRIER between
#                  rollout steps (generation goes silent while weights sync; without
#                  the heartbeat the session would idle out). HEARTBEAT=0
#                  is 05_cpu_deepscaler.sh on your own Runtime.
#                  Over a 200-step convergence run there are ~200 barriers, so the
#                  keep-warm path is exercised continuously — this is where it earns
#                  its keep vs. the 10-step smoke.
# BOTTLENECK     : trainer + 2 SGLang engines (pure-Python answer checker is cheap);
#                  the keep-warm heartbeat is off the hot path (idle-time pings only).
# CONVERGENCE    : eval/deepscaler mean reward (answer-correct rate) rises then plateaus.
# REWARD         : RLVR binary (final answer correct = 1.0 else 0.0); GRPO group-relative.
# PREREQS        : run INSIDE agentcore_p4; AWS creds for the remote sandbox.
# RUNTIME/CKPT   : ~200 steps; final ckpt ~56 GB (disk caveat).
# TEARDOWN       : sessions close per-episode; the SessionKeeper thread ends with the run.
# ----------------------------------------------------------------------------
# Overridable, e.g.:  NR=401 HEARTBEAT_IDLE=900 bash 15_cpu_tasks_heartbeat.sh
#   HEARTBEAT=0  -> disable keep-warm (== 05_cpu_deepscaler.sh with SANDBOX=agentcore-runtime)
# ============================================================================
set -euo pipefail

CONV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO="$(cd "$CONV_DIR/.." && pwd)/demo_cpu_tasks"

: "${NR:=201}"; : "${RB:=8}"; : "${NS:=8}"; : "${TEMP:=1.0}"
: "${EVAL_INTERVAL:=20}"; : "${EVAL_NS:=4}"; : "${SAVE_INTERVAL:=100}"
: "${MODE:=B}"; : "${CONC:=4}"
: "${DATASET:=deepscaler}"
: "${HEARTBEAT:=1}"   # keep-warm across the weight-sync barrier (the point of this run)
: "${SANDBOX:=agentcore-runtime}"   # heartbeat only matters on your own Runtime
export NR RB NS TEMP EVAL_INTERVAL EVAL_NS SAVE_INTERVAL MODE CONC DATASET HEARTBEAT SANDBOX

LOG="$DEMO/train.convergence_$(date +%Y%m%d_%H%M%S).log"
echo ">>> converging demo_cpu_tasks HEARTBEAT=$HEARTBEAT SANDBOX=$SANDBOX ($DATASET)  NR=$NR RB=$RB NS=$NS steps=$((NR-1)) MODE=$MODE  log=$LOG"
bash "$DEMO/run.sh" 2>&1 | tee "$LOG"
