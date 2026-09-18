#!/bin/bash
# ============================================================================
# CONVERGENCE RUN 17 — demo_cpu_customruntime  (FAMILY=deepscaler; YOUR custom CPU Runtime)
# ============================================================================
# TASK / DATASET : DeepScaleR competition-math (same task as 05/15/16). Multi-turn
#                  solve-in-sandbox, RLVR final-answer check. Train ~400 prompts,
#                  disjoint held-out eval.
# SUBSTRATE      : the sandbox is YOUR OWN custom CPU AgentCore Runtime
#                  (SANDBOX=agentcore-runtime), NOT the managed Code Interpreter. The
#                  point of the demo is that the identical training loop runs unchanged
#                  against a custom-built sandbox image you deployed — the run.sh bakes
#                  the K=2 custom CPU sandbox ARNs, so no ARN needs to be set. (Override
#                  with AGENTCORE_SANDBOX_RUNTIME_ARNS to point at your own Runtime.)
# BOTTLENECK     : same as the standard CPU trainers — trainer + 2 SGLang engines; the
#                  custom Runtime backend behaves like the managed one for a Python checker.
# CONVERGENCE    : eval/deepscaler mean reward rises then plateaus.
# REWARD         : RLVR binary (final answer correct = 1.0 else 0.0); GRPO group-relative.
# PREREQS        : run INSIDE agentcore_p4; AWS creds; a live custom CPU sandbox Runtime
#                  (the baked ARNs, or your own via AGENTCORE_SANDBOX_RUNTIME_ARNS).
# RUNTIME/CKPT   : ~200 steps; final ckpt ~56 GB (disk caveat).
# TEARDOWN       : remote sandbox sessions close per-episode; nothing host-side.
# ----------------------------------------------------------------------------
# Overridable, e.g.:  NR=401 AGENTCORE_SANDBOX_RUNTIME_ARNS=<your-arns> bash 17_cpu_customruntime.sh
# ============================================================================
set -euo pipefail

CONV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO="$(cd "$CONV_DIR/.." && pwd)/demo_cpu_customruntime"

: "${NR:=201}"; : "${RB:=8}"; : "${NS:=8}"; : "${TEMP:=1.0}"
: "${EVAL_INTERVAL:=20}"; : "${EVAL_NS:=4}"; : "${SAVE_INTERVAL:=100}"
: "${MODE:=B}"; : "${CONC:=4}"
export NR RB NS TEMP EVAL_INTERVAL EVAL_NS SAVE_INTERVAL MODE CONC

LOG="$DEMO/train.convergence_$(date +%Y%m%d_%H%M%S).log"
echo ">>> converging demo_cpu_customruntime (deepscaler, custom CPU Runtime)  NR=$NR RB=$RB NS=$NS steps=$((NR-1)) MODE=$MODE  log=$LOG"
bash "$DEMO/run.sh" 2>&1 | tee "$LOG"
