#!/bin/bash
# ============================================================================
# CONVERGENCE RUN 02 — demo_cpu_math  (FAMILY=math / Hendrycks MATH)
# ============================================================================
# TASK / DATASET : Hendrycks MATH competition problems. The policy produces a
#                  worked solution ending in a \boxed{} answer; the checker
#                  extracts and normalizes it and compares to the gold answer.
# SUBSTRATE      : REMOTE managed AgentCore sandbox Runtime (SANDBOX=agentcore-runtime),
#                  K=2 CPU sandbox ARNs baked in. Answer-match checker is trivial
#                  Python => CPU sandbox.
# BOTTLENECK     : Inference + trainer bound (checker is near-instant). Longer
#                  chains-of-thought here stress SGLang decode more than the sandbox.
# CONVERGENCE    : eval/math mean reward (answer-match rate) rises then plateaus.
# REWARD         : RLVR binary (answer matches = 1.0 else 0.0); GRPO group-relative.
# PREREQS        : run INSIDE agentcore_p4; AWS creds for the remote sandbox.
# RUNTIME/CKPT   : ~200 steps; final ckpt ~56 GB (disk caveat).
# ----------------------------------------------------------------------------
# Overridable, e.g.:  NR=401 TEMP=1.1 bash 02_cpu_math.sh
# ============================================================================
set -euo pipefail

CONV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO="$(cd "$CONV_DIR/.." && pwd)/demo_cpu_math"

: "${NR:=201}"; : "${RB:=8}"; : "${NS:=8}"; : "${TEMP:=1.0}"
: "${EVAL_INTERVAL:=20}"; : "${EVAL_NS:=4}"; : "${SAVE_INTERVAL:=100}"
: "${MODE:=B}"; : "${CONC:=4}"
export NR RB NS TEMP EVAL_INTERVAL EVAL_NS SAVE_INTERVAL MODE CONC

LOG="$DEMO/train.convergence_$(date +%Y%m%d_%H%M%S).log"
echo ">>> converging demo_cpu_math (Hendrycks MATH)  NR=$NR RB=$RB NS=$NS steps=$((NR-1)) MODE=$MODE  log=$LOG"
bash "$DEMO/run.sh" 2>&1 | tee "$LOG"
