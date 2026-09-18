#!/bin/bash
# ============================================================================
# CONVERGENCE RUN 03 — demo_cpu_gsm8k  (FAMILY=gsm8k)
# ============================================================================
# TASK / DATASET : GSM8K grade-school word problems. The policy reasons step by
#                  step and emits a final numeric answer; the checker parses the
#                  final number and compares to gold.
# SUBSTRATE      : REMOTE managed AgentCore sandbox Runtime (SANDBOX=agentcore-runtime),
#                  K=2 CPU sandbox ARNs baked in. Numeric-match checker => CPU sandbox.
# BOTTLENECK     : Inference + trainer bound; the arithmetic checker is instant.
# CONVERGENCE    : eval/gsm8k mean reward (numeric-match rate) rises then plateaus.
#                  GSM8K is the EASIEST math family here — a good FIRST convergence
#                  run to validate the whole pipeline before the harder ones.
# REWARD         : RLVR binary (answer matches = 1.0 else 0.0); GRPO group-relative.
# PREREQS        : run INSIDE agentcore_p4; AWS creds for the remote sandbox.
# RUNTIME/CKPT   : ~200 steps; final ckpt ~56 GB (disk caveat).
# ----------------------------------------------------------------------------
# Overridable, e.g.:  SAVE_INTERVAL=1000000 bash 03_cpu_gsm8k.sh  (save only at end)
# ============================================================================
set -euo pipefail

CONV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO="$(cd "$CONV_DIR/.." && pwd)/demo_cpu_gsm8k"

: "${NR:=201}"; : "${RB:=8}"; : "${NS:=8}"; : "${TEMP:=1.0}"
: "${EVAL_INTERVAL:=20}"; : "${EVAL_NS:=4}"; : "${SAVE_INTERVAL:=100}"
: "${MODE:=B}"; : "${CONC:=4}"
export NR RB NS TEMP EVAL_INTERVAL EVAL_NS SAVE_INTERVAL MODE CONC

LOG="$DEMO/train.convergence_$(date +%Y%m%d_%H%M%S).log"
echo ">>> converging demo_cpu_gsm8k (GSM8K)  NR=$NR RB=$RB NS=$NS steps=$((NR-1)) MODE=$MODE  log=$LOG"
bash "$DEMO/run.sh" 2>&1 | tee "$LOG"
