#!/bin/bash
# ============================================================================
# CONVERGENCE RUN 11 — demo_cpu_mixtask  (FAMILY=mixed)
# ============================================================================
# TASK / DATASET : MIXED-task RL. Each rollout batch spans SEVERAL task families
#                  (math / coding / SQL / tool-use / RAG) drawn from one merged
#                  pool; each prompt carries its own family-specific hidden checker.
#                  No harness/sandbox change — the run_episode loop is task-agnostic.
# SUBSTRATE      : REMOTE managed AgentCore sandbox Runtime (SANDBOX=agentcore-runtime),
#                  K=2 CPU sandbox ARNs baked in.
# BOTTLENECK     : Heterogeneous rewards across families. The interesting risk is
#                  GRPO gradient imbalance — easy families (GSM8K) saturate while
#                  hard ones (deepscaler/xlam) lag. Trainer + inference bound.
# CONVERGENCE    : the AGGREGATE eval reward rises, but read the PER-FAMILY eval
#                  breakdown in the log — convergence here means each family's
#                  curve is climbing, not just the average. Expect a longer run;
#                  consider NR=401.
# REWARD         : per-family RLVR, pooled; GRPO group-relative within each prompt.
# PREREQS        : run INSIDE agentcore_p4; AWS creds for the remote sandbox.
# RUNTIME/CKPT   : ~200 steps; final ckpt ~56 GB (disk caveat).
# ----------------------------------------------------------------------------
# Overridable, e.g.:  NR=401 RB=16 bash 11_cpu_mixtask.sh
# ============================================================================
set -euo pipefail

CONV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO="$(cd "$CONV_DIR/.." && pwd)/demo_cpu_mixtask"

: "${NR:=201}"; : "${RB:=8}"; : "${NS:=8}"; : "${TEMP:=1.0}"
: "${EVAL_INTERVAL:=20}"; : "${EVAL_NS:=4}"; : "${SAVE_INTERVAL:=100}"
: "${MODE:=B}"; : "${CONC:=4}"
export NR RB NS TEMP EVAL_INTERVAL EVAL_NS SAVE_INTERVAL MODE CONC

LOG="$DEMO/train.convergence_$(date +%Y%m%d_%H%M%S).log"
echo ">>> converging demo_cpu_mixtask (mixed families)  NR=$NR RB=$RB NS=$NS steps=$((NR-1)) MODE=$MODE  log=$LOG"
bash "$DEMO/run.sh" 2>&1 | tee "$LOG"
