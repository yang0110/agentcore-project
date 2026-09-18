#!/bin/bash
# ============================================================================
# CONVERGENCE RUN 10 — demo_cpu_mcp  (FAMILY=mcp)
# ============================================================================
# TASK / DATASET : MCP tool-use. The policy must call sandbox MCP tools
#                  (mcp__sandbox__*) with valid names/arguments to accomplish a
#                  task; the checker verifies the tool was invoked correctly and
#                  the resulting state/output matches gold.
# SUBSTRATE      : REMOTE managed AgentCore sandbox Runtime (SANDBOX=agentcore-runtime),
#                  K=2 CPU sandbox ARNs baked in. Tool calls EXECUTE in the sandbox.
# BOTTLENECK     : Multi-turn tool loop => more sandbox round-trips per episode
#                  than the math families, but each op is cheap; inference + trainer
#                  still dominate. Watch for the "undefined function" dead-reward
#                  trap (hallucinated tool names) early in training.
# CONVERGENCE    : eval/mcp mean reward (correct tool invocation) rises then plateaus.
# REWARD         : RLVR (valid tool call + correct effect = 1.0 else 0.0); GRPO group-relative.
# PREREQS        : run INSIDE agentcore_p4; AWS creds for the remote sandbox.
# RUNTIME/CKPT   : ~200 steps; final ckpt ~56 GB (disk caveat).
# ----------------------------------------------------------------------------
# Overridable, e.g.:  TEMP=1.1 bash 10_cpu_mcp.sh
# ============================================================================
set -euo pipefail

CONV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO="$(cd "$CONV_DIR/.." && pwd)/demo_cpu_mcp"

: "${NR:=201}"; : "${RB:=8}"; : "${NS:=8}"; : "${TEMP:=1.0}"
: "${EVAL_INTERVAL:=20}"; : "${EVAL_NS:=4}"; : "${SAVE_INTERVAL:=100}"
: "${MODE:=B}"; : "${CONC:=4}"
export NR RB NS TEMP EVAL_INTERVAL EVAL_NS SAVE_INTERVAL MODE CONC

LOG="$DEMO/train.convergence_$(date +%Y%m%d_%H%M%S).log"
echo ">>> converging demo_cpu_mcp (MCP tool-use)  NR=$NR RB=$RB NS=$NS steps=$((NR-1)) MODE=$MODE  log=$LOG"
bash "$DEMO/run.sh" 2>&1 | tee "$LOG"
