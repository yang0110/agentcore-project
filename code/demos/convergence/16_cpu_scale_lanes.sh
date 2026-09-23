#!/bin/bash
# ============================================================================
# CONVERGENCE RUN 16 — demo_cpu_scale_lanes  (FAMILY=deepscaler; 4x4x4 lanes)
# ============================================================================
# TASK / DATASET : DeepScaleR competition-math (same task as 05/15). Multi-turn
#                  solve-in-sandbox, RLVR final-answer check. Train ~400 prompts,
#                  disjoint held-out eval.
# SUBSTRATE      : REMOTE managed AgentCore sandbox Runtime (SANDBOX=agentcore-runtime),
#                  K=2 CPU sandbox ARNs baked in; one PERSISTENT microVM session per lane.
# TOPOLOGY       : LANES=4 => 4 warm harness workers : 4 SGLang engines (1 GPU each) :
#                  4 microVMs, worker w pinned to engine w and its own session (1:1:1).
# PROPERTY (LIVE): scales the HARNESS TIER, not the sandbox. Rollouts are generated
#                  by a WARM pool of HARNESS_WORKERS in-process run_episode workers
#                  (harness_pool_generate.generate) that persist across rollout steps
#                  instead of being spun up per batch. The pool size is INDEPENDENT of
#                  sandbox concurrency (CONC is derived to match the SGLang engines).
#                  This is a mini-agent worker-pool warm-start ablation layered on a
#                  real training run — over ~200 steps the pool is re-used continuously.
# MODE           : pinned to A by the demo run.sh (the pool returns a fan-out sample
#                  list incompatible with Mode-B's fully-async sort key) — do NOT force B.
# PARSERS        : the run.sh adds --sglang-tool-call-parser qwen25 / --reasoning-parser
#                  qwen3 so the mini-harness tool loop actually fires (the a3/b3 fix).
# BOTTLENECK     : with a warm pool the harness setup cost is amortized; the trainer +
#                  inference engines dominate. Larger HARNESS_WORKERS => more concurrent
#                  episodes in flight (bounded by sandbox/engine capacity).
# CONVERGENCE    : eval/deepscaler mean reward rises then plateaus.
# REWARD         : RLVR binary; GRPO group-relative.
# PREREQS        : run INSIDE agentcore_p4; AWS creds for the remote sandbox.
# RUNTIME/CKPT   : ~200 steps; final ckpt ~56 GB (disk caveat).
# ----------------------------------------------------------------------------
# Overridable, e.g.:  LANES=2 NR=401 bash 16_cpu_scale_lanes.sh
#   LANES=1 collapses to one worker : one engine : one microVM. HARNESS_WORKERS alone
#   unbalances just the harness tier.
# NOTE: MODE is intentionally NOT exported here — the demo pins MODE=A internally.
# ============================================================================
set -euo pipefail

CONV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO="$(cd "$CONV_DIR/.." && pwd)/demo_cpu_scale_lanes"

: "${NR:=201}"; : "${RB:=8}"; : "${NS:=8}"; : "${TEMP:=1.0}"
: "${EVAL_INTERVAL:=20}"; : "${EVAL_NS:=4}"; : "${SAVE_INTERVAL:=100}"
: "${LANES:=4}"   # N workers : N engines : N microVMs (the demo's point)
export NR RB NS TEMP EVAL_INTERVAL EVAL_NS SAVE_INTERVAL LANES

LOG="$DEMO/train.convergence_$(date +%Y%m%d_%H%M%S).log"
echo ">>> converging demo_cpu_scale_lanes (deepscaler, LANES=$LANES)  NR=$NR RB=$RB NS=$NS steps=$((NR-1)) MODE=A  log=$LOG"
bash "$DEMO/run.sh" 2>&1 | tee "$LOG"
