#!/bin/bash
# ============================================================================
# CONVERGENCE RUN 19 — demo_cpu_instance_pool  (DATASET=deepscaler; warm m5 pool, SANDBOX_POOL=2)
# ============================================================================
# (was demo_cpu_bigimage -> demo_cpu_tasks VARIANT=bigimage, fresh instance per episode;
#  now a standalone demo that SHARES a warm instance pool, 2026-09-25.)
# TASK / DATASET : DeepScaleR competition-math (same task as 05). Multi-turn
#                  solve-in-sandbox, RLVR final-answer check.
# SUBSTRATE      : a ~4 GB sandbox image (torch CPU + scientific stack, over the 2 GB
#                  microVM cap) on CPU capacity-provider INSTANCES (m5.xlarge), Runtime
#                  acs_sandbox_bigcpu (SANDBOX=agentcore-ray-farm, SANDBOX_FARM=cpu).
# TOPOLOGY       : SANDBOX_POOL=N shared runtimeSessionIds = N warm m5 instances (cold boot
#                  ~80 s paid N times per run, not once per episode). Every episode goes to
#                  the LEAST-LOADED pool member and gets its own isolated sub_session.
# ISOLATION      : per-sub_session uid (20000+slot), private 0700 dir, prlimit, kill-all +
#                  wipe on close, path-escape checks, no AWS_* inherited. Episodes still
#                  share the instance's kernel, network namespace and CPU/RAM (see the
#                  demo README's "Isolation guarantees" and isolation_probe.py).
# BOTTLENECK     : trainer + SGLang engines; per-instance CPU/RAM if SANDBOX_POOL is too
#                  small for CONC x engines concurrent episodes.
# CONVERGENCE    : eval/deepscaler mean reward rises then plateaus (0.76 after 10 steps
#                  on the old fresh-instance path, 2026-09-21).
# REWARD         : RLVR binary; GRPO group-relative.
# PREREQS        : run INSIDE agentcore_p4; AWS creds; the capacity provider + Runtime from
#                  the demo's steps 2-4 (deploy/runtime.env holds AGENTCORE_SANDBOX_RUNTIME_ARN).
# RUNTIME/CKPT   : ~200 steps; final ckpt ~56 GB (disk caveat).
# TEARDOWN       : idle pool instances are reclaimed after the idle timeout (600 s);
#                  bash 6_cleanup.sh (DELETE_CP=1) deletes the Runtime (and capacity provider).
# ----------------------------------------------------------------------------
# Overridable, e.g.:  SANDBOX_POOL=4 CONC=8 NR=401 bash 19_cpu_instance_pool.sh
# ============================================================================
set -euo pipefail

CONV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO="$(cd "$CONV_DIR/.." && pwd)/demo_cpu_instance_pool"

# Pick up the deployed ARN if it is not already in the environment.
if [ -z "${AGENTCORE_SANDBOX_RUNTIME_ARN:-}" ] && [ -f "$DEMO/deploy/runtime.env" ]; then
  set -a; . "$DEMO/deploy/runtime.env"; set +a
fi

: "${NR:=201}"; : "${RB:=8}"; : "${NS:=8}"; : "${TEMP:=1.0}"
: "${EVAL_INTERVAL:=20}"; : "${EVAL_NS:=4}"; : "${SAVE_INTERVAL:=100}"
: "${MODE:=B}"; : "${CONC:=4}"
: "${DATASET:=deepscaler}"; : "${SANDBOX_POOL:=2}"
export NR RB NS TEMP EVAL_INTERVAL EVAL_NS SAVE_INTERVAL MODE CONC DATASET SANDBOX_POOL

LOG="$DEMO/train.convergence_pool${SANDBOX_POOL}_$(date +%Y%m%d_%H%M%S).log"
echo ">>> converging demo_cpu_instance_pool DATASET=$DATASET SANDBOX_POOL=$SANDBOX_POOL (warm m5 pool)  NR=$NR RB=$RB NS=$NS steps=$((NR-1)) MODE=$MODE  log=$LOG"
bash "$DEMO/run.sh" 2>&1 | tee "$LOG"
