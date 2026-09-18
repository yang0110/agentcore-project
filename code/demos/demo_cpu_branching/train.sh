#!/bin/bash
# Standalone TRAIN-TO-CONVERGENCE wrapper for demo_cpu_branching. A thin, documented layer over
# ./run.sh that exposes only the knobs a client tunes for a real (non-smoke) run, then delegates.
# Run it exactly like run.sh — INSIDE the agentcore_p4 container, with the repo at /work:
#     docker exec agentcore_p4 bash -lc 'cd /work && \
#        ADAPTER_PUBLIC_HOST=10.0.x.y AGENTCORE_SANDBOX_RUNTIME_ARN=arn:... \
#        bash code/demos/demo_cpu_branching/train.sh'
#
# TOP-LEVEL KNOBS (env vars):
#   STEPS         training steps = real GRPO updates.            -> run.sh NR = STEPS+1
#   EVAL_EVERY    held-out eval cadence, in steps (0=off).       -> run.sh EVAL_INTERVAL
#   SANDBOX_CONC  concurrent episodes (pipeline width).          -> run.sh CONC ; S = CONC*num_engines
#   FORK_THRESH   adapter fork threshold in tokens (lower =>     -> run.sh BRANCHING_FORK_THRESHOLD
#                 more compaction forks => more fan-out).
#
# REQUIRED (pass straight through to run.sh; no sane default):
#   ADAPTER_PUBLIC_HOST            private VPC IP the sandbox microVM dials to reach the adapter.
#   AGENTCORE_SANDBOX_RUNTIME_ARN  a CLI-capable sandbox Runtime (node+claude baked; deploy/Dockerfile.sandbox.cli).
#
# Everything else (RB/NS/TEMP/SANDBOX/CKPT_BASE/AWS creds/ADAPTER_PORT/…) passes through to run.sh.
# NOTE: MODE is forced to A inside run.sh (fully-async can't handle the fan-out shape).
#
# CONVERGENCE NOTE: watch eval/swe (metrics.py) and raise STEPS. Staying at STEPS <= (train_rows/RB - 1)
# keeps the run inside one epoch so the ONLY checkpoint written is the final one; grow the train
# split (build_swe_dataset.py --repeat) for a longer single-epoch run.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

STEPS="${STEPS:-50}"                 # convergence budget (SMOKE: STEPS=2)
EVAL_EVERY="${EVAL_EVERY:-10}"       # held-out eval cadence (steps; 0=off)
SANDBOX_CONC="${SANDBOX_CONC:-4}"    # concurrent episodes

export NR="$(( STEPS + 1 ))"         # run.sh runs range(1,NR) => STEPS real updates
export EVAL_INTERVAL="$EVAL_EVERY"
export CONC="$SANDBOX_CONC"
[ -n "${FORK_THRESH:-}" ] && export BRANCHING_FORK_THRESHOLD="$FORK_THRESH"

echo ">>> $(basename "$HERE") train-to-convergence: STEPS=$STEPS (NR=$NR) EVAL_EVERY=$EVAL_EVERY CONC=$SANDBOX_CONC"
exec bash "$HERE/run.sh"
