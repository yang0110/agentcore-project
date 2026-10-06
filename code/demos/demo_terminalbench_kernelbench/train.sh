#!/bin/bash
# Standalone TRAIN-TO-CONVERGENCE wrapper for this demo. A thin, documented layer over ./run.sh that
# exposes only the knobs a client tunes for a real (non-smoke) run, then delegates everything else.
# Run it exactly like run.sh — INSIDE the agentcore_p4 container, with the repo at /work:
#     docker exec agentcore_p4 bash -lc 'cd /work && bash code/demos/<this-demo>/train.sh'
#
# TOP-LEVEL KNOBS (env vars):
#   STEPS             training steps = real GRPO updates.        -> run.sh NR = STEPS+1
#   EVAL_EVERY        held-out eval cadence, in steps (0=off).   -> run.sh EVAL_INTERVAL
#   SANDBOX_CONC      concurrent episodes (pipeline width).      -> run.sh CONC ; S = CONC*num_engines
#
# There is no Runtime-count knob: each episode goes to its own task's Runtime (strict image routing).
#
# Everything else (RB/NS/TEMP/SANDBOX/DATASET/HEARTBEAT/CKPT_BASE/AWS creds/…) passes straight through to run.sh.
#
# CONVERGENCE NOTE: the right STEPS is task-dependent — watch eval/<task> (metrics.py) and raise it.
# slime force-saves at the final step AND at every epoch boundary; staying at STEPS <= (train_rows/RB - 1)
# keeps the run inside one epoch so the ONLY checkpoint written is the final one. Beyond that you get
# extra epoch-boundary checkpoints (more disk) — grow the train split (prepare_data.sh --limit) for a
# longer single-epoch run.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

STEPS="${STEPS:-50}"                       # convergence budget (SMOKE: STEPS=2)
EVAL_EVERY="${EVAL_EVERY:-10}"             # held-out eval cadence (steps; 0=off)
SANDBOX_CONC="${SANDBOX_CONC:-4}"          # concurrent episodes

export NR="$(( STEPS + 1 ))"               # run.sh runs range(1,NR) => STEPS real updates
export EVAL_INTERVAL="$EVAL_EVERY"
export CONC="$SANDBOX_CONC"


echo ">>> $(basename "$HERE") train-to-convergence: STEPS=$STEPS (NR=$NR) EVAL_EVERY=$EVAL_EVERY CONC=$SANDBOX_CONC"
exec bash "$HERE/run.sh"
