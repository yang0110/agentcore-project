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
#   SANDBOX_RUNTIMES  # DISTINCT sandbox Runtimes K to spread    -> truncates AGENTCORE_SANDBOX_RUNTIME_ARNS
#                     episodes across (round-robin). unset=use      to its first K entries.
#                     run.sh's baked default (K=2).
#
# The two sandbox knobs are deliberately SEPARATE (they scale different things):
#   * SANDBOX_CONC = how many episodes run at once (throughput/staleness pipeline width);
#   * SANDBOX_RUNTIMES (K) = across how many physical Runtime endpoints those episodes are spread.
# CONC can exceed K (many concurrent sessions per Runtime) or be below it. See run.sh's header + the
# sandbox-concurrency ablation (docs/sandbox-concurrency-ablation.md).
#
# Everything else (RB/NS/TEMP/MODE/SANDBOX/DATASET/HEARTBEAT/CKPT_BASE/AWS creds/…) passes straight through to run.sh.
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
SANDBOX_RUNTIMES="${SANDBOX_RUNTIMES:-}"   # K: cap distinct sandbox Runtimes (unset => run.sh default)

export NR="$(( STEPS + 1 ))"               # run.sh runs range(1,NR) => STEPS real updates
export EVAL_INTERVAL="$EVAL_EVERY"
export CONC="$SANDBOX_CONC"

# K knob: cap the sandbox Runtime ARN list to the first K without duplicating the baked default —
# read it from run.sh when the caller has not supplied AGENTCORE_SANDBOX_RUNTIME_ARNS themselves.
if [ -n "$SANDBOX_RUNTIMES" ]; then
   _base="${AGENTCORE_SANDBOX_RUNTIME_ARNS:-$(grep -oE 'arn:aws:bedrock-agentcore:[A-Za-z0-9:_/-]+' "$HERE/run.sh" | paste -sd, -)}"
   export AGENTCORE_SANDBOX_RUNTIME_ARNS="$(printf '%s' "$_base" | tr ',' '\n' | sed '/^$/d' | head -n "$SANDBOX_RUNTIMES" | paste -sd, -)"
   _kmsg="K=$(printf '%s' "$AGENTCORE_SANDBOX_RUNTIME_ARNS" | tr ',' '\n' | sed '/^$/d' | grep -c .)"
else
   _kmsg="K=<run.sh default>"   # left unset => run.sh uses its own baked ARN list
fi

echo ">>> $(basename "$HERE") train-to-convergence: STEPS=$STEPS (NR=$NR) EVAL_EVERY=$EVAL_EVERY CONC=$SANDBOX_CONC $_kmsg"
exec bash "$HERE/run.sh"
