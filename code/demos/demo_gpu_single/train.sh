#!/bin/bash
# Standalone TRAIN-TO-CONVERGENCE wrapper for this GPU-sandbox demo. Thin, documented layer over
# ./run.sh exposing only the knobs a client tunes for a real run, then delegating everything else.
# Run it like run.sh — INSIDE agentcore_p4, repo at /work. The KernelBench checker's GPU is the
# REMOTE g6 sandbox Runtime (or LOCAL_GPU=1 on host GPUs 6,7); slime keeps its 4 train + 4 infer GPUs.
#     docker exec agentcore_p4 bash -lc 'cd /work && bash code/demos/<this-demo>/train.sh'
#     # remote g6 Runtime: export AGENTCORE_SANDBOX_RUNTIME_ARN first; local fallback: LOCAL_GPU=1
#
# TOP-LEVEL KNOBS (env vars):
#   STEPS         training steps = real GRPO updates.        -> run.sh NR = STEPS+1
#   EVAL_EVERY    held-out eval cadence, in steps (0=off).   -> run.sh EVAL_INTERVAL
#   SANDBOX_CONC  concurrent episodes (pipeline width).      -> run.sh CONC ; S = CONC*num_engines
#
# NOTE — "sandbox count" (K distinct Runtimes) does NOT apply here: the GPU sandbox is a SINGLE
# GPU-backed AgentCore Runtime (AGENTCORE_SANDBOX_RUNTIME_ARN, singular), and demo_gpu_shared packs
# many concurrent envs onto that one GPU via its Ray farm rather than across multiple Runtimes.
# SANDBOX_CONC is therefore the only sandbox-width knob. Everything else (RB/NS/TEMP/LOCAL_GPU/
# CKPT_BASE/AWS creds/…) passes straight through to run.sh.
#
# CONVERGENCE NOTE: KernelBench keeps NR*RB <= 100 (the train split is 100 level_1 problems, one
# epoch), so STEPS <= (100/RB - 1) = 24 at the default RB=4 keeps the run inside one epoch with a
# single final checkpoint. Watch eval (level_2 fusions) via metrics.py and raise STEPS as needed.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

STEPS="${STEPS:-20}"                  # convergence budget (SMOKE: STEPS=2); keep STEPS*RB <= 100 for 1 epoch
EVAL_EVERY="${EVAL_EVERY:-10}"        # held-out eval cadence (steps; 0=off)
SANDBOX_CONC="${SANDBOX_CONC:-4}"     # concurrent episodes

export NR="$(( STEPS + 1 ))"          # run.sh runs range(1,NR) => STEPS real updates
export EVAL_INTERVAL="$EVAL_EVERY"
export CONC="$SANDBOX_CONC"

echo ">>> $(basename "$HERE") train-to-convergence: STEPS=$STEPS (NR=$NR) EVAL_EVERY=$EVAL_EVERY CONC=$SANDBOX_CONC LOCAL_GPU=${LOCAL_GPU:-0}"
[ $(( NR * ${RB:-4} )) -gt 100 ] && echo "!! note: NR*RB=$(( NR * ${RB:-4} )) > 100 (train split) => runs >1 epoch; extra epoch-boundary checkpoints will be written" || true
exec bash "$HERE/run.sh"
