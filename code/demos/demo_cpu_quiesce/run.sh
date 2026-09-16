#!/bin/bash
# demo_cpu_quiesce — quiesce/resume the sandbox across the weight-sync barrier.
#
# Unlike the training demos this is a COORDINATION-PATTERN demo (not a convergence run): it plays
# the naive -> heartbeat -> recover arc in seconds, offline, with no AWS and no GPU. See README.md
# and docs/demos-program.md ("quiesce / resume the sandbox").
#
# Knobs: IDLE (simulated AgentCore idle timeout, s), PAUSE (weight-sync stall, s), TICK (poll, s).
set -euo pipefail

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Import the vendored library copy (self-contained, layout B) + the demo-local modules.
export PYTHONPATH="${DEMO_DIR}:${PYTHONPATH:-}"

export IDLE="${IDLE:-4}"
export PAUSE="${PAUSE:-10}"
export TICK="${TICK:-0.5}"

cd "$DEMO_DIR"
exec python3 quiesce_demo.py
