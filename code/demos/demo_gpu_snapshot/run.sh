#!/bin/bash
# demo_gpu_snapshot — application-level snapshot/resume across a GPU instance recycle.
#
# The only feasible "GPU instance snapshot" on AgentCore is application-level (serialize the
# workspace to a store you own); there is no VM/EC2 snapshot. See README.md and docs/gpu-snapshot.md.
# Runs offline in a second: no AWS, no GPU.
#
# Knobs: TARGET (steps to finish), RECYCLE_AFTER (instance recycle after N steps).
set -euo pipefail

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${DEMO_DIR}:${PYTHONPATH:-}"

export TARGET="${TARGET:-4}"
export RECYCLE_AFTER="${RECYCLE_AFTER:-2}"

cd "$DEMO_DIR"
exec python3 snapshot_demo.py
