#!/bin/bash
# Warm-start probe (HOST, no GPU/model/trainer): WHY Terminal-Bench's look-ahead pre-open works and what
# heartbeat adds. Runs the same M MockPolicy episodes cold / pre-opened / across an idle gap with and
# without heartbeat, against a stdlib CPU Runtime with a short (60 s) idle timeout and/or the managed
# Code Interpreter. Writes warmstart/logs/<tag>.{log,csv,summary.txt}.
#     bash warmstart/run_warmstart.sh                                   # deploy the Runtime if needed, run
#     BACKENDS=runtime,codeinterpreter MODES=cold,preopen,heartbeat M=24 GAP_S=150 bash warmstart/run_warmstart.sh
#     BACKENDS=local IDLE_S=3 GAP_S=5 bash warmstart/run_warmstart.sh   # offline, seconds
#     bash warmstart/run_warmstart.sh cleanup                           # delete the probe's Runtime
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$HERE/runtime.env"
export AWS_REGION="${AWS_REGION:-us-west-2}"
if [ "${1:-}" = cleanup ]; then
  python3 "$HERE/deploy_warmstart.py" --cleanup && rm -f "$ENV_FILE"; exit 0
fi
case "${BACKENDS:-runtime}" in *runtime*)
  if [ -z "${AGENTCORE_SANDBOX_RUNTIME_ARN:-}" ] && [ ! -s "$ENV_FILE" ]; then
    echo "[warmstart] no Runtime yet — creating it (reuses the shared acs-sandbox-cpu:latest image)" >&2
    python3 "$HERE/deploy_warmstart.py" | grep '^export ' | sed 's/^export //' > "$ENV_FILE"
  fi
  if [ -z "${AGENTCORE_SANDBOX_RUNTIME_ARN:-}" ]; then set -a; . "$ENV_FILE"; set +a; fi;;
esac
TAG="${RUN_TAG:-warmstart}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$HERE/logs"
OUT="$HERE/logs/$TAG" python3 "$HERE/warmstart_demo.py" 2>&1 | tee "$HERE/logs/$TAG.log"
echo "[warmstart] done — warmstart/logs/$TAG.{log,csv,summary.txt}; delete the Runtime: bash warmstart/run_warmstart.sh cleanup" >&2
