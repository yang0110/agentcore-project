#!/usr/bin/env bash
# Standalone, offline harness-scaling demo — sweep the number of warm harness workers W and show
#   throughput = min(inference, sandbox, harness)
# No AWS, no GPU, no network. Runs in a few seconds. See README.md for the diagram + the reading.
#
#   ./run.sh                       # default sweep W in {1,2,4,6,8,12}, CINF=4, CSBX=8
#   WORKERS=1,4,16 CINF=8 ./run.sh # lift the inference ceiling and watch the plateau rise
#
# Env knobs (all optional): EPISODES, WORKERS, CINF, CSBX, TINF, TSBX, JSON_OUT.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# Ensure the DeepScaleR data is present; it ships in data/, but rebuild if missing.
if [[ ! -s data/deepscaler_train.jsonl ]]; then
  echo "data/deepscaler_train.jsonl missing — run ./prepare_data.sh inside the agentcore_p4 container" >&2
  exit 1
fi

PYTHONPATH="$DIR:${PYTHONPATH:-}" python3 harness_scaling_demo.py \
  --episodes "${EPISODES:-48}" \
  --workers "${WORKERS:-1,2,4,6,8,12}" \
  --cinf "${CINF:-4}" --csbx "${CSBX:-8}" \
  --tinf "${TINF:-0.10}" --tsbx "${TSBX:-0.05}" \
  ${JSON_OUT:+--json "$JSON_OUT"}
