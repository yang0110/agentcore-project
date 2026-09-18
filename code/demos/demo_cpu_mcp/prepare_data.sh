#!/bin/bash
# Download + prepare the demo_cpu_mcp dataset (Team-ACE/ToolACE (MCP-style JSON tool schemas)) into data/. Run INSIDE agentcore_p4:
#     docker exec agentcore_p4 bash /work/code/demos/demo_cpu_mcp/prepare_data.sh
# RLVR (verifiable reward): order-insensitive, arg-normalized gold-call match — no LLM judge. Idempotent. Builds mcp (300/40).
# The builder is vendored (agentcore_slime_demo/build_real_datasets.py) so this folder is self-contained.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
B="python3 $DIR/agentcore_slime_demo/build_real_datasets.py"
mkdir -p "$DIR/data"
$B --dataset mcp --limit 300 --offset 0 --out "$DIR/data/mcp_train.jsonl"
$B --dataset mcp --limit 40 --offset 300  --out "$DIR/data/mcp_eval.jsonl"
echo "done -> $DIR/data : mcp (300/40)"
