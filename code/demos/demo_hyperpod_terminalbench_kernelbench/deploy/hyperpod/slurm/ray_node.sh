#!/bin/bash
# One Ray node of the 2-node job, INSIDE the trainer container (train.sbatch starts one per GPU node):
#     ray_node.sh head            # the inference node: Ray head, the job driver and (usually) the RolloutManager
#     ray_node.sh worker HEAD_IP  # the trainer node
# Everything exported here is inherited by the Ray workers on THIS node, which is how per-node values reach
# them: ADAPTER_PUBLIC_HOST (the address the sandbox agent loops dial back, so it is right wherever the
# RolloutManager lands; run.sh leaves it out of the job's runtime_env) and the NCCL transport (net_env.sh:
# EFA by default, HP_EFA=0 for TCP sockets).
set -euo pipefail
ROLE="$1"; HEAD_IP="${2:-}"
# shellcheck disable=SC1091
source "$(dirname "$0")/net_env.sh"
export ADAPTER_PUBLIC_HOST="$NODE_IP"
echo ">>> ray $ROLE on $(hostname) ip=$NODE_IP iface=$NODE_IFACE gpus=$(nvidia-smi -L | wc -l) nccl=$NET_DESC"
pkill -9 sglang 2>/dev/null || true; ray stop --force >/dev/null 2>&1 || true
if [ "$ROLE" = head ]; then
  exec ray start --head --node-ip-address "$NODE_IP" --num-gpus 8 --disable-usage-stats \
       --dashboard-host=0.0.0.0 --dashboard-port=8265 --block
fi
exec ray start --address "$HEAD_IP:6379" --node-ip-address "$NODE_IP" --num-gpus 8 --disable-usage-stats --block
