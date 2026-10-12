#!/bin/bash
# The cluster's NCCL check (`hyperpod.py nccl-test`): nccl_bw.py on both GPU nodes, 8 ranks each, first over
# EFA (HP_EFA=1; fails if a node has no EFA device), then over TCP sockets (HP_EFA=0), same container.
# Output: logs/nccl_bw.out, one [nccl_bw] line per (transport, op, size).
#SBATCH --job-name=tbhp-nccl
#SBATCH --exclusive
set -euo pipefail
: "${HP_DEMO:?}" "${HP_SQSH:?}"
C_DEMO="/work/code/demos/$HP_DEMO"
MASTER_ADDR="$(srun --overlap -N1 -n1 -w "$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)" hostname -I \
  | tr ' ' '\n' | grep -m1 -F "${HP_SUBNET_PREFIX:-172.31.}")"
export MASTER_ADDR MASTER_PORT=29533
for EFA in 1 0; do
  echo ">>> HP_EFA=$EFA"
  HP_EFA=$EFA srun --overlap -N2 --ntasks-per-node=8 --container-image="$HP_SQSH" --container-name=slime \
    --container-mounts=/fsx/work:/work,/fsx:/fsx --container-writable --container-remap-root --no-container-mount-home \
    bash -c "source $C_DEMO/deploy/hyperpod/slurm/net_env.sh && export NET_DESC && python3 $C_DEMO/deploy/hyperpod/slurm/nccl_bw.py" \
    || echo "!! HP_EFA=$EFA failed (exit $?)"
done
