#!/bin/bash
# The RL run on the HyperPod cluster: 2 x ml.g6e.48xlarge, one Ray node each, in the trainer image
# (pyxis/enroot, /fsx/enroot/slime.sqsh). Submitted by `hyperpod.py submit TAG` (5_run.sh), which passes
# -p rl -N 2 --exclusive and the run knobs in the environment (srun forwards them into the container).
#
# Roles: slime puts the actor on the first ACTOR_GPUS bundles of ONE placement group sorted by node IP, so the
# LOWER-IP node trains (8 GPUs, Megatron) and the other serves inference (8 GPUs, SGLang). The Ray head, the
# job driver and run.sh go on the inference node, next to the engines and the sandbox-facing rollout tier.
#SBATCH --job-name=tbhp
#SBATCH --exclusive
set -euo pipefail
: "${RUN_TAG:?}"
DEMO="${HP_DEMO:?}"                                # this demo's folder name (set by hyperpod.py submit)
C_DEMO="/work/code/demos/$DEMO"                    # in the container (= /fsx/work/... on the nodes)
LOG="/fsx/work/code/demos/$DEMO/logs/$RUN_TAG.log"
IMG="${HP_SQSH:?}"                                 # /fsx/enroot/<image tag>.sqsh (hyperpod.py import-image)
CT=(--container-image="$IMG" --container-name=slime --container-mounts=/fsx/work:/work,/fsx:/fsx
    --container-writable --container-remap-root --no-container-mount-home)

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[ "${#NODES[@]}" = 2 ] || { echo "!! expected 2 nodes, got ${NODES[*]}" >&2; exit 2; }
# Each node's VPC address, picked exactly as ray_node.sh picks its --node-ip-address (slime sorts by that).
ipof() { srun --overlap -N1 -n1 -w "$1" hostname -I | tr ' ' '\n' | grep -m1 -F "${HP_SUBNET_PREFIX:-172.31.}"; }   # = net_env.sh NODE_IP
mapfile -t BY_IP < <(for n in "${NODES[@]}"; do echo "$(ipof "$n") $n"; done | sort -t. -k1,1n -k2,2n -k3,3n -k4,4n)
TRAIN_NODE="${BY_IP[0]#* }"; INFER_NODE="${BY_IP[1]#* }"; INFER_IP="${BY_IP[1]%% *}"
echo ">>> [$DEMO] job $SLURM_JOB_ID: trainer=$TRAIN_NODE (${BY_IP[0]%% *})  inference+head=$INFER_NODE ($INFER_IP)" | tee -a "$LOG"

_cleanup() {
  set +e
  [ -n "${HP_BUCKET:-}" ] && aws s3 cp --only-show-errors "$LOG" "s3://$HP_BUCKET/logs/$RUN_TAG.log"
  kill "${RAY_PIDS[@]}" 2>/dev/null; wait 2>/dev/null
}
trap _cleanup EXIT
RAY_PIDS=()
srun --overlap -N1 -n1 -w "$INFER_NODE" "${CT[@]}" bash "$C_DEMO/deploy/hyperpod/slurm/ray_node.sh" head &
RAY_PIDS+=($!)
for _ in $(seq 60); do   # the head must be up before the worker joins
  srun --overlap -N1 -n1 -w "$INFER_NODE" "${CT[@]}" ray status --address "$INFER_IP:6379" >/dev/null 2>&1 && break
  sleep 10
done
srun --overlap -N1 -n1 -w "$TRAIN_NODE" "${CT[@]}" bash "$C_DEMO/deploy/hyperpod/slurm/ray_node.sh" worker "$INFER_IP" &
RAY_PIDS+=($!)
for _ in $(seq 60); do   # both nodes' 16 GPUs registered
  G=$(srun --overlap -N1 -n1 -w "$INFER_NODE" "${CT[@]}" python3 -c \
      "import ray; ray.init(address='$INFER_IP:6379', logging_level='ERROR'); print(int(ray.cluster_resources().get('GPU', 0)))" 2>/dev/null | tail -1)
  [ "${G:-0}" -ge 16 ] && break
  sleep 10
done
[ "${G:-0}" -ge 16 ] || { echo "!! Ray sees ${G:-0} GPUs, expected 16" | tee -a "$LOG" >&2; exit 3; }
echo ">>> Ray up: $G GPUs on 2 nodes" | tee -a "$LOG"

RC=0
srun --overlap -N1 -n1 -w "$INFER_NODE" "${CT[@]}" \
  bash -lc "cd /work && set -o pipefail && bash $C_DEMO/run.sh 2>&1 | tee -a $LOG" || RC=$?
echo ">>> run.sh exit $RC" | tee -a "$LOG"
exit "$RC"
