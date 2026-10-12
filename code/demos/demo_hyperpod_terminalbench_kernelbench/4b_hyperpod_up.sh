#!/bin/bash
# Step 4b/6 — the SageMaker HyperPod cluster slime runs on (HOST, real AWS $$$: 2 x ml.g6e.48xlarge + FSx
# for Lustre bill while they exist; `bash 6_cleanup.sh` removes them). Run after steps 2-4 (the sandbox tiers).
#   4b.1  the trainer image: slimerl/slime:agentcore_p4-20261006 (never :latest) + the EFA layer
#         (Dockerfile.efa; HP_EFA=0 skips it) -> ECR
#   4b.2  bucket + node IAM role + cluster SG + FSx + vendored lifecycle scripts + the Slurm cluster (~25 min)
#   4b.3  SG rules: sandbox sessions -> the cluster's adapters, the cluster -> the m5 / g6 agents
#   4b.4  enroot-import the image once to /fsx/enroot/<tag>.sqsh
#   4b.5  NCCL all_reduce + weight-push bandwidth between the 2 nodes, EFA then TCP sockets (logs/nccl_bw.out)
# Needs the AWS session-manager-plugin on this box (4b.4 and step 5 talk to the controller over SSM).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/deploy/steps_common.sh"
need_host
cd "$HERE"
HP="python3 deploy/hyperpod/hyperpod.py"
log "step 4b.1: trainer image (+EFA) -> ECR"
$HP image
log "step 4b.2: HyperPod cluster (Slurm: controller + gpu x${HP_GPU_COUNT:-2} ${HP_GPU_TYPE:-ml.g6e.48xlarge}) + FSx"
$HP up
log "step 4b.3: SG rules between the cluster and the sandbox tiers"
$HP wire
log "step 4b.4: import the trainer image to FSx (once)"
$HP import-image
log "step 4b.5: NCCL between the nodes: EFA vs TCP sockets"
$HP nccl-test
log "done — next: bash 1_prepare_data.sh (if not yet), then bash 5_run.sh"
