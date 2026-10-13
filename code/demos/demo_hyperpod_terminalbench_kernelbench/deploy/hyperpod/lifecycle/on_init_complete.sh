#!/bin/bash
# OUR hook (not upstream): HyperPod's OnInitComplete for every node, as root, after HyperPod itself has set up
# Slurm (SlurmConfig + Orchestrator.Slurm "Managed": no OnCreate / provisioning_parameters.json any more) and
# mounted FSx for Lustre at /fsx (InstanceStorageConfigs.FsxLustreConfig).
#   * enroot + pyxis: the vendored upstream script, which skips whatever the HyperPod AMI already provides and
#     points enroot's data/cache at the local NVMe;
#   * /fsx/work and /fsx/enroot owned by ubuntu (hyperpod.py runs everything as ubuntu over SSM).
set -euxo pipefail
cd "$(dirname "$(readlink -f "$0")")"
bash utils/install_enroot_pyxis.sh
if mountpoint -q /fsx; then
  mkdir -p /fsx/work /fsx/enroot
  chown ubuntu:ubuntu /fsx/work /fsx/enroot
else
  echo "!! /fsx is not mounted" >&2
  exit 1
fi
