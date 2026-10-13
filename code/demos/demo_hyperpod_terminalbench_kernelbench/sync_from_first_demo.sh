#!/bin/bash
# Keep this demo's AgentCore + sandbox side IDENTICAL to demo_terminalbench_kernelbench (HOST or container).
# This demo = that demo + slime on HyperPod; only the files in HP_OWN below are its own. Everything else
# (sandbox clients, m5 agent, task corpus + route maps, KernelBench, data, deploy helpers, shared AWS state
# deploy/{m5,g6}_host.json + harness_cli.json) is copied from the first demo, byte for byte.
#     bash sync_from_first_demo.sh --check   # list what differs, change nothing (exit 1 if anything does)
#     bash sync_from_first_demo.sh           # copy the first demo's sandbox side over this one
# Never run it while a 5_run.sh of this demo is running (bash reads scripts as it goes).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$(cd "$HERE/../demo_terminalbench_kernelbench" && pwd)"
CHECK=0; [ "${1:-}" = "--check" ] && CHECK=1

# This demo's own files (HyperPod launch of slime): never overwritten.
HP_OWN=(run.sh 5_run.sh 6_cleanup.sh 4b_hyperpod_up.sh 1_prepare_data.sh 4_deploy_runtime.sh prepare_qwen35.sh
        deploy/demo.env deploy/hyperpod README.md PROGRESS.md .gitignore sync_from_first_demo.sh)
EXCL=(--exclude=__pycache__ --exclude='*.pyc' --exclude=logs/ --exclude=ckpt/ --exclude='data_bak_*'
      --exclude='*.bak_*' --exclude=log-summary/)
for f in "${HP_OWN[@]}"; do EXCL+=("--exclude=/$f"); done

if [ "$CHECK" = 1 ]; then
  out="$(rsync -a --delete --dry-run --itemize-changes --checksum "${EXCL[@]}" "$SRC/" "$HERE/" | grep -v '^\.d' || true)"
  if [ -z "$out" ]; then echo "identical: sandbox side matches $SRC"; exit 0; fi
  echo "differs from $SRC (> = would be copied, *deleting = only here):"; echo "$out"; exit 1
fi
rsync -a --delete --checksum --itemize-changes "${EXCL[@]}" "$SRC/" "$HERE/" | grep -v '^\.d' || true
echo "synced from $SRC (kept: ${HP_OWN[*]})"
