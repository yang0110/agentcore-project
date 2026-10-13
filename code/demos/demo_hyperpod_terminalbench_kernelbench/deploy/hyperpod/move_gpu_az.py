#!/usr/bin/env python3
"""Move the cluster's GPU nodes to another AZ when the current one has no capacity (HOST):

    python3 deploy/hyperpod/move_gpu_az.py us-west-2c [ml.g6e.48xlarge]

HyperPod can't change an instance group's OverrideVpcConfig in place, so this ADDS a group
`gpu-<az>` (same partition, same lifecycle + storage, OverrideVpcConfig = the AZ's egress subnet + the
cluster SG; both nodes stay in one AZ, as EFA needs) and scales every other GPU group to 0. The controller
and FSx stay where they are (FSx for Lustre mounts across AZs). Records the active group in hyperpod.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

import boto3

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import agentcore_deploy as d  # noqa: E402
import hyperpod as h  # noqa: E402


def main() -> int:
    az = sys.argv[1]
    itype = sys.argv[2] if len(sys.argv) > 2 else h.GPU_TYPE
    st = h.load_state()
    sm = boto3.client("sagemaker", region_name=d.REGION)
    c = sm.describe_cluster(ClusterName=st["name"])
    subnet = d.ensure_egress_subnets(st["vpc"], [az])[0]
    new = f"gpu-{az.replace(d.REGION, '').strip('-') or az}"     # e.g. gpu-c for us-west-2c
    groups, have = [], False
    for g in c["InstanceGroups"]:
        spec = {k: g[k] for k in ("InstanceGroupName", "InstanceType", "LifeCycleConfig", "ExecutionRole",
                                  "ThreadsPerCore", "InstanceStorageConfigs", "SlurmConfig", "OverrideVpcConfig")
                if k in g}
        spec["InstanceCount"] = g["TargetCount"]
        if g.get("SlurmConfig", {}).get("NodeType") == "Compute":
            spec["InstanceCount"] = h.GPU_COUNT if g["InstanceGroupName"] == new else 0
            have |= g["InstanceGroupName"] == new
            template = spec
        groups.append(spec)
    if not have:
        groups.append(template | {"InstanceGroupName": new, "InstanceType": itype, "InstanceCount": h.GPU_COUNT,
                                  "OverrideVpcConfig": {"SecurityGroupIds": [st["sg"]], "Subnets": [subnet]}})
    sm.update_cluster(ClusterName=st["name"], InstanceGroups=groups)
    h.save_state(gpu_group=new, gpu_az=az, gpu_type=itype)
    d.err(f"GPU nodes -> group {new} ({itype} x{h.GPU_COUNT}, {az}, {subnet}); other GPU groups -> 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
