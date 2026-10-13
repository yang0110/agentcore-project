#!/usr/bin/env python3
"""Move the cluster's GPU nodes to another AZ when the current one has no capacity (HOST):

    python3 deploy/hyperpod/move_gpu_az.py us-west-2c [ml.g6e.48xlarge] [spot]

HyperPod can't change an instance group's OverrideVpcConfig in place, so this ADDS a group
`gpu-<az letter>[-<family>]` (same partition, same lifecycle + storage, OverrideVpcConfig = the AZ's egress subnet + the
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
    spot = len(sys.argv) > 3 and sys.argv[3] == "spot"    # Spot capacity (its own "cluster spot instance usage" quota)
    st = h.load_state()
    sm = boto3.client("sagemaker", region_name=d.REGION)
    c = sm.describe_cluster(ClusterName=st["name"])
    subnet = d.ensure_egress_subnets(st["vpc"], [az])[0]
    family = itype.removeprefix("ml.").split(".")[0]              # g6e, g7e, p5en, ...
    new = f"gpu-{az[-1]}" + ("" if family == "g6e" else f"-{family}")   # gpu-c (g6e), gpu-b-g7e, gpu-a-p5en
    new += "-spot" if spot else ""
    groups, have = [], False
    for g in c["InstanceGroups"]:
        spec = {k: g[k] for k in ("InstanceGroupName", "InstanceType", "LifeCycleConfig", "ExecutionRole",
                                  "ThreadsPerCore", "InstanceStorageConfigs", "SlurmConfig", "OverrideVpcConfig",
                                  "CapacityRequirements")
                if k in g}
        spec["InstanceCount"] = g["TargetCount"]
        if g.get("SlurmConfig", {}).get("NodeType") == "Compute":
            spec["InstanceCount"] = h.GPU_COUNT if g["InstanceGroupName"] == new else 0
            have |= g["InstanceGroupName"] == new
            template = spec
        groups.append(spec)
    if not have:
        groups.append(template | {"InstanceGroupName": new, "InstanceType": itype, "InstanceCount": h.GPU_COUNT,
                                  "OverrideVpcConfig": {"SecurityGroupIds": [st["sg"]], "Subnets": [subnet]}}
                      | ({"CapacityRequirements": {"Spot": {}}} if spot else {}))
        if not spot:
            groups[-1].pop("CapacityRequirements", None)
    sm.update_cluster(ClusterName=st["name"], InstanceGroups=groups)
    h.save_state(gpu_group=new, gpu_az=az, gpu_type=itype, gpu_spot=spot)
    d.err(f"GPU nodes -> group {new} ({itype} x{h.GPU_COUNT}, {az}, {subnet}); other GPU groups -> 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
