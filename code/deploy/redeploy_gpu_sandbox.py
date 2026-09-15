#!/usr/bin/env python3
"""Surgical redeploy of ONLY the GPU sandbox Runtime #2 (acs_sandbox_gpu), so it picks up the new
Dockerfile.sandbox.gpu that BAKES torch into the image (was installed onto the capacity-provider
EBS volume via {"mode":"prepare"}, which did NOT persist across instance recycles -> B3/B2 reward 0;
root-caused 2026-09-09).

Reuses the exec role + GPU capacity provider that `all` already created (idempotent); does NOT touch
the harness or CPU sandbox, and never runs prepare_gpu_volume (torch is baked now, no volume install).
Recreating the runtime under the same name mints a NEW ARN — printed at the end for the b3/b2 launch.
"""
import os
import time

os.environ.setdefault("AWS_REGION", "us-west-2")

import boto3

import deploy_two_runtime as d

control = boto3.client("bedrock-agentcore-control", region_name=d.REGION)
GPU = f"{d.PREFIX}_sandbox_gpu"

# --- delete the GPU sandbox runtime, then wait until the name is actually gone (delete is async;
#     recreating before the name frees up would collide). ------------------------------------------
arn, rid, status = d._existing_runtime_arn(control, GPU)
if rid:
    control.delete_agent_runtime(agentRuntimeId=rid)
    d.err(f"deleting runtime {GPU} ({status})")
    for _ in range(60):
        arn, rid, _ = d._existing_runtime_arn(control, GPU)
        if not rid:
            break
        time.sleep(5)
    else:
        raise SystemExit(f"{GPU} still present after 5min — aborting before recreate")
    d.err(f"{GPU} deleted")

# --- rebuild+push ONLY the GPU sandbox image (native amd64 on this box; now ~2.7 GB with torch) ---
uri = d._build_push_one(f"{d.PREFIX}-sandbox-gpu", "Dockerfile.sandbox.gpu", "linux/amd64")

# --- recreate the GPU runtime on the EXISTING capacity provider (reused, no new instances) --------
xrole = d.exec_role(f"{d.PREFIX}-exec-role")
op_arn = d.operator_role(f"{d.PREFIX}-operator-role")
cp_arn = d.gpu_capacity_provider(f"{d.PREFIX}_gpu_cp", op_arn)   # reuses if present
gpu_arn = d.gpu_runtime_on_cp(
    GPU, uri, xrole, cp_arn, {"AWS_REGION": d.REGION, "SANDBOX_WORKDIR": "/workspace"})

print(f"OUT AGENTCORE_SANDBOX_GPU_RUNTIME_ARN={gpu_arn}")
