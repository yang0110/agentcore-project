#!/usr/bin/env python3
"""Deploy (or redeploy) ONLY the Demo-B4 GPU sandbox Runtime #2 — the g6 whose single GPU is shared
by a Ray farm running INSIDE it (sandbox_ray_farm.RayGpuFarm, driven by sandbox_server_agent).

This is a SEPARATE runtime from the b2/b3 GPU sandbox (acs_sandbox_gpu): b4 needs `ray` baked in and
its server run AS A PACKAGE (relative imports on the farm path), so it uses its own image
(Dockerfile.sandbox.gpu.shared -> acs-sandbox-gpu-b4). It reuses the exec role + operator role + GPU
capacity provider that `deploy_two_runtime.py all` already created (idempotent), and never runs
mode=prepare (b4 installs torch at container startup, uses no volume for torch).

Recreating under the same name mints a NEW ARN, printed at the end as the export for a remote-g6 Ray-farm run
(demo_gpu_single, incl. its SNAPSHOT_EVERY=1 angle, with SANDBOX=agentcore-ray-farm; the former demo_gpu_shared):
    export AGENTCORE_SANDBOX_RUNTIME_ARN=<printed arn>

Farm tuning (injected as runtime env; retunable here without a rebuild). Defaults suit one g6.xlarge
(1 L4, 24 GB): frac=0.25 -> floor(1/0.25)=4 sandboxes sharing the GPU, matching run.sh CONC=4.
    B4_GPU_FRACTION (0.25)  B4_FARM_GPUS (1 — MUST be set; slim image has no nvidia-smi for Ray to
    autodetect)  B4_PER_SANDBOX_MEM_FRACTION (0.25)  B4_MAX_SANDBOXES (unset => floor(gpus/frac))
    B4_WORKSPACE_ROOT (/workspace/b4-sbx)  B4_WARM_CUDA (1)

Usage:
    python3 redeploy_gpu_sandbox_shared.py            # build + push + (re)create the b4 GPU runtime
    SKIP_BUILD=1 python3 redeploy_gpu_sandbox_shared.py   # reuse the already-pushed image
"""
import os
import time

os.environ.setdefault("AWS_REGION", "us-west-2")

import boto3

import deploy_two_runtime as d

control = boto3.client("bedrock-agentcore-control", region_name=d.REGION)
B4 = f"{d.PREFIX}_sandbox_gpu_b4"

# --- delete any existing b4 runtime, then wait until the name is actually gone (delete is async;
#     recreating before the name frees up would collide). ------------------------------------------
arn, rid, status = d._existing_runtime_arn(control, B4)
if rid:
    control.delete_agent_runtime(agentRuntimeId=rid)
    d.err(f"deleting runtime {B4} ({status})")
    for _ in range(60):
        arn, rid, _ = d._existing_runtime_arn(control, B4)
        if not rid:
            break
        time.sleep(5)
    else:
        raise SystemExit(f"{B4} still present after 5min — aborting before recreate")
    d.err(f"{B4} deleted")

# --- build + push ONLY the b4 GPU sandbox image (native amd64 on this box) -------------------------
uri = d._build_push_one(f"{d.PREFIX}-sandbox-gpu-b4", "Dockerfile.sandbox.gpu.shared", "linux/amd64")

# --- (re)create the b4 runtime on the EXISTING GPU capacity provider (reused, no new instances) ----
xrole = d.exec_role(f"{d.PREFIX}-exec-role")
op_arn = d.operator_role(f"{d.PREFIX}-operator-role")
cp_arn = d.gpu_capacity_provider(f"{d.PREFIX}_gpu_cp", op_arn)   # reuses if present

b4_env = {
    "AWS_REGION": d.REGION,
    "SANDBOX_WORKDIR": "/workspace",
    # Ray GPU-sharing farm config (read by sandbox_ray_farm.RayGpuFarm.from_env on the g6).
    "B4_GPU_FRACTION": os.environ.get("B4_GPU_FRACTION", "0.25"),
    "B4_FARM_GPUS": os.environ.get("B4_FARM_GPUS", "1"),   # slim image => Ray can't autodetect; force it
    "B4_PER_SANDBOX_MEM_FRACTION": os.environ.get("B4_PER_SANDBOX_MEM_FRACTION", "0.25"),
    "B4_WORKSPACE_ROOT": os.environ.get("B4_WORKSPACE_ROOT", "/workspace/b4-sbx"),
    "B4_WARM_CUDA": os.environ.get("B4_WARM_CUDA", "1"),
}
if os.environ.get("B4_MAX_SANDBOXES"):
    b4_env["B4_MAX_SANDBOXES"] = os.environ["B4_MAX_SANDBOXES"]

gpu_arn = d.gpu_runtime_on_cp(B4, uri, xrole, cp_arn, b4_env)

print(f"OUT AGENTCORE_SANDBOX_GPU_B4_RUNTIME_ARN={gpu_arn}")
print("\n# ---- export for Demo B4 (remote g6 + Ray GPU-sharing farm inside) ----")
print(f"export AGENTCORE_SANDBOX_RUNTIME_ARN={gpu_arn}")
