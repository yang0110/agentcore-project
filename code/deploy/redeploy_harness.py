#!/usr/bin/env python3
"""Surgical redeploy of ONLY the mini harness Runtime (#1): acs_harness, so the deployed image
picks up the harness_runtime_agent.py changes (per-episode max_turns from the payload + the
AgentCoreRuntimeSandbox transient-5xx retry).

Deliberately does NOT touch the GPU capacity provider, the GPU sandbox, or the CPU sandbox, and
never runs prepare_gpu_volume — so it launches no g6 and costs nothing beyond the ECR push. Reuses
the exec role / VPC endpoints / adapter-port ingress that `all` already created (idempotent).

Recreating a runtime under the same name mints a NEW ARN (random suffix), so the new ARNs are
printed at the end as OUT lines for the b3 / b2 launch env.
"""
import os
import time

os.environ.setdefault("AWS_REGION", "us-west-2")

import boto3

import deploy_two_runtime as d

private_ip = os.environ.get("ADAPTER_PUBLIC_HOST")
if not private_ip:
    raise SystemExit("set ADAPTER_PUBLIC_HOST to this box PRIVATE IP (the harness dials it back)")

control = boto3.client("bedrock-agentcore-control", region_name=d.REGION)

MINI = f"{d.PREFIX}_harness"

# --- delete the harness runtime, then wait until the name is actually gone (delete is async;
#     recreating before the name frees up would collide). ------------------------------------------
arn, rid, status = d._existing_runtime_arn(control, MINI)
if rid:
    control.delete_agent_runtime(agentRuntimeId=rid)
    d.err(f"deleting runtime {MINI} ({status})")
for _ in range(60):
    arn, rid, _ = d._existing_runtime_arn(control, MINI)
    if not rid:
        break
    time.sleep(5)
else:
    raise SystemExit(f"{MINI} still present after 5min — aborting before recreate")
d.err(f"{MINI} deleted")

# --- rebuild+push ONLY the mini harness image (arm64) -------------------------------------------
mini_uri = d._build_push_one(f"{d.PREFIX}-harness", "Dockerfile.harness", "linux/arm64")

# --- recreate the mini harness runtime (same VPC wiring as cmd_all) -----------------------------
xrole = d.exec_role(f"{d.PREFIX}-exec-role")
vpc_id, subnets, runtime_sg = d.default_vpc()
d.open_adapter_port(private_ip, runtime_sg)
harness_subnets = d.ensure_vpc_endpoints(vpc_id, subnets, runtime_sg)
if not harness_subnets:
    raise SystemExit("no subnet served by all VPC endpoints; cannot place VPC-mode harness")
vpc_net = {"networkMode": "VPC",
           "networkModeConfig": {"securityGroups": [runtime_sg], "subnets": harness_subnets}}
common = {"AWS_REGION": d.REGION}

mini_arn = d.managed_runtime(MINI, mini_uri, xrole, {**common, "HARNESS_KIND": "mini"}, vpc_net)

print(f"OUT AGENTCORE_HARNESS_RUNTIME_ARN={mini_arn}")
