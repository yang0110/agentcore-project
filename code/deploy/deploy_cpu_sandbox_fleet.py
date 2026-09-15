#!/usr/bin/env python3
"""Stand up a FLEET of identical CPU sandbox Runtimes for the runtime-count throughput ablation.

The base runtime `acs_sandbox_cpu` (from deploy_two_runtime.py) already exists. This creates
`acs_sandbox_cpu_1 .. acs_sandbox_cpu_{N-1}` from the SAME already-pushed image + execution role,
so the ablation can fan episodes out across up to N Runtimes (AGENTCORE_SANDBOX_RUNTIME_ARNS).

No image build — reuses the pushed :latest. Idempotent: existing names are reused, not recreated.

    FLEET_N=4 python3 deploy_cpu_sandbox_fleet.py
    # prints: export AGENTCORE_SANDBOX_RUNTIME_ARNS=<arn0>,<arn1>,<arn2>,<arn3>

Real AWS mutation: creates (N-1) managed serverless AgentCore Runtimes (billed per session/use).
"""
import os
import sys

import deploy_two_runtime as d

REGION = d.REGION
PREFIX = d.PREFIX
N = int(os.environ.get("FLEET_N", "4"))


def main() -> int:
    account = d.account()
    image_uri = f"{account}.dkr.ecr.{REGION}.amazonaws.com/{PREFIX}-sandbox-cpu:latest"
    d.err(f"reusing pushed image {image_uri}")
    xrole = d.exec_role(f"{PREFIX}-exec-role")
    public_net = {"networkMode": "PUBLIC"}
    env = {"AWS_REGION": REGION, "SANDBOX_WORKDIR": "/workspace"}

    names = [f"{PREFIX}_sandbox_cpu"] + [f"{PREFIX}_sandbox_cpu_{i}" for i in range(1, N)]
    arns = []
    for name in names:
        arn = d.managed_runtime(name, image_uri, xrole, env, public_net)
        d.err(f"  {name} -> {arn}")
        arns.append(arn)

    joined = ",".join(arns)
    print(f"\nexport AGENTCORE_SANDBOX_RUNTIME_ARNS={joined}")
    print(f"# {len(arns)} CPU sandbox Runtimes ready (sweep runtime_count up to {len(arns)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
