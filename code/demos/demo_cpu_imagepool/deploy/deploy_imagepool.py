#!/usr/bin/env python3
"""Provision demo_cpu_imagepool's POOL of custom CPU sandbox Runtimes (one image per family).

Builds the three custom sandbox images (datasci=pandas, scipy, sklearn) for arm64 (the managed
serverless fleet is arm64-only), pushes each to its own ECR repo, and registers one MANAGED
AgentCore Runtime per image, in networkMode=VPC — reached only via the IAM control plane
(invoke_agent_runtime), never dialing back, and NO public egress (private VPC endpoints only).
No inbound port and no security-group ingress rule are opened (unlike the CLI demo, which needs
one SG→SG rule for the adapter). The images are self-contained (deps baked at build), so the
runtime needs no internet at all — VPC mode with private endpoints suffices to boot + be invoked.

Reuses the shared deploy_two_runtime helpers (account/ensure_ecr/_ensure_buildx/exec_role/
managed_runtime) so it shares the SAME execution role + buildx builder as the rest of the fleet.

Idempotent: existing ECR repos + Runtime names are reused (delete a Runtime to pick up a new image).
Honors SKIP_BUILD=1 to reuse already-pushed :latest images and only (re)register the Runtimes.

    python3 deploy_imagepool.py
    # prints: export AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE='{"datasci":...,"scipy":...,"sklearn":...}'

Real AWS mutation: up to 3 ECR repos + images and 3 managed serverless Runtimes (billed per session).
"""
import json
import os
import sys

# The shared deploy helpers live in code/deploy/. Add it to the path so this self-contained
# demo-local script can reuse them rather than re-implementing build/push/register.
HERE = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.abspath(os.path.join(HERE, ".."))          # code/demos/demo_cpu_imagepool
SHARED_DEPLOY = os.path.abspath(os.path.join(DEMO_DIR, "..", "..", "deploy"))
sys.path.insert(0, SHARED_DEPLOY)

import deploy_two_runtime as d  # noqa: E402

REGION = d.REGION
PREFIX = d.PREFIX

# family -> (ECR repo, dockerfile in THIS demo's deploy/). Build context = the demo dir (the
# Dockerfiles COPY agentcore_slime_demo/sandbox_server_agent.py). All arm64 (managed fleet).
FAMILIES = {
    "datasci": (f"{PREFIX}-sandbox-datasci", "Dockerfile.sandbox.datasci"),
    "scipy":   (f"{PREFIX}-sandbox-scipy",   "Dockerfile.sandbox.scipy"),
    "sklearn": (f"{PREFIX}-sandbox-sklearn", "Dockerfile.sandbox.sklearn"),
}
PLATFORM = "linux/arm64"


def build_push_one(repo: str, dockerfile: str) -> str:
    """Build+push ONE image with context = the demo dir (not code/). Returns the :latest URI."""
    acct = d.account()
    registry = f"{acct}.dkr.ecr.{REGION}.amazonaws.com"
    uri = d.ensure_ecr(repo) + ":latest"
    if os.environ.get("SKIP_BUILD") == "1":
        d.err(f"SKIP_BUILD=1 — reusing already-pushed {uri}")
        return uri
    d._ensure_buildx()
    d.sh(f"aws ecr get-login-password --region {REGION} | "
         f"docker login --username AWS --password-stdin {registry}")
    d.sh(f"docker buildx build --builder {d.BUILDX_BUILDER} --platform {PLATFORM} "
         f"--provenance=false --sbom=false -f {HERE}/{dockerfile} -t {uri} --push {DEMO_DIR}")
    d.err(f"pushed {uri} ({PLATFORM})")
    return uri


def main() -> int:
    xrole = d.exec_role(f"{PREFIX}-exec-role")
    env = {"AWS_REGION": REGION, "SANDBOX_WORKDIR": "/workspace"}

    # networkMode=VPC: private egress only, NO public network and NO inbound port. Reuse the
    # shared VPC endpoints (ecr.api/ecr.dkr/logs/sts/bedrock-agentcore/s3) the harness deploy
    # created — ensure_vpc_endpoints is idempotent and returns the subnets whose AZ every
    # interface endpoint serves (a VPC-mode runtime that lands in an unserved AZ fails to start).
    vpc_id, subnets, runtime_sg = d.default_vpc()
    d.err(f"default VPC {vpc_id}; runtime SG {runtime_sg}")
    ep_subnets = d.ensure_vpc_endpoints(vpc_id, subnets, runtime_sg)
    if not ep_subnets:
        d.die("no subnet is served by all required VPC endpoints; cannot place VPC-mode runtime")
    vpc_net = {"networkMode": "VPC",
               "networkModeConfig": {"securityGroups": [runtime_sg], "subnets": ep_subnets}}

    arn_by_image = {}
    for family, (repo, dockerfile) in FAMILIES.items():
        uri = build_push_one(repo, dockerfile)
        name = f"{PREFIX}_sandbox_{family}"
        arn = d.managed_runtime(name, uri, xrole, env, vpc_net)
        d.err(f"  {family}: {name} -> {arn}")
        arn_by_image[family] = arn

    print("\nexport AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE='" + json.dumps(arn_by_image) + "'")
    print(f"# {len(arn_by_image)} custom-image sandbox Runtimes ready (datasci/scipy/sklearn)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
