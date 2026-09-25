#!/usr/bin/env python3
"""Provision demo_cpu_instance_pool's LARGE custom-Runtime sandbox on a CPU capacity-provider INSTANCE.

The image (deploy/Dockerfile.sandbox.bigcpu: torch CPU + scientific stack) is ~4 GB — past the
~2 GB AgentCore Code Interpreter *microVM* image cap (prove it offline: `python3 probe_bigimage.py`).
So it CANNOT run on a serverless microVM; it must run on an AgentCore **capacity provider**, i.e. an
EC2 instance family you choose (default m5.xlarge). This is the CPU analogue of the g6 GPU path in
deploy/agentcore_deploy.py (vendored) — same helpers, a CPU instance type, no models volume.

Networking (see docs/agentcore-networking.md): DEFAULT is the SAFE **pull** shape — networkMode=PUBLIC,
reached only by IAM `invoke_agent_runtime` (SigV4/TLS), **no inbound port, no SG ingress rule opened**.
That is all this demo needs (the harness is in-process; the sandbox never dials back). Only if you
add an in-sandbox agent that must reach a service you host would you switch to VPC + an SG->SG rule —
validate that with `python3 netcheck.py` BEFORE deploying.

    python3 deploy_bigimage.py                 # build+push, CP, runtime -> prints the ARN export
    CPU_INSTANCE=m5.2xlarge python3 deploy_bigimage.py
    SKIP_BUILD=1 python3 deploy_bigimage.py    # reuse the pushed :latest, just (re)register

Real AWS mutation: 1 ECR repo + image, 1 CPU capacity provider, 1 Runtime (billed while a session
runs). Idempotent: existing repo / CP / Runtime names are reused.

ONE Runtime serves the whole warm pool: run.sh's SANDBOX_POOL=N mints N runtimeSessionIds against
this ARN, and AgentCore places each session on its own instance from the capacity provider.
"""
import os
import sys
import time

import boto3

# Reuse the vendored deploy helpers (deploy/agentcore_deploy.py) rather than re-implementing build/push/roles.
HERE = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.abspath(os.path.join(HERE, ".."))          # code/demos/demo_cpu_instance_pool
sys.path.insert(0, HERE)  # this demo's vendored deploy/agentcore_deploy.py (no code/deploy dependency)

import agentcore_deploy as d  # noqa: E402

REGION = d.REGION
PREFIX = d.PREFIX

REPO = f"{PREFIX}-sandbox-bigcpu"
DOCKERFILE = "Dockerfile.sandbox.bigcpu"
# Capacity-provider instance is x86_64 -> build amd64 (this box is amd64, so NO cross-build/QEMU).
PLATFORM = "linux/amd64"
CPU_INSTANCE = os.environ.get("CPU_INSTANCE", "m5.xlarge")
CP_NAME = f"{PREFIX}_cpu_cp_bigimage"
RUNTIME_NAME = f"{PREFIX}_sandbox_bigcpu"
# Root volume must comfortably hold the ~4 GB image unpacked; give generous free space.
ROOT_FREE_GIB = int(os.environ.get("ROOT_FREE_GIB", "40"))


def build_push() -> str:
    """Build+push the big image (context = the demo dir, so it COPYs agentcore_slime_demo/)."""
    acct = d.account()
    registry = f"{acct}.dkr.ecr.{REGION}.amazonaws.com"
    uri = d.ensure_ecr(REPO) + ":latest"
    if os.environ.get("SKIP_BUILD") == "1":
        d.err(f"SKIP_BUILD=1 — reusing already-pushed {uri}")
        return uri
    d._ensure_buildx()
    d.sh(f"aws ecr get-login-password --region {REGION} | "
         f"docker login --username AWS --password-stdin {registry}")
    d.sh(f"docker buildx build --builder {d.BUILDX_BUILDER} --platform {PLATFORM} "
         f"--provenance=false --sbom=false -f {HERE}/{DOCKERFILE} -t {uri} --push {DEMO_DIR}")
    d.err(f"pushed {uri} ({PLATFORM})")
    return uri


def cpu_capacity_provider(name: str, operator_arn: str) -> str:
    """A CPU capacity provider (x86_64, CPU_INSTANCE). Mirrors agentcore_deploy.gpu_capacity_provider
    but with a CPU instance type and NO models EBS volume (the big image is self-contained)."""
    control = boto3.client("bedrock-agentcore-control", region_name=REGION)
    try:
        for cp in control.list_capacity_providers().get("capacityProviders", []):
            if cp.get("name") == name:
                d.err(f"reusing capacity provider {name}")
                return cp["capacityProviderArn"]
    except Exception:  # noqa: BLE001
        pass
    _, subnets, sg = d.default_vpc()
    resp = control.create_capacity_provider(
        name=name,
        description=f"demo_cpu_instance_pool CPU capacity provider ({CPU_INSTANCE})",
        permissionsConfiguration={"capacityProviderOperatorRoleArn": operator_arn},
        computeConfiguration={"ec2Configuration": {
            "launchTemplateSource": {"launchParameters": {
                "operatingSystem": "LINUX_X86_64",
                "instanceRequirements": {"allowedInstanceTypes": [CPU_INSTANCE]}}},
            "vpcConfiguration": {"subnets": subnets, "securityGroups": [sg]},
            "rootVolume": {"freeSpaceGiB": ROOT_FREE_GIB, "volumeType": "gp3"},
            "lifecycleConfiguration": {"idleInstanceTimeout": 600, "maxLifetime": 86400}}},
    )
    cp_id = resp["capacityProviderId"]
    d.err(f"waiting for {cp_id} READY (launches NO instances yet)")
    while True:
        got = control.get_capacity_provider(capacityProviderId=cp_id)
        if got["status"] == "READY":
            break
        if "FAILED" in got["status"]:
            d.die(f"{got['status']}: {got.get('statusReason')}")
        time.sleep(5)
    return resp["capacityProviderArn"]


def runtime_on_cp(name: str, image_uri: str, exec_arn: str, cp_arn: str, env: dict) -> str:
    """Register the Runtime on the CPU capacity provider. networkMode=PUBLIC (pull-only, no inbound);
    no filesystem volume (unlike the GPU path, which mounts a models EBS volume)."""
    control = boto3.client("bedrock-agentcore-control", region_name=REGION)
    # A new image or env (e.g. turning on the CPU farm) needs a fresh CreateAgentRuntime — the reuse
    # path does NOT update an existing runtime's image/env. FORCE_RECREATE=1 deletes the existing one
    # first so the redeploy picks up the change (this mints a new ARN — the script prints it).
    if os.environ.get("FORCE_RECREATE") == "1":
        _, rid, _ = d._existing_runtime_arn(control, name)
        if rid:
            d.err(f"FORCE_RECREATE=1 — deleting existing runtime {name} to pick up new image/env")
            control.delete_agent_runtime(agentRuntimeId=rid)
            time.sleep(10)
    arn = d._reuse_or_clear(control, name)
    if arn:
        d.err(f"reusing runtime {name} (set FORCE_RECREATE=1 to pick up a new image/env)")
        return arn
    resp = control.create_agent_runtime(
        agentRuntimeName=name,
        roleArn=exec_arn,
        agentRuntimeArtifact={"containerConfiguration": {"containerUri": image_uri}},
        protocolConfiguration={"serverProtocol": "HTTP"},
        capacityProviderConfiguration={"capacityProviderArn": cp_arn},
        lifecycleConfiguration={"idleRuntimeSessionTimeout": 600, "maxLifetime": 86400},
        environmentVariables=env,
    )
    d._wait_runtime(control, resp["agentRuntimeId"])
    return resp["agentRuntimeArn"]


def main() -> int:
    uri = build_push()
    xrole = d.exec_role(f"{PREFIX}-exec-role")
    op_arn = d.operator_role(f"{PREFIX}-cpu-operator-role")
    cp_arn = cpu_capacity_provider(CP_NAME, op_arn)
    # SANDBOX_FARM=cpu turns on the shared-instance CPU farm (fresh env per sub_session). It is also
    # baked into the image (Dockerfile ENV); set here too so it's explicit in the runtime config.
    env = {"AWS_REGION": REGION, "SANDBOX_WORKDIR": "/workspace", "SANDBOX_FARM": "cpu"}
    arn = runtime_on_cp(RUNTIME_NAME, uri, xrole, cp_arn, env)
    d.err(f"runtime {RUNTIME_NAME} -> {arn}")

    print(f"\nexport AGENTCORE_SANDBOX_RUNTIME_ARN='{arn}'")
    print(f"# big-image ({CPU_INSTANCE}) custom Runtime ready — bash 5_run.sh (SANDBOX_POOL=N warm instances, default 2)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
