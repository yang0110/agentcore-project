#!/usr/bin/env python3
"""Deploy the two-Runtime target for the mini-harness demos (A3 / B3).

Corrected hosting model (verified against the bedrock-agentcore-control API model, not assumed):

  * Harness Runtime #1    — MANAGED serverless microVM (no capacity provider, no GPU), but with
                            networkMode=VPC so its egress sits in the default VPC. It is the ONLY
                            component that dials back to the AnthropicAdapter proxy on the trainer
                            box (ADAPTER_PUBLIC_HOST:ADAPTER_PORT, a PRIVATE IP), so a single narrow
                            security-group ingress rule — no public port — is what lets it in.
  * CPU sandbox (A3)      — MANAGED serverless microVM, networkMode=PUBLIC. Reached only via the
                            IAM control plane (invoke_agent_runtime); it never dials back, so it
                            needs no VPC.
  * GPU sandbox (B3)      — CAPACITY PROVIDER (g6) + persistent EBS "models" volume. The CUDA torch
                            stack is >2 GB (over the image cap), so it is installed onto the volume
                            at a one-time {"mode":"prepare"} invoke and reused. Reached via IAM.

networkConfiguration shape (from the control model): VPC mode is
    {"networkMode": "VPC", "networkModeConfig": {"securityGroups":[...], "subnets":[...]}}
(the key is networkModeConfig, NOT vpcConfig — that latter is the code-interpreter tool client).

Real, mostly hard-to-reverse AWS mutations: ECR repos + images, one IAM execution role, a GPU
capacity provider, three runtimes, and one SG ingress rule. The first B3 invoke cold-launches a
real g6.xlarge (~$0.80-1/hr) + EBS. `cleanup` (below) tears the stack back down.

Env knobs (all optional; sensible defaults):
    AWS_REGION (us-west-2), ADAPTER_PORT (18021), ADAPTER_PUBLIC_HOST (this box PRIVATE IP),
    GPU_INSTANCE (g6.xlarge), MODELS_GIB (40), SKIP_BUILD=1 (reuse pushed images),
    SKIP_PREPARE=1 (don't cold-launch the GPU to build the volume yet),
    AGENTCORE_TRAINER_SG (override auto-detected trainer SG), NAME_PREFIX (acs).

Usage:
    python3 deploy_two_runtime.py all             # full deploy (A3/B3), prints the export block
    python3 deploy_two_runtime.py cleanup         # delete runtimes + capacity provider
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-west-2"
PREFIX = os.environ.get("NAME_PREFIX", "acs")
ADAPTER_PORT = int(os.environ.get("ADAPTER_PORT", "18021"))
GPU_INSTANCE = os.environ.get("GPU_INSTANCE", "g6.xlarge")
MODELS_GIB = int(os.environ.get("MODELS_GIB", "40"))
MODELS_VOL = "models"
MODELS_MOUNT = "/mnt/models"

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_CTX = os.path.abspath(os.path.join(HERE, ".."))          # build context = code/
OPERATOR_MANAGED_POLICY = "arn:aws:iam::aws:policy/BedrockAgentCoreRuntimeInstancesOperatorRolePolicy"

BUILDX_BUILDER = "acsbuilder"

# Architecture matters and is NOT uniform:
#   * MANAGED serverless runtimes (harness, CPU sandbox) run on AgentCore's ARM fleet -> linux/arm64
#     (CreateAgentRuntime rejects amd64 with "Supported platforms: [arm64]").
#   * The GPU sandbox runs on a g6 capacity-provider instance, which is x86_64 -> linux/amd64.
# This box is amd64, so the arm64 images are cross-built via QEMU under a buildx container builder.
IMAGES = {  # logical name -> (ecr repo, dockerfile, platform)
    "harness":     (f"{PREFIX}-harness",     "Dockerfile.harness",     "linux/arm64"),
    "sandbox_cpu": (f"{PREFIX}-sandbox-cpu", "Dockerfile.sandbox.cpu", "linux/arm64"),
    "sandbox_gpu": (f"{PREFIX}-sandbox-gpu", "Dockerfile.sandbox.gpu", "linux/amd64"),
}


def err(m): print(f"    {m}", file=sys.stderr, flush=True)
def out(k, v): print(f"OUT {k}={v}", flush=True)
def die(m): print(f"ERROR: {m}", file=sys.stderr); sys.exit(1)
def sh(cmd): err(f"$ {cmd}"); subprocess.run(cmd, shell=True, check=True)


def account():
    return boto3.client("sts", region_name=REGION).get_caller_identity()["Account"]


# ----------------------------------------------------------------------------- ECR + image build
def ensure_ecr(repo: str) -> str:
    ecr = boto3.client("ecr", region_name=REGION)
    try:
        ecr.create_repository(repositoryName=repo, imageScanningConfiguration={"scanOnPush": False})
        err(f"created ECR repo {repo}")
    except ecr.exceptions.RepositoryAlreadyExistsException:
        err(f"reusing ECR repo {repo}")
    return ecr.describe_repositories(repositoryNames=[repo])["repositories"][0]["repositoryUri"]


def _ensure_buildx() -> None:
    """A buildx CONTAINER builder + QEMU so this amd64 box can cross-build the arm64 images.
    Idempotent; safe to call on every deploy."""
    # Register the arm64 emulator (no-op if already installed).
    sh("docker run --privileged --rm tonistiigi/binfmt --install arm64")
    r = subprocess.run(f"docker buildx inspect {BUILDX_BUILDER}", shell=True,
                       capture_output=True, text=True)
    if r.returncode != 0:
        sh(f"docker buildx create --name {BUILDX_BUILDER} --driver docker-container --bootstrap")


def build_push() -> dict:
    """Build each image for its REQUIRED arch (arm64 managed / amd64 GPU-CP) and push.
    Returns {logical: image_uri}."""
    acct = account()
    registry = f"{acct}.dkr.ecr.{REGION}.amazonaws.com"
    uris = {}
    for logical, (repo, dockerfile, platform) in IMAGES.items():
        uris[logical] = ensure_ecr(repo) + ":latest"
    if os.environ.get("SKIP_BUILD") == "1":
        err("SKIP_BUILD=1 — reusing already-pushed :latest images")
        return uris
    _ensure_buildx()
    sh(f"aws ecr get-login-password --region {REGION} | docker login --username AWS --password-stdin {registry}")
    for logical, (repo, dockerfile, platform) in IMAGES.items():
        uri = uris[logical]
        # buildx --push emits a single-arch image straight to ECR. --provenance/--sbom=false keep it a
        # plain image, not an attestation manifest list (AgentCore's containerUri wants a plain image).
        sh(f"docker buildx build --builder {BUILDX_BUILDER} --platform {platform} "
           f"--provenance=false --sbom=false -f {HERE}/{dockerfile} -t {uri} --push {CODE_CTX}")
        err(f"pushed {uri} ({platform})")
    return uris


# ----------------------------------------------------------------------------- IAM roles
def exec_role(name: str) -> str:
    """Runtime execution role: ECR pull + CloudWatch logs + (for the harness) invoke the sandbox."""
    iam = boto3.client("iam")
    acct = account()
    trust = {"Version": "2012-10-17", "Statement": [{
        "Effect": "Allow", "Principal": {"Service": "bedrock-agentcore.amazonaws.com"},
        "Action": "sts:AssumeRole", "Condition": {"StringEquals": {"aws:SourceAccount": acct}}}]}
    policy = {"Version": "2012-10-17", "Statement": [
        {"Effect": "Allow", "Action": [
            "ecr:GetAuthorizationToken", "ecr:BatchCheckLayerAvailability",
            "ecr:GetDownloadUrlForLayer", "ecr:BatchGetImage"], "Resource": "*"},
        {"Effect": "Allow", "Action": [
            "logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents",
            "logs:DescribeLogStreams", "logs:DescribeLogGroups"], "Resource": "*"},
        # The harness invokes the sandbox Runtime #2 from inside Runtime #1.
        {"Effect": "Allow", "Action": ["bedrock-agentcore:InvokeAgentRuntime"], "Resource": "*"},
        # VPC networkMode (harness) attaches an ENI in the default VPC — the role must manage it
        # (same set Lambda-in-VPC needs). Harmless for the PUBLIC-mode CPU sandbox that shares this role.
        {"Effect": "Allow", "Action": [
            "ec2:CreateNetworkInterface", "ec2:DescribeNetworkInterfaces",
            "ec2:DeleteNetworkInterface", "ec2:CreateNetworkInterfacePermission",
            "ec2:AssignPrivateIpAddresses", "ec2:UnassignPrivateIpAddresses"], "Resource": "*"},
    ]}
    try:
        arn = iam.create_role(RoleName=name, AssumeRolePolicyDocument=json.dumps(trust),
                              Description="AgentCore two-Runtime execution role")["Role"]["Arn"]
        err(f"created exec role {name}")
    except iam.exceptions.EntityAlreadyExistsException:
        iam.update_assume_role_policy(RoleName=name, PolicyDocument=json.dumps(trust))
        arn = iam.get_role(RoleName=name)["Role"]["Arn"]
        err(f"reusing exec role {name}")
    iam.put_role_policy(RoleName=name, PolicyName="acs-exec-inline", PolicyDocument=json.dumps(policy))
    time.sleep(10)  # let the role propagate before AgentCore assumes it
    return arn


def operator_role(name: str) -> str:
    """Role AgentCore assumes to provision the GPU capacity provider's EC2."""
    iam = boto3.client("iam")
    acct = account()
    trust = {"Version": "2012-10-17", "Statement": [{
        "Effect": "Allow", "Principal": {"Service": "bedrock-agentcore.amazonaws.com"},
        "Action": "sts:AssumeRole", "Condition": {"StringEquals": {"aws:SourceAccount": acct}}}]}
    try:
        arn = iam.create_role(RoleName=name, AssumeRolePolicyDocument=json.dumps(trust),
                              Description="AgentCore GPU capacity provider operator role")["Role"]["Arn"]
        err(f"created operator role {name}")
    except iam.exceptions.EntityAlreadyExistsException:
        iam.update_assume_role_policy(RoleName=name, PolicyDocument=json.dumps(trust))
        arn = iam.get_role(RoleName=name)["Role"]["Arn"]
        err(f"reusing operator role {name}")
    iam.attach_role_policy(RoleName=name, PolicyArn=OPERATOR_MANAGED_POLICY)
    time.sleep(10)
    return arn


# ----------------------------------------------------------------------------- VPC / SG
def default_vpc():
    """(vpc_id, [subnets], default_sg_id) for the default VPC — the runtime SG the harness lands in."""
    ec2 = boto3.client("ec2", region_name=REGION)
    vpcs = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"]
    if not vpcs:
        die("no default VPC; set one or adapt default_vpc()")
    vpc_id = vpcs[0]["VpcId"]
    subnets = [s["SubnetId"] for s in
               ec2.describe_subnets(Filters=[{"Name": "vpc-id", "Values": [vpc_id]}])["Subnets"]]
    override = os.environ.get("AGENTCORE_SECURITY_GROUP_IDS")
    if override:
        sg = override.split(",")[0]
    else:
        g = ec2.describe_security_groups(Filters=[
            {"Name": "vpc-id", "Values": [vpc_id]},
            {"Name": "group-name", "Values": ["default"]}])["SecurityGroups"]
        if not g:
            die(f"no default SG in {vpc_id}; set AGENTCORE_SECURITY_GROUP_IDS")
        sg = g[0]["GroupId"]
    return vpc_id, subnets, sg


def trainer_sgs(private_ip: str) -> list[str]:
    """Security groups on the trainer box's ENI (the box that runs the adapter proxy)."""
    if os.environ.get("AGENTCORE_TRAINER_SG"):
        return [s for s in os.environ["AGENTCORE_TRAINER_SG"].split(",") if s]
    ec2 = boto3.client("ec2", region_name=REGION)
    enis = ec2.describe_network_interfaces(
        Filters=[{"Name": "private-ip-address", "Values": [private_ip]}])["NetworkInterfaces"]
    if not enis:
        die(f"no ENI with private IP {private_ip}; set AGENTCORE_TRAINER_SG explicitly")
    return [g["GroupId"] for g in enis[0]["Groups"]]


def open_adapter_port(private_ip: str, runtime_sg: str) -> None:
    """Allow the harness (in runtime_sg) inbound to the adapter port on the trainer box's SG.
    NO public port: the source is the runtime security group, not a CIDR."""
    ec2 = boto3.client("ec2", region_name=REGION)
    for sg in trainer_sgs(private_ip):
        try:
            ec2.authorize_security_group_ingress(GroupId=sg, IpPermissions=[{
                "IpProtocol": "tcp", "FromPort": ADAPTER_PORT, "ToPort": ADAPTER_PORT,
                "UserIdGroupPairs": [{"GroupId": runtime_sg,
                                      "Description": "AgentCore harness Runtime to AnthropicAdapter proxy"}]}])
            err(f"opened tcp/{ADAPTER_PORT} on {sg} from {runtime_sg}")
        except ClientError as e:
            if "InvalidPermission.Duplicate" in str(e):
                err(f"ingress tcp/{ADAPTER_PORT} on {sg} from {runtime_sg} already present")
            else:
                raise


# The harness runs in networkMode=VPC, so its egress has NO internet route. To PULL its ECR image
# and reach the AWS APIs it needs (ECR, CloudWatch Logs, bedrock-agentcore invoke, STS), we add
# PRIVATE VPC endpoints instead of a NAT/public IP — keeping the "nothing public" mandate intact.
# S3 (where ECR layers live) is a Gateway endpoint on the route tables; the rest are Interface
# endpoints in the default SG (whose self-reference already allows the runtime ENI -> endpoint :443).
INTERFACE_ENDPOINT_SERVICES = ["ecr.api", "ecr.dkr", "logs", "sts", "bedrock-agentcore"]


def ensure_vpc_endpoints(vpc_id: str, subnets: list[str], sg: str) -> list[str]:
    """Create the private endpoints the VPC-mode harness needs to start, and return the subnets
    the harness may safely land in — i.e. those whose AZ is served by EVERY interface endpoint.
    A VPC-mode runtime with no internet route CANNOT start unless it can reach ecr.api/ecr.dkr/s3
    (image pull), logs, AND bedrock-agentcore (phone-home). If it lands in an AZ that some service
    doesn't offer, create fails with a null reason + empty logs — so we intersect the AZs."""
    ec2 = boto3.client("ec2", region_name=REGION)
    existing = {e["ServiceName"]: e for e in ec2.describe_vpc_endpoints(
        Filters=[{"Name": "vpc-id", "Values": [vpc_id]}])["VpcEndpoints"]}
    # An interface endpoint takes at most ONE subnet per AZ (else DuplicateSubnetsInSameZone).
    # Default VPCs can have two subnets in one AZ, so dedupe to one-per-AZ before creating.
    az_of = {s["SubnetId"]: s["AvailabilityZone"] for s in ec2.describe_subnets(
        SubnetIds=subnets)["Subnets"]}
    one_per_az: dict[str, str] = {}
    for sn in subnets:
        one_per_az.setdefault(az_of[sn], sn)

    # Endpoints must reachable on :443 from the harness ENI AND from this box (which may be in a
    # different SG). A VPC-CIDR ingress rule on the endpoint SG covers both without SG guesswork.
    vpc_cidr = ec2.describe_vpcs(VpcIds=[vpc_id])["Vpcs"][0]["CidrBlock"]
    try:
        ec2.authorize_security_group_ingress(GroupId=sg, IpPermissions=[{
            "IpProtocol": "tcp", "FromPort": 443, "ToPort": 443,
            "IpRanges": [{"CidrIp": vpc_cidr,
                          "Description": "VPC-internal HTTPS to interface VPC endpoints"}]}])
        err(f"opened tcp/443 on endpoint SG {sg} from {vpc_cidr}")
    except ClientError as e:
        if "InvalidPermission.Duplicate" not in str(e):
            raise

    served_azs: dict[str, set[str]] = {}  # svc -> AZs it offers
    for svc in INTERFACE_ENDPOINT_SERVICES:
        svc_name = f"com.amazonaws.{REGION}.{svc}"
        try:
            det = ec2.describe_vpc_endpoint_services(ServiceNames=[svc_name])["ServiceDetails"]
            served_azs[svc] = set(det[0]["AvailabilityZones"]) if det else set()
        except ClientError:
            served_azs[svc] = set()
        # Only place the endpoint in AZs the service actually supports (else the whole create fails).
        ep_subnets = [sn for az, sn in one_per_az.items()
                      if not served_azs[svc] or az in served_azs[svc]]
        if svc_name in existing:
            err(f"reusing interface endpoint {svc}")
            continue
        if not ep_subnets:
            err(f"interface endpoint {svc}: no subnet in a supported AZ — skipping")
            continue
        try:
            ec2.create_vpc_endpoint(
                VpcEndpointType="Interface", VpcId=vpc_id, ServiceName=svc_name,
                SubnetIds=ep_subnets, SecurityGroupIds=[sg], PrivateDnsEnabled=True)
            err(f"created interface endpoint {svc} in {[az_of[s] for s in ep_subnets]}")
        except ClientError as e:
            err(f"interface endpoint {svc}: {e}")
    # S3 gateway endpoint (ECR image layers) — attach to all of the VPC's route tables.
    s3_name = f"com.amazonaws.{REGION}.s3"
    if s3_name in existing:
        err("reusing S3 gateway endpoint")
    else:
        rts = [rt["RouteTableId"] for rt in ec2.describe_route_tables(
            Filters=[{"Name": "vpc-id", "Values": [vpc_id]}])["RouteTables"]]
        ec2.create_vpc_endpoint(VpcEndpointType="Gateway", VpcId=vpc_id,
                                ServiceName=s3_name, RouteTableIds=rts)
        err("created S3 gateway endpoint")

    # The harness may only land where EVERY interface service is reachable. Intersect the AZs;
    # if a service reported no AZs (e.g. describe failed), don't let it shrink the set.
    ok_azs = set(one_per_az)
    for svc in INTERFACE_ENDPOINT_SERVICES:
        if served_azs.get(svc):
            ok_azs &= served_azs[svc]
    harness_subnets = [sn for az, sn in one_per_az.items() if az in ok_azs]
    err(f"harness will use subnets in AZs {sorted(ok_azs)}: {harness_subnets}")
    return harness_subnets


# ----------------------------------------------------------------------------- Runtimes
def _wait_runtime(control, runtime_id: str) -> None:
    while True:
        st = control.get_agent_runtime(agentRuntimeId=runtime_id)["status"]
        err(f"  {runtime_id} status: {st}")
        if st == "READY":
            return
        if "FAILED" in st:
            die(f"runtime {runtime_id} {st}")
        time.sleep(10)


def _existing_runtime_arn(control, name: str):
    tok = None
    while True:
        kw = {"nextToken": tok} if tok else {}
        resp = control.list_agent_runtimes(**kw)
        for r in resp.get("agentRuntimes", []):
            if r["agentRuntimeName"] == name:
                return r["agentRuntimeArn"], r["agentRuntimeId"], r.get("status", "")
        tok = resp.get("nextToken")
        if not tok:
            return None, None, None


def _reuse_or_clear(control, name):
    """Return an existing runtime ARN to reuse, or None if it should be (re)created.
    A CREATE_FAILED leftover is deleted so we can create it cleanly."""
    arn, rid, status = _existing_runtime_arn(control, name)
    if arn and "FAILED" in (status or ""):
        err(f"deleting {status} runtime {name} before recreating")
        control.delete_agent_runtime(agentRuntimeId=rid)
        time.sleep(10)
        return None
    return arn


def managed_runtime(name, image_uri, exec_arn, env, network) -> str:
    """MANAGED serverless runtime (harness / CPU sandbox). Idempotent: reuse if the name exists."""
    control = boto3.client("bedrock-agentcore-control", region_name=REGION)
    arn = _reuse_or_clear(control, name)
    if arn:
        err(f"reusing runtime {name} (delete it to pick up a new image/env)")
        return arn
    resp = control.create_agent_runtime(
        agentRuntimeName=name,
        roleArn=exec_arn,
        agentRuntimeArtifact={"containerConfiguration": {"containerUri": image_uri}},
        protocolConfiguration={"serverProtocol": "HTTP"},
        networkConfiguration=network,
        environmentVariables=env,
    )
    _wait_runtime(control, resp["agentRuntimeId"])
    return resp["agentRuntimeArn"]


def gpu_capacity_provider(name, operator_arn) -> str:
    control = boto3.client("bedrock-agentcore-control", region_name=REGION)
    # Reuse if present.
    try:
        for cp in control.list_capacity_providers().get("capacityProviders", []):
            if cp.get("name") == name:
                err(f"reusing capacity provider {name}")
                return cp["capacityProviderArn"]
    except (ClientError, KeyError):
        pass
    _, subnets, sg = default_vpc()
    resp = control.create_capacity_provider(
        name=name,
        description="AgentCore GPU sandbox (B3) capacity provider",
        permissionsConfiguration={"capacityProviderOperatorRoleArn": operator_arn},
        computeConfiguration={"ec2Configuration": {
            "launchTemplateSource": {"launchParameters": {
                "operatingSystem": "LINUX_X86_64",
                "instanceRequirements": {"allowedInstanceTypes": [GPU_INSTANCE]}}},
            "vpcConfiguration": {"subnets": subnets, "securityGroups": [sg]},
            "volumes": [{"ebsConfiguration": {
                "name": MODELS_VOL, "sizeGiB": MODELS_GIB, "volumeType": "gp3", "encrypted": True}}],
            "rootVolume": {"freeSpaceGiB": 30, "volumeType": "gp3"},
            "lifecycleConfiguration": {"idleInstanceTimeout": 600, "maxLifetime": 86400}}},
    )
    cp_id = resp["capacityProviderId"]
    err(f"waiting for {cp_id} READY (launches NO instances yet)")
    while True:
        got = control.get_capacity_provider(capacityProviderId=cp_id)
        if got["status"] == "READY":
            break
        if "FAILED" in got["status"]:
            die(f"{got['status']}: {got.get('statusReason')}")
        time.sleep(5)
    return resp["capacityProviderArn"]


def gpu_runtime_on_cp(name, image_uri, exec_arn, cp_arn, env) -> str:
    control = boto3.client("bedrock-agentcore-control", region_name=REGION)
    arn = _reuse_or_clear(control, name)
    if arn:
        err(f"reusing GPU runtime {name}")
        return arn
    resp = control.create_agent_runtime(
        agentRuntimeName=name,
        roleArn=exec_arn,
        agentRuntimeArtifact={"containerConfiguration": {"containerUri": image_uri}},
        protocolConfiguration={"serverProtocol": "HTTP"},
        capacityProviderConfiguration={"capacityProviderArn": cp_arn},
        filesystemConfigurations=[{"capacityProviderVolume": {
            "volumeName": MODELS_VOL, "mountPath": MODELS_MOUNT}}],
        lifecycleConfiguration={"idleRuntimeSessionTimeout": 600, "maxLifetime": 86400},
        environmentVariables=env,
    )
    _wait_runtime(control, resp["agentRuntimeId"])
    return resp["agentRuntimeArn"]


def prepare_gpu_volume(gpu_arn: str) -> None:
    """One-time {"mode":"prepare"} — cold-launches the g6 and installs the CUDA torch stack onto
    the volume. Minutes long; widen the read timeout well past botocore's 60s default."""
    read_timeout = int(os.environ.get("INVOKE_READ_TIMEOUT", "1200"))
    cfg = Config(read_timeout=read_timeout, connect_timeout=60, retries={"max_attempts": 0})
    data = boto3.client("bedrock-agentcore", region_name=REGION, config=cfg)
    err("invoking GPU sandbox mode=prepare (cold-launches g6, installs torch to the volume)...")
    # runtimeSessionId must be in [33, 128] chars (AgentCore ParamValidation); pad the label.
    resp = data.invoke_agent_runtime(agentRuntimeArn=gpu_arn,
                                     runtimeSessionId="prepare-gpu-volume-padded-000000000000",
                                     payload=b'{"mode":"prepare"}')
    print(resp["response"].read().decode())


def _build_push_one(repo: str, dockerfile: str, platform: str) -> str:
    """Build+push a SINGLE image (used by redeploy_harness.py so we don't rebuild the whole A3/B3
    set). Returns the :latest image URI. Honors SKIP_BUILD=1 to reuse an already-pushed image."""
    acct = account()
    registry = f"{acct}.dkr.ecr.{REGION}.amazonaws.com"
    uri = ensure_ecr(repo) + ":latest"
    if os.environ.get("SKIP_BUILD") == "1":
        err(f"SKIP_BUILD=1 — reusing already-pushed {uri}")
        return uri
    _ensure_buildx()
    sh(f"aws ecr get-login-password --region {REGION} | docker login --username AWS --password-stdin {registry}")
    sh(f"docker buildx build --builder {BUILDX_BUILDER} --platform {platform} "
       f"--provenance=false --sbom=false -f {HERE}/{dockerfile} -t {uri} --push {CODE_CTX}")
    err(f"pushed {uri} ({platform})")
    return uri


# ----------------------------------------------------------------------------- orchestration
def cmd_all():
    private_ip = os.environ.get("ADAPTER_PUBLIC_HOST")
    if not private_ip:
        die("set ADAPTER_PUBLIC_HOST to this box PRIVATE IP (the harness dials it back)")

    uris = build_push()
    xrole = exec_role(f"{PREFIX}-exec-role")
    vpc_id, subnets, runtime_sg = default_vpc()
    err(f"default VPC {vpc_id}; runtime SG (harness lands here) {runtime_sg}")

    # No public port: open the pinned adapter port on the trainer box SG, source = runtime SG.
    open_adapter_port(private_ip, runtime_sg)

    # The harness is networkMode=VPC (no internet route), so add private VPC endpoints for the APIs
    # it needs — ECR (image pull), Logs, STS, bedrock-agentcore. Without these the harness container
    # can't pull its image / phone home and lands CREATE_FAILED with an empty log group. Returns the
    # subnets whose AZ is served by every endpoint — the harness must land only in those.
    harness_subnets = ensure_vpc_endpoints(vpc_id, subnets, runtime_sg)
    if not harness_subnets:
        die("no subnet is served by all required VPC endpoints; cannot place VPC-mode harness")

    vpc_net = {"networkMode": "VPC",
               "networkModeConfig": {"securityGroups": [runtime_sg], "subnets": harness_subnets}}
    public_net = {"networkMode": "PUBLIC"}
    # No runtime needs AWS_BEARER_TOKEN_BEDROCK: the harness reaches the LLM through the adapter
    # (auth = the session-id bearer), and both runtimes sign AWS calls with their execution role.
    # We deliberately do NOT inject the token into any runtime env — storing it in the control-plane
    # environmentVariables would spread the secret (visible via describe / CloudTrail) for no reason.
    common = {"AWS_REGION": REGION}

    # CPU sandbox (A3): managed, PUBLIC — reached via IAM, never dials back.
    sandbox_cpu_arn = managed_runtime(
        f"{PREFIX}_sandbox_cpu", uris["sandbox_cpu"], xrole,
        {**common, "SANDBOX_WORKDIR": "/workspace"}, public_net)
    out("AGENTCORE_SANDBOX_CPU_RUNTIME_ARN", sandbox_cpu_arn)

    # GPU sandbox (B3): capacity provider + volume.
    op_arn = operator_role(f"{PREFIX}-operator-role")
    cp_arn = gpu_capacity_provider(f"{PREFIX}_gpu_cp", op_arn)
    sandbox_gpu_arn = gpu_runtime_on_cp(
        f"{PREFIX}_sandbox_gpu", uris["sandbox_gpu"], xrole, cp_arn,
        {**common, "SANDBOX_WORKDIR": "/workspace"})   # SANDBOX_PYLIB/LD_LIBRARY_PATH are in the image ENV
    out("AGENTCORE_SANDBOX_GPU_RUNTIME_ARN", sandbox_gpu_arn)

    # Harness (#1): managed serverless BUT networkMode=VPC so it can dial the private adapter.
    harness_arn = managed_runtime(
        f"{PREFIX}_harness", uris["harness"], xrole, {**common, "HARNESS_KIND": "mini"}, vpc_net)
    out("AGENTCORE_HARNESS_RUNTIME_ARN", harness_arn)

    if os.environ.get("SKIP_PREPARE") != "1":
        prepare_gpu_volume(sandbox_gpu_arn)
    else:
        err("SKIP_PREPARE=1 — run mode=prepare before the first B3 run")

    print("\n# ---- export for the demos (A3 uses the CPU sandbox, B3 the GPU sandbox) ----")
    print(f"export SANDBOX=agentcore-runtime AWS_REGION={REGION}")
    print(f"export AGENTCORE_HARNESS_RUNTIME_ARN={harness_arn}")
    print(f"export ADAPTER_PUBLIC_HOST={private_ip} ADAPTER_PORT={ADAPTER_PORT}")
    print(f"# A3:  export AGENTCORE_SANDBOX_RUNTIME_ARN={sandbox_cpu_arn}")
    print(f"# B3:  export AGENTCORE_SANDBOX_RUNTIME_ARN={sandbox_gpu_arn}")


def cmd_cleanup():
    control = boto3.client("bedrock-agentcore-control", region_name=REGION)
    for logical in ("harness", "sandbox_cpu", "sandbox_gpu"):
        name = f"{PREFIX}_{logical}"
        arn, rid, status = _existing_runtime_arn(control, name)
        if rid:
            try:
                control.delete_agent_runtime(agentRuntimeId=rid)
                err(f"deleted runtime {name}")
            except ClientError as e:
                err(f"delete {name}: {e}")
    time.sleep(10)
    try:
        for cp in control.list_capacity_providers().get("capacityProviders", []):
            if cp.get("name") == f"{PREFIX}_gpu_cp":
                control.delete_capacity_provider(capacityProviderId=cp["capacityProviderId"])
                err(f"deleted capacity provider {cp['capacityProviderId']}")
    except ClientError as e:
        err(f"cp delete: {e}")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    {"all": cmd_all, "cleanup": cmd_cleanup}.get(cmd, lambda: die(f"unknown: {cmd}"))()


if __name__ == "__main__":
    main()
