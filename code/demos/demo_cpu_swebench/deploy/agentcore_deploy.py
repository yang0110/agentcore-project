#!/usr/bin/env python3
"""Vendored AgentCore deploy helper — THIS demo's own copy (identical in every demo's deploy/).

Everything a demo needs to put its sandbox image on AgentCore, with no dependency on code/deploy/:
ECR repos, the buildx/QEMU arm64 cross-builder, the IAM execution + capacity-provider operator roles,
default-VPC lookup + private endpoints (VPC mode), managed microVM Runtimes, capacity providers
(m5 CPU / g6 GPU Instances) and Runtimes on them, and teardown. Library functions are the same ones
code/deploy/deploy_two_runtime.py uses (the demo deploy_*.py scripts import this as `d`); the CLI
below is what the numbered step scripts call.

Architecture is NOT uniform:
  * managed serverless microVM Runtimes run on AgentCore's ARM fleet -> linux/arm64, <= 2 GB image.
  * capacity-provider Instances (m5 CPU, g6 GPU) are x86_64 -> linux/amd64, no image-size cap.

CLI (idempotent; an existing Runtime / capacity provider of the same name is reused):
    agentcore_deploy.py ecr-uri   --repo acs-tasks-cpu                  # create repo if needed, print URI
    agentcore_deploy.py runtime   --name acs_tasks_cpu --image URI [--env K=V ...]
                                  [--instance m5.2xlarge --cp acs_bigimage_cp [--root-gib 40] [--volume-gib 0]]
                                  [--network VPC|PUBLIC]               # prints the Runtime ARN (last line)
    agentcore_deploy.py adapter-ingress                                # SG->SG dial-back rule(s) only
    agentcore_deploy.py delete-runtime --name acs_tasks_cpu
    agentcore_deploy.py delete-cp      --name acs_bigimage_cp

Env: AWS_REGION (us-west-2), NAME_PREFIX (acs), FORCE_RECREATE=1 (delete + recreate the Runtime so a
new image/env is picked up — AgentCore pins the image digest at create time).
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
ADAPTER_PORT = int(os.environ.get("ADAPTER_PORT", "18021"))          # branching's pinned CLI adapter
# The in-session mini_claude loop (HARNESS_WHERE=session) dials the trainer's per-engine adapters back
# on ADAPTER_PORT_BASE .. +ADAPTER_PORT_SPAN-1 (in_session_harness.adapter_bind); opened SG->SG below.
ADAPTER_PORT_BASE = int(os.environ.get("ADAPTER_PORT_BASE", "18100"))
ADAPTER_PORT_SPAN = int(os.environ.get("ADAPTER_PORT_SPAN", "64"))
# Every Runtime is created in networkMode=VPC by default (private endpoints, no public route), so the
# session can dial the adapter back over the VPC. RUNTIME_NETWORK=PUBLIC keeps the old pull-only shape.
RUNTIME_NETWORK = os.environ.get("RUNTIME_NETWORK", "VPC").upper()
GPU_INSTANCE = os.environ.get("GPU_INSTANCE", "g6.xlarge")
MODELS_GIB = int(os.environ.get("MODELS_GIB", "40"))
# Session lifecycle of every Runtime created here. The idle timeout must outlast the worst trainer
# stall (train step + weight sync + eval + checkpoint save): past it AgentCore reclaims the session and
# the same runtimeSessionId silently comes back as an EMPTY sandbox. HEARTBEAT=1 (default on our own
# Runtimes) bridges shorter gaps; this is the backstop. AgentCore range 60 s–8 h; a longer timeout
# bills an abandoned session longer. Applies to NEW Runtimes only — FORCE_RECREATE=1 to change one.
RUNTIME_IDLE_TIMEOUT = int(os.environ.get("RUNTIME_IDLE_TIMEOUT", "3600"))
MICROVM_MAX_LIFETIME = 28800    # managed microVM hard max (8 h)
INSTANCE_MAX_LIFETIME = 86400   # capacity-provider Runtime sessions
CP_IDLE_INSTANCE_TIMEOUT = 600  # an instance with NO session left is released after this (g6 is costly)
MODELS_VOL = "models"
MODELS_MOUNT = "/mnt/models"

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_CTX = os.path.abspath(os.path.join(HERE, ".."))          # build context = THIS demo dir
OPERATOR_MANAGED_POLICY = "arn:aws:iam::aws:policy/BedrockAgentCoreRuntimeInstancesOperatorRolePolicy"

BUILDX_BUILDER = "acsbuilder"

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


def open_adapter_port(private_ip: str, runtime_sg: str, from_port: int = ADAPTER_PORT,
                      to_port: int | None = None) -> None:
    """Allow the harness (in runtime_sg) inbound to the adapter port(s) on the trainer box's SG.
    NO public port: the source is the runtime security group, not a CIDR."""
    ec2 = boto3.client("ec2", region_name=REGION)
    to_port = from_port if to_port is None else to_port
    ports = f"{from_port}" if to_port == from_port else f"{from_port}-{to_port}"
    for sg in trainer_sgs(private_ip):
        try:
            ec2.authorize_security_group_ingress(GroupId=sg, IpPermissions=[{
                "IpProtocol": "tcp", "FromPort": from_port, "ToPort": to_port,
                "UserIdGroupPairs": [{"GroupId": runtime_sg,
                                      "Description": "AgentCore session harness to AnthropicAdapter"}]}])
            err(f"opened tcp/{ports} on {sg} from {runtime_sg}")
        except ClientError as e:
            if "InvalidPermission.Duplicate" in str(e):
                err(f"ingress tcp/{ports} on {sg} from {runtime_sg} already present")
            else:
                raise


def trainer_private_ip() -> str:
    """This box's private IP (the adapter host the sessions dial): ADAPTER_PUBLIC_HOST, else the
    egress interface address (no packet is sent)."""
    import socket
    if os.environ.get("ADAPTER_PUBLIC_HOST"):
        return os.environ["ADAPTER_PUBLIC_HOST"]
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    finally:
        s.close()


def open_dialback(runtime_sg: str) -> None:
    """SG->SG ingress on the trainer box for every adapter a session may dial: the in-session
    mini_claude range and branching's pinned CLI port. Idempotent."""
    ip = trainer_private_ip()
    open_adapter_port(ip, runtime_sg, ADAPTER_PORT_BASE, ADAPTER_PORT_BASE + ADAPTER_PORT_SPAN - 1)
    open_adapter_port(ip, runtime_sg, ADAPTER_PORT)


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


# RUNTIME_EGRESS=nat: a VPC Runtime whose tasks need the internet (Terminal-Bench / SWE-bench tests
# pip/uv/apt at grade time) lands in OUR private subnets whose only default route is an EXISTING NAT
# gateway: outbound-only, still no public IP and no inbound path. We never create a NAT gateway (it is
# billed per hour); if the VPC has none, this stops with instructions. Default (none) = endpoints only.
RUNTIME_EGRESS = os.environ.get("RUNTIME_EGRESS", "none").lower()
EGRESS_TAG = f"{PREFIX}-egress"


def ensure_egress_subnets(vpc_id: str, azs: list[str]) -> list[str]:
    """One private `<prefix>-egress-<az>` subnet per AZ in ``azs``, all on one route table
    (0.0.0.0/0 -> the VPC's NAT gateway, plus the S3 gateway endpoint). Idempotent."""
    import ipaddress
    ec2 = boto3.client("ec2", region_name=REGION)
    nat = os.environ.get("AGENTCORE_NAT_GATEWAY") or next(
        (n["NatGatewayId"] for n in ec2.describe_nat_gateways(Filters=[
            {"Name": "vpc-id", "Values": [vpc_id]}, {"Name": "state", "Values": ["available"]}])["NatGateways"]),
        None)
    if not nat:
        die(f"RUNTIME_EGRESS=nat but {vpc_id} has no available NAT gateway; create one (billed hourly) "
            "and/or set AGENTCORE_NAT_GATEWAY, or deploy with RUNTIME_EGRESS=none")
    all_subnets = ec2.describe_subnets(Filters=[{"Name": "vpc-id", "Values": [vpc_id]}])["Subnets"]
    mine = {sn["AvailabilityZone"]: sn["SubnetId"] for sn in all_subnets
            if any(t["Key"] == "Name" and t["Value"].startswith(EGRESS_TAG) for t in sn.get("Tags", []))}
    used = [ipaddress.ip_network(sn["CidrBlock"]) for sn in all_subnets]
    vpc_net = ipaddress.ip_network(ec2.describe_vpcs(VpcIds=[vpc_id])["Vpcs"][0]["CidrBlock"])
    free = (c for c in vpc_net.subnets(new_prefix=22) if not any(c.overlaps(u) for u in used))
    for az in azs:
        if az in mine:
            continue
        cidr = str(next(free))
        sn = ec2.create_subnet(VpcId=vpc_id, AvailabilityZone=az, CidrBlock=cidr, TagSpecifications=[{
            "ResourceType": "subnet", "Tags": [{"Key": "Name", "Value": f"{EGRESS_TAG}-{az}"}]}])["Subnet"]
        mine[az] = sn["SubnetId"]
        err(f"created private egress subnet {sn['SubnetId']} {cidr} in {az}")

    rts = ec2.describe_route_tables(Filters=[{"Name": "vpc-id", "Values": [vpc_id]},
                                             {"Name": "tag:Name", "Values": [EGRESS_TAG]}])["RouteTables"]
    if rts:
        rt = rts[0]["RouteTableId"]
    else:
        rt = ec2.create_route_table(VpcId=vpc_id, TagSpecifications=[{
            "ResourceType": "route-table", "Tags": [{"Key": "Name", "Value": EGRESS_TAG}]}])["RouteTable"]["RouteTableId"]
        ec2.create_route(RouteTableId=rt, DestinationCidrBlock="0.0.0.0/0", NatGatewayId=nat)
        err(f"created route table {rt}: 0.0.0.0/0 -> {nat}")
    assoc = {a.get("SubnetId") for r in ec2.describe_route_tables(RouteTableIds=[rt])["RouteTables"]
             for a in r.get("Associations", [])}
    for sn in mine.values():
        if sn not in assoc:
            ec2.associate_route_table(RouteTableId=rt, SubnetId=sn)
    # ECR layers come from S3: keep that on the gateway endpoint rather than through the NAT.
    for ep in ec2.describe_vpc_endpoints(Filters=[
            {"Name": "vpc-id", "Values": [vpc_id]},
            {"Name": "service-name", "Values": [f"com.amazonaws.{REGION}.s3"]}])["VpcEndpoints"]:
        if rt not in ep.get("RouteTableIds", []):
            ec2.modify_vpc_endpoint(VpcEndpointId=ep["VpcEndpointId"], AddRouteTableIds=[rt])
    return [mine[az] for az in azs]


def vpc_network(egress: str = RUNTIME_EGRESS) -> dict:
    """networkConfiguration for a VPC-mode managed Runtime: private endpoints always; with
    egress=nat, our NAT-routed private subnets in the same (endpoint-served) AZs."""
    vpc_id, subnets, sg = default_vpc()
    subnets = ensure_vpc_endpoints(vpc_id, subnets, sg)
    ec2 = boto3.client("ec2", region_name=REGION)
    azs = sorted({s["AvailabilityZone"] for s in ec2.describe_subnets(SubnetIds=subnets)["Subnets"]})
    if egress == "nat":
        subnets = ensure_egress_subnets(vpc_id, azs)
    else:
        # The AZ's DEFAULT subnet (no public IP is assigned to a Runtime ENI, so no internet), not
        # whichever subnet the endpoint picked: another team's NAT-routed private subnet would give
        # sessions in that one AZ egress the others lack.
        dflt = {sn["AvailabilityZone"]: sn["SubnetId"] for sn in ec2.describe_subnets(Filters=[
            {"Name": "vpc-id", "Values": [vpc_id]}, {"Name": "default-for-az", "Values": ["true"]}])["Subnets"]}
        subnets = [dflt[az] for az in azs if az in dflt] or subnets
    return {"networkMode": "VPC", "networkModeConfig": {"securityGroups": [sg], "subnets": subnets}}


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


def runtime_lifecycle(max_lifetime: int) -> dict:
    if not 60 <= RUNTIME_IDLE_TIMEOUT <= 28800:
        die(f"RUNTIME_IDLE_TIMEOUT={RUNTIME_IDLE_TIMEOUT} outside AgentCore's 60–28800 s range")
    return {"idleRuntimeSessionTimeout": RUNTIME_IDLE_TIMEOUT,
            "maxLifetime": max(max_lifetime, RUNTIME_IDLE_TIMEOUT)}


def _warn_lifecycle(control, name, rid):
    """A reused Runtime keeps the idle timeout it was created with; say so if it differs, because
    HEARTBEAT_IDLE (run.sh) must stay <= the timeout the Runtime actually has."""
    try:
        lc = control.get_agent_runtime(agentRuntimeId=rid).get("lifecycleConfiguration") or {}
    except ClientError:
        return
    have = lc.get("idleRuntimeSessionTimeout")
    if have is not None and have != RUNTIME_IDLE_TIMEOUT:
        err(f"note: reused runtime {name} keeps idleRuntimeSessionTimeout={have}s "
            f"(RUNTIME_IDLE_TIMEOUT={RUNTIME_IDLE_TIMEOUT}s applies only on create; FORCE_RECREATE=1)")


def _reuse_or_clear(control, name):
    """Return an existing runtime ARN to reuse, or None if it should be (re)created.
    A CREATE_FAILED leftover is deleted so we can create it cleanly."""
    arn, rid, status = _existing_runtime_arn(control, name)
    if arn and "FAILED" in (status or ""):
        err(f"deleting {status} runtime {name} before recreating")
        control.delete_agent_runtime(agentRuntimeId=rid)
        time.sleep(10)
        return None
    if arn:
        _warn_lifecycle(control, name, rid)
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
        lifecycleConfiguration=runtime_lifecycle(MICROVM_MAX_LIFETIME),
        environmentVariables=env,
    )
    _wait_runtime(control, resp["agentRuntimeId"])
    return resp["agentRuntimeArn"]


def capacity_provider(name, operator_arn, instance, root_free_gib=30, volume_gib=0) -> str:
    """Capacity provider = the EC2 Instance fleet a Runtime lands on (m5 CPU or g6 GPU; x86_64).
    Creating it launches NO instances — the first invoke of a session cold-boots one."""
    control = boto3.client("bedrock-agentcore-control", region_name=REGION)
    try:
        for cp in control.list_capacity_providers().get("capacityProviders", []):
            if cp.get("name") == name:
                err(f"reusing capacity provider {name}")
                return cp["capacityProviderArn"]
    except (ClientError, KeyError):
        pass
    _, subnets, sg = default_vpc()
    ec2cfg = {
        "launchTemplateSource": {"launchParameters": {
            "operatingSystem": "LINUX_X86_64",
            "instanceRequirements": {"allowedInstanceTypes": [instance]}}},
        "vpcConfiguration": {"subnets": subnets, "securityGroups": [sg]},
        "rootVolume": {"freeSpaceGiB": root_free_gib, "volumeType": "gp3"},
        "lifecycleConfiguration": {"idleInstanceTimeout": CP_IDLE_INSTANCE_TIMEOUT,
                                   "maxLifetime": INSTANCE_MAX_LIFETIME}}
    if volume_gib:
        ec2cfg["volumes"] = [{"ebsConfiguration": {
            "name": MODELS_VOL, "sizeGiB": volume_gib, "volumeType": "gp3", "encrypted": True}}]
    resp = control.create_capacity_provider(
        name=name, description=f"AgentCore sandbox capacity provider ({instance})",
        permissionsConfiguration={"capacityProviderOperatorRoleArn": operator_arn},
        computeConfiguration={"ec2Configuration": ec2cfg})
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


def gpu_capacity_provider(name, operator_arn) -> str:
    return capacity_provider(name, operator_arn, GPU_INSTANCE, volume_gib=MODELS_GIB)


def runtime_on_cp(name, image_uri, exec_arn, cp_arn, env, volume=False) -> str:
    """Runtime on a capacity provider (Instance substrate). One runtimeSessionId = one whole instance."""
    control = boto3.client("bedrock-agentcore-control", region_name=REGION)
    arn = _reuse_or_clear(control, name)
    if arn:
        err(f"reusing runtime {name} on capacity provider")
        return arn
    kw = {}
    if volume:
        kw["filesystemConfigurations"] = [{"capacityProviderVolume": {
            "volumeName": MODELS_VOL, "mountPath": MODELS_MOUNT}}]
    resp = control.create_agent_runtime(
        agentRuntimeName=name,
        roleArn=exec_arn,
        agentRuntimeArtifact={"containerConfiguration": {"containerUri": image_uri}},
        protocolConfiguration={"serverProtocol": "HTTP"},
        capacityProviderConfiguration={"capacityProviderArn": cp_arn},
        lifecycleConfiguration=runtime_lifecycle(INSTANCE_MAX_LIFETIME),
        environmentVariables=env,
        **kw,
    )
    _wait_runtime(control, resp["agentRuntimeId"])
    return resp["agentRuntimeArn"]


def gpu_runtime_on_cp(name, image_uri, exec_arn, cp_arn, env) -> str:
    return runtime_on_cp(name, image_uri, exec_arn, cp_arn, env, volume=True)


def delete_runtime(name: str, wait: bool = True) -> None:
    control = boto3.client("bedrock-agentcore-control", region_name=REGION)
    _, rid, status = _existing_runtime_arn(control, name)
    if not rid:
        err(f"runtime {name}: not present")
        return
    control.delete_agent_runtime(agentRuntimeId=rid)
    err(f"deleting runtime {name} ({status})")
    for _ in range(60 if wait else 0):          # delete is async; the name frees up only when gone
        if not _existing_runtime_arn(control, name)[1]:
            err(f"runtime {name} deleted")
            return
        time.sleep(5)


def delete_capacity_provider(name: str) -> None:
    control = boto3.client("bedrock-agentcore-control", region_name=REGION)
    for cp in control.list_capacity_providers().get("capacityProviders", []):
        if cp.get("name") == name:
            control.delete_capacity_provider(capacityProviderId=cp["capacityProviderId"])
            err(f"deleted capacity provider {name}")
            return
    err(f"capacity provider {name}: not present")


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
    """Build+push a SINGLE image (used by the redeploy_*_sandbox.py helpers so we don't rebuild the
    whole set). Returns the :latest image URI. Honors SKIP_BUILD=1 to reuse an already-pushed image."""
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


# ----------------------------------------------------------------------------- CLI (step scripts)
def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ecr-uri"); p.add_argument("--repo", required=True)
    p = sub.add_parser("runtime")
    p.add_argument("--name", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--env", action="append", default=[], metavar="K=V")
    p.add_argument("--network", default=RUNTIME_NETWORK, choices=["PUBLIC", "VPC"],
                   help="managed microVM network mode (default RUNTIME_NETWORK=VPC); Instances take the CP's VPC")
    p.add_argument("--egress", default=RUNTIME_EGRESS, choices=["none", "nat"],
                   help="VPC microVM internet egress: none (endpoints only) | nat (existing NAT, outbound only)")
    p.add_argument("--instance", default="", help="capacity-provider instance type (m5.*, g6.*); empty = microVM")
    p.add_argument("--cp", default="", help="capacity provider name (default <PREFIX>_<instance family>_cp)")
    p.add_argument("--root-gib", type=int, default=30)
    p.add_argument("--volume-gib", type=int, default=0)
    p = sub.add_parser("delete-runtime"); p.add_argument("--name", required=True)
    p = sub.add_parser("delete-cp"); p.add_argument("--name", required=True)
    sub.add_parser("adapter-ingress", help="open the SG->SG dial-back rule(s) on this box only")
    a = ap.parse_args()

    if a.cmd == "ecr-uri":
        print(ensure_ecr(a.repo))
    elif a.cmd == "delete-runtime":
        delete_runtime(a.name)
    elif a.cmd == "delete-cp":
        delete_capacity_provider(a.name)
    elif a.cmd == "adapter-ingress":
        open_dialback(default_vpc()[2])
    elif a.cmd == "runtime":
        env = {"AWS_REGION": REGION, "SANDBOX_WORKDIR": "/workspace"}
        env.update(kv.split("=", 1) for kv in a.env)
        if os.environ.get("FORCE_RECREATE") == "1":
            delete_runtime(a.name)
        xrole = exec_role(f"{PREFIX}-exec-role")
        if a.instance:
            op = operator_role(f"{PREFIX}-operator-role")
            cp = capacity_provider(a.cp or f"{PREFIX}_{a.instance.split('.')[0]}_cp", op, a.instance,
                                   root_free_gib=a.root_gib, volume_gib=a.volume_gib)
            arn = runtime_on_cp(a.name, a.image, xrole, cp, env, volume=bool(a.volume_gib))
        else:
            if a.network == "VPC":
                network = vpc_network(a.egress)
            else:
                network = {"networkMode": "PUBLIC"}
            arn = managed_runtime(a.name, a.image, xrole, env, network)
        if a.instance or a.network == "VPC":
            # Instances sit in the CP's default-VPC subnets + SG; VPC microVMs in the same SG. Either
            # way the in-session harness dials this box back, so admit that SG (not a CIDR).
            open_dialback(default_vpc()[2])
        print(arn)


if __name__ == "__main__":
    main()
