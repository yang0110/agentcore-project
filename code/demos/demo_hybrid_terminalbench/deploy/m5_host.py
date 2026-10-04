#!/usr/bin/env python3
"""Provision / inspect / tear down demo_hybrid_terminalbench's m5 Docker host (HOST, real AWS $).

The m5 runs the Terminal-Bench tasks AgentCore microVMs can't (compose, x86-only, > 2 GB): plain EC2,
Amazon Linux 2023 x86_64, Docker + compose, and m5_agent/compose_agent.py on :18300. Network:

    trainer SG  --tcp 18300, 32768-60999-->  m5 SG     (agent API + the clients' published :8080)
    m5 SG       --tcp 18100-18163-------->  trainer SG (the in-session agent loop dials the adapter)

The m5 lands in the `<prefix>-egress-<az>` private subnet of the trainer's AZ (NAT egress for pip /
apt / Docker Hub / uv at grade time; no public IP). No instance role by default; IMDSv2 with hop
limit 1 + the bootstrap's DOCKER-USER rule keep containers away from instance metadata.

    python3 deploy/m5_host.py up         # create (or reuse) the host, wait for the agent -> deploy/m5_host.json
    python3 deploy/m5_host.py status     # agent health (docker, live projects, disk, built tasks)
    python3 deploy/m5_host.py update     # push the current m5_agent/compose_agent.py (agent restarts)
    python3 deploy/m5_host.py down       # terminate the instance, drop the SG rules + SG
    python3 deploy/m5_host.py roles      # IAM: builder (ECR push to <prefix>-tbhdk-*) + episode (ECR pull)
    python3 deploy/m5_host.py bake-ami   # per-episode mode: a Docker-ready AMI (no task images) -> state["episode"]
    python3 deploy/m5_host.py gc-episodes  # terminate every per-episode instance (tag acs-tbh-m5-ep)

Two m5 modes (TB_M5_MODE in run.sh):
  shared       (the instance above) many episodes at once, one compose project each — shared kernel
  per_episode  ONE fresh EC2 instance per episode (M5_EP_INSTANCE_TYPE=m5.xlarge, M5_EP_DISK_GB=40), from
               the baked AMI, pulling that task's images from ECR; terminated after grading. The
               instance above then only BUILDS images and pushes them to ECR.

Knobs: M5_INSTANCE_TYPE=m5.xlarge (builder: builds + ECR push only)  M5_DISK_GB=500  M5_AZ=<trainer AZ>  M5_MAX_PROJECTS=16
M5_CLIENT_MEM=8g  M5_CLIENT_CPUS=4  M5_INSTANCE_PROFILE=<name> (optional, e.g. for SSM).
"""
from __future__ import annotations

import base64
import gzip
import json
import os
import secrets
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

HERE = Path(__file__).resolve().parent
DEMO_DIR = HERE.parent
sys.path.insert(0, str(HERE))
import agentcore_deploy as d  # noqa: E402

STATE = HERE / "m5_host.json"                    # gitignored: instance id, IP, token
AGENT = DEMO_DIR / "m5_agent" / "compose_agent.py"
BOOTSTRAP = DEMO_DIR / "m5_agent" / "bootstrap.sh"
NAME = f"{d.PREFIX}-tbh-m5"
AGENT_PORT = int(os.environ.get("M5_AGENT_PORT", "18300"))
EPHEMERAL = (32768, 60999)                       # Docker's published-port range (Linux ip_local_port_range)
INSTANCE_TYPE = os.environ.get("M5_INSTANCE_TYPE", "m5.xlarge")
DISK_GB = int(os.environ.get("M5_DISK_GB", "500"))
COMPOSE_VERSION = os.environ.get("M5_COMPOSE_VERSION", "v5.5.1")   # = the trainer host's (tested here)
BUILDX_VERSION = os.environ.get("M5_BUILDX_VERSION", "v0.37.1")


def load_state() -> dict:
    if not STATE.exists():
        d.die(f"{STATE} not found — provision the m5 first: python3 deploy/m5_host.py up")
    return json.loads(STATE.read_text())


def call(state: dict, payload: dict, timeout: float = 120.0) -> dict:
    """One authenticated op on the m5 agent. Raises on HTTP errors (the body's error text included)."""
    req = urllib.request.Request(
        f"http://{state['ip']}:{state['port']}/op", data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json", "X-ACS-Token": state["token"]})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        raise RuntimeError(f"m5 agent {payload.get('op')}: HTTP {e.code} {body[-800:]}") from None


def put_tar(state: dict, data: bytes, dest: str) -> None:
    """Upload a tar.gz to the agent in 3 MB base64 chunks (dest = 'ctx' or 'task:<name>')."""
    b64, chunk, key = base64.b64encode(data).decode(), 3 * 2**20, secrets.token_hex(6)
    for i in range(0, max(1, len(b64)), chunk):
        call(state, {"op": "put_tar", "dest": dest, "key": key, "b64": b64[i:i + chunk],
                     "more": i + chunk < len(b64)}, timeout=300)


EP_NAME = f"{d.PREFIX}-tbh-m5-ep"                  # tag Name of per-episode instances
ECR_PREFIX = f"{d.PREFIX}-tbhdk"                   # ECR repos <prefix>-tbhdk-<task>-<service>
ROLE_BUILDER, ROLE_EPISODE = f"{d.PREFIX}-tbh-m5-builder", f"{d.PREFIX}-tbh-m5-episode"


def _ec2():
    return boto3.client("ec2", region_name=d.REGION)


def _ensure_profile(name: str, actions: list[str]) -> str:
    """An EC2 instance profile whose role may do `actions` on the <prefix>-tbhdk-* ECR repos only."""
    iam = boto3.client("iam")
    trust = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"},
                                                     "Action": "sts:AssumeRole"}]}
    try:
        iam.create_role(RoleName=name, AssumeRolePolicyDocument=json.dumps(trust),
                        Description="demo_hybrid_terminalbench m5 (ECR only)")
    except ClientError as e:
        if "EntityAlreadyExists" not in str(e):
            raise
    acct = d.account()
    iam.put_role_policy(RoleName=name, PolicyName="ecr", PolicyDocument=json.dumps({"Version": "2012-10-17", "Statement": [
        {"Effect": "Allow", "Action": "ecr:GetAuthorizationToken", "Resource": "*"},
        {"Effect": "Allow", "Action": actions,
         "Resource": f"arn:aws:ecr:{d.REGION}:{acct}:repository/{ECR_PREFIX}-*"}]}))
    try:
        iam.create_instance_profile(InstanceProfileName=name)
        iam.add_role_to_instance_profile(InstanceProfileName=name, RoleName=name)
        time.sleep(10)                                # instance profiles take a moment to be usable
    except ClientError as e:
        if "EntityAlreadyExists" not in str(e):
            raise
    return name


PULL = ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer", "ecr:BatchCheckLayerAvailability"]
PUSH = PULL + ["ecr:InitiateLayerUpload", "ecr:UploadLayerPart", "ecr:CompleteLayerUpload", "ecr:PutImage",
               "ecr:CreateRepository", "ecr:DescribeRepositories"]


def roles() -> int:
    """Builder role (push) on the shared instance, episode role (pull) for per-episode instances."""
    st = load_state()
    _ensure_profile(ROLE_BUILDER, PUSH)
    _ensure_profile(ROLE_EPISODE, PULL)
    ec2 = _ec2()
    cur = ec2.describe_iam_instance_profile_associations(
        Filters=[{"Name": "instance-id", "Values": [st["instance_id"]]}])["IamInstanceProfileAssociations"]
    if not cur:
        for attempt in range(6):
            try:
                ec2.associate_iam_instance_profile(IamInstanceProfile={"Name": ROLE_BUILDER}, InstanceId=st["instance_id"])
                break
            except ClientError as e:
                if attempt == 5 or "Invalid IAM Instance Profile" not in str(e):
                    raise
                time.sleep(10)
        d.err(f"attached {ROLE_BUILDER} to the builder {st['instance_id']}")
    st["episode_profile"] = ROLE_EPISODE
    st["ecr_prefix"] = f"{d.account()}.dkr.ecr.{d.REGION}.amazonaws.com/{ECR_PREFIX}"
    STATE.write_text(json.dumps(st, indent=2))
    print(f"# roles: builder {ROLE_BUILDER} (push), episode {ROLE_EPISODE} (pull) on {ECR_PREFIX}-*")
    return 0


def episode_user_data(token: str) -> bytes:
    return gzip.compress(("#!/bin/bash\nset -e\nexec > /var/log/acs-episode.log 2>&1\numask 077\n"
                          f"echo '{token}' > /srv/acs/token\nsystemctl restart acs-compose-agent\n").encode())


def bake_ami() -> int:
    """Launch a small instance with the full bootstrap, wait for its agent, image it (no task images:
    the episode pulls its own from ECR), terminate it. Records the AMI in state["episode"]."""
    st = load_state()
    if not st.get("episode_profile"):
        d.die("run `python3 deploy/m5_host.py roles` first")
    ec2 = _ec2()
    ami = boto3.client("ssm", region_name=d.REGION).get_parameter(
        Name="/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64")["Parameter"]["Value"]
    disk = int(os.environ.get("M5_EP_DISK_GB", "40"))
    tmp_token = secrets.token_urlsafe(32)
    inst = ec2.run_instances(
        ImageId=ami, InstanceType="m5.large", MinCount=1, MaxCount=1, UserData=_user_data(tmp_token),
        NetworkInterfaces=[{"DeviceIndex": 0, "SubnetId": st["subnet"], "Groups": [st["sg"]],
                            "AssociatePublicIpAddress": False}],
        BlockDeviceMappings=[{"DeviceName": "/dev/xvda", "Ebs": {"VolumeSize": disk, "VolumeType": "gp3",
                                                                 "DeleteOnTermination": True}}],
        MetadataOptions={"HttpTokens": "required", "HttpPutResponseHopLimit": 1, "HttpEndpoint": "enabled"},
        TagSpecifications=[{"ResourceType": "instance", "Tags": [{"Key": "Name", "Value": f"{NAME}-bake"}]}])["Instances"][0]
    iid = inst["InstanceId"]
    d.err(f"bake: launched {iid}; waiting for docker + agent")
    try:
        ec2.get_waiter("instance_running").wait(InstanceIds=[iid])
        ip = ec2.describe_instances(InstanceIds=[iid])["Reservations"][0]["Instances"][0]["PrivateIpAddress"]
        _wait_agent({"ip": ip, "port": AGENT_PORT, "token": tmp_token})
        img = ec2.create_image(InstanceId=iid, Name=f"{EP_NAME}-{time.strftime('%Y%m%d-%H%M%S')}",
                               Description="demo_hybrid_terminalbench per-episode m5: Docker + compose agent",
                               TagSpecifications=[{"ResourceType": "image", "Tags": [{"Key": "Name", "Value": EP_NAME}]}])
        d.err(f"bake: creating AMI {img['ImageId']} (reboots the bake instance)")
        ec2.get_waiter("image_available").wait(ImageIds=[img["ImageId"]], WaiterConfig={"Delay": 15, "MaxAttempts": 120})
    finally:
        ec2.terminate_instances(InstanceIds=[iid])
    st["episode"] = {"ami": img["ImageId"], "subnet": st["subnet"], "sg": st["sg"], "profile": st["episode_profile"],
                     "instance_type": os.environ.get("M5_EP_INSTANCE_TYPE", "m5.xlarge"), "disk_gb": disk,
                     "agent_port": AGENT_PORT, "ecr_prefix": st.get("ecr_prefix", ""), "tag": EP_NAME}
    STATE.write_text(json.dumps(st, indent=2))
    print(f"# per-episode AMI {img['ImageId']} ({st['episode']['instance_type']}, {disk} GB) -> {STATE}")
    return 0


def gc_episodes() -> int:
    ec2 = _ec2()
    ids = [i["InstanceId"] for r in ec2.describe_instances(Filters=[
        {"Name": "tag:Name", "Values": [EP_NAME]},
        {"Name": "instance-state-name", "Values": ["pending", "running", "stopping", "stopped"]}])["Reservations"]
        for i in r["Instances"]]
    if ids:
        ec2.terminate_instances(InstanceIds=ids)
    print(f"# terminated {len(ids)} per-episode instance(s)")
    return 0


def _trainer_az() -> str:
    if os.environ.get("M5_AZ"):
        return os.environ["M5_AZ"]
    ip = d.trainer_private_ip()
    eni = _ec2().describe_network_interfaces(
        Filters=[{"Name": "private-ip-address", "Values": [ip]}])["NetworkInterfaces"]
    return eni[0]["AvailabilityZone"] if eni else f"{d.REGION}b"


def _ensure_sg(vpc_id: str) -> str:
    ec2 = _ec2()
    got = ec2.describe_security_groups(Filters=[{"Name": "vpc-id", "Values": [vpc_id]},
                                                {"Name": "group-name", "Values": [NAME]}])["SecurityGroups"]
    sg = got[0]["GroupId"] if got else ec2.create_security_group(
        GroupName=NAME, VpcId=vpc_id, Description="demo_hybrid_terminalbench m5 Docker host")["GroupId"]
    trainer_ip = d.trainer_private_ip()
    for tsg in d.trainer_sgs(trainer_ip):
        for lo, hi, what in ((AGENT_PORT, AGENT_PORT, "compose agent"), (*EPHEMERAL, "episode client servers")):
            try:
                ec2.authorize_security_group_ingress(GroupId=sg, IpPermissions=[{
                    "IpProtocol": "tcp", "FromPort": lo, "ToPort": hi,
                    "UserIdGroupPairs": [{"GroupId": tsg, "Description": f"trainer to {what}"}]}])
                d.err(f"opened tcp/{lo}-{hi} on {sg} from trainer {tsg}")
            except ClientError as e:
                if "InvalidPermission.Duplicate" not in str(e):
                    raise
    # The episodes' in-session agent loops dial the trainer's adapters back (same rule as AgentCore's).
    d.open_adapter_port(trainer_ip, sg, d.ADAPTER_PORT_BASE, d.ADAPTER_PORT_BASE + d.ADAPTER_PORT_SPAN - 1)
    return sg


def _user_data(token: str) -> bytes:
    text = BOOTSTRAP.read_text()
    for k, v in {"__TOKEN__": token, "__AGENT_B64__": base64.b64encode(AGENT.read_bytes()).decode(),
                 "__AGENT_PORT__": str(AGENT_PORT), "__MAX_PROJECTS__": os.environ.get("M5_MAX_PROJECTS", "16"),
                 "__CLIENT_MEM__": os.environ.get("M5_CLIENT_MEM", "8g"),
                 "__CLIENT_CPUS__": os.environ.get("M5_CLIENT_CPUS", "4"),
                 "__COMPOSE_VERSION__": COMPOSE_VERSION, "__BUILDX_VERSION__": BUILDX_VERSION}.items():
        text = text.replace(k, v)
    raw = gzip.compress(text.encode())
    if len(raw) > 16384:
        d.die(f"user-data is {len(raw)} bytes gzipped (EC2 cap 16384)")
    return raw          # bytes: boto3 base64-encodes UserData itself (the cap applies before encoding)


def _find_instance() -> dict | None:
    for r in _ec2().describe_instances(Filters=[
            {"Name": "tag:Name", "Values": [NAME]},
            {"Name": "instance-state-name", "Values": ["pending", "running", "stopping", "stopped"]}])["Reservations"]:
        for i in r["Instances"]:
            return i
    return None


def _wait_agent(state: dict, deadline_s: float = 1200) -> dict:
    end = time.time() + deadline_s
    while True:
        try:
            return call(state, {"op": "health"}, timeout=15)
        except Exception as e:  # noqa: BLE001 — still booting
            if time.time() > end:
                d.die(f"m5 agent at {state['ip']}:{state['port']} not healthy after {deadline_s:.0f}s: {e}")
            time.sleep(10)


def up() -> int:
    vpc_id, _subnets, _sg = d.default_vpc()
    az = _trainer_az()
    subnet = d.ensure_egress_subnets(vpc_id, [az])[0]
    sg = _ensure_sg(vpc_id)
    inst = _find_instance()
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    if inst is None:
        token = secrets.token_urlsafe(32)
        ami = boto3.client("ssm", region_name=d.REGION).get_parameter(
            Name="/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64")["Parameter"]["Value"]
        kw = {}
        if os.environ.get("M5_INSTANCE_PROFILE"):
            kw["IamInstanceProfile"] = {"Name": os.environ["M5_INSTANCE_PROFILE"]}
        inst = _ec2().run_instances(
            ImageId=ami, InstanceType=INSTANCE_TYPE, MinCount=1, MaxCount=1, UserData=_user_data(token),
            NetworkInterfaces=[{"DeviceIndex": 0, "SubnetId": subnet, "Groups": [sg],
                                "AssociatePublicIpAddress": False}],
            BlockDeviceMappings=[{"DeviceName": "/dev/xvda",
                                  "Ebs": {"VolumeSize": DISK_GB, "VolumeType": "gp3", "DeleteOnTermination": True}}],
            MetadataOptions={"HttpTokens": "required", "HttpPutResponseHopLimit": 1, "HttpEndpoint": "enabled"},
            TagSpecifications=[{"ResourceType": "instance", "Tags": [{"Key": "Name", "Value": NAME}]},
                               {"ResourceType": "volume", "Tags": [{"Key": "Name", "Value": NAME}]}],
            **kw)["Instances"][0]
        d.err(f"launched {inst['InstanceId']} ({INSTANCE_TYPE}, {DISK_GB} GB gp3) in {subnet} / {az}")
        state = {"token": token}
    elif not state.get("token"):
        d.die(f"{NAME} ({inst['InstanceId']}) exists but {STATE} has no token — run `down` and `up` again")
    elif inst["State"]["Name"] in ("stopped", "stopping"):
        _ec2().get_waiter("instance_stopped").wait(InstanceIds=[inst["InstanceId"]])
        _ec2().start_instances(InstanceIds=[inst["InstanceId"]])
        d.err(f"starting stopped {inst['InstanceId']}")
    _ec2().get_waiter("instance_running").wait(InstanceIds=[inst["InstanceId"]])
    inst = _ec2().describe_instances(InstanceIds=[inst["InstanceId"]])["Reservations"][0]["Instances"][0]
    state.update(instance_id=inst["InstanceId"], ip=inst["PrivateIpAddress"], port=AGENT_PORT, sg=sg,
                 subnet=subnet, az=az, instance_type=inst["InstanceType"])
    STATE.write_text(json.dumps(state, indent=2))
    os.chmod(STATE, 0o600)
    d.err(f"waiting for the compose agent on {state['ip']}:{AGENT_PORT} (first boot installs docker: ~3-5 min)")
    h = _wait_agent(state)
    print(json.dumps({k: h.get(k) for k in ("docker", "live", "max_projects", "disk_free_gb")}))
    print(f"# m5 host {state['instance_id']} {state['ip']} -> {STATE}")
    return 0


def status() -> int:
    print(json.dumps(call(load_state(), {"op": "health"}, timeout=30), indent=2))
    return 0


def update() -> int:
    st = load_state()
    call(st, {"op": "self_update", "b64": base64.b64encode(AGENT.read_bytes()).decode()})
    time.sleep(3)
    print(json.dumps(_wait_agent(st, 120).get("docker")))
    return 0


def down() -> int:
    ec2 = _ec2()
    inst = _find_instance()
    if inst:
        ec2.terminate_instances(InstanceIds=[inst["InstanceId"]])
        d.err(f"terminating {inst['InstanceId']}")
        ec2.get_waiter("instance_terminated").wait(InstanceIds=[inst["InstanceId"]])
    vpc_id, _s, _g = d.default_vpc()
    got = ec2.describe_security_groups(Filters=[{"Name": "vpc-id", "Values": [vpc_id]},
                                                {"Name": "group-name", "Values": [NAME]}])["SecurityGroups"]
    if got:
        sg = got[0]["GroupId"]
        for tsg in d.trainer_sgs(d.trainer_private_ip()):      # the dial-back rule references our SG
            perms = [p for p in ec2.describe_security_groups(GroupIds=[tsg])["SecurityGroups"][0]["IpPermissions"]
                     if any(g.get("GroupId") == sg for g in p.get("UserIdGroupPairs", []))]
            for p in perms:
                p["UserIdGroupPairs"] = [g for g in p["UserIdGroupPairs"] if g.get("GroupId") == sg]
                for k in ("IpRanges", "Ipv6Ranges", "PrefixListIds"):
                    p.pop(k, None)
            if perms:
                ec2.revoke_security_group_ingress(GroupId=tsg, IpPermissions=perms)
                d.err(f"revoked {len(perms)} rule(s) on {tsg} from {sg}")
        ec2.delete_security_group(GroupId=sg)
        d.err(f"deleted SG {sg}")
    STATE.unlink(missing_ok=True)
    print("# m5 host removed (the shared egress subnet is kept)")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    fn = {"up": up, "status": status, "update": update, "down": down, "roles": roles, "bake-ami": bake_ami,
          "gc-episodes": gc_episodes}.get(cmd)
    if fn is None:
        d.die(f"usage: {sys.argv[0]} up|status|update|down|roles|bake-ami|gc-episodes")
    sys.exit(fn())
