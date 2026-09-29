#!/usr/bin/env python3
"""ECS-on-EC2 substrate for demo_cpu_swebench_ecs: render, create, smoke-test and delete it.

    python3 deploy/ecs_deploy.py render     # READ-ONLY: discover VPC/subnets/trainer SGs/ECS AMI ->
                                            #   deploy/ecs/rendered/*.json + user_data.sh, print the plan.
    ECS_CREATE=1 python3 deploy/ecs_deploy.py up
                                            # (billed: EC2 nodes) instance role + profile, node/task SGs +
                                            #   dial-back rules, launch template, ASG, capacity provider with
                                            #   managed scaling, cluster, one task definition per image
                                            #   -> swebench/taskdef_by_task.json + swebench/ecs_network.json,
                                            #   then one smoke task
    python3 deploy/ecs_deploy.py smoke      # one task from the first image: open timings + isolation proof
    python3 deploy/ecs_deploy.py status
    ECS_DELETE=1 python3 deploy/ecs_deploy.py down   # stop tasks, delete cluster/CP/ASG/LT/task defs/SGs/roles

`up` without ECS_CREATE=1 only renders and stops, so a stray `bash 4_deploy_runtime.sh` cannot start
nodes. Pure boto3 (no CLI, no CloudFormation), idempotent: every resource is looked up by name first.

What gets created (all named ${NAME_PREFIX}-swe-ecs*):
  * nodes: ECS-optimized AL2023 AMI (SSM parameter), ECS_NODE_TYPE, root gp3 ECS_NODE_DISK_GIB, IMDSv2
    required; user data sets ECS_CLUSTER, ECS_AWSVPC_BLOCK_IMDS=true (tasks cannot reach the node's
    credentials), ECS_IMAGE_PULL_BEHAVIOR=prefer-cached, long image-cleanup ages, and (ECS_PREPULL=1)
    pulls every pool image once at boot so an episode's task skips the pull;
  * awsvpcTrunking enabled for the instance role (an m5.2xlarge gets 40+ task ENIs instead of 3);
  * ASG in the ${NAME_PREFIX}-egress-* NAT subnets (they must exist — any AgentCore demo's
    RUNTIME_EGRESS=nat deploy creates them) + a capacity provider with managed scaling (target 100 %)
    and managed termination protection, so queued RunTasks add nodes and idle nodes go away;
  * node SG: no ingress, all egress (ECR/ECS/logs via NAT). Task SG: ingress tcp/8080 from the trainer
    SGs only; egress ONLY to the trainer's adapter ports (the default allow-all egress rule is revoked) —
    the "egress none" of the AgentCore demo; plus the trainer-side SG->SG dial-back rules;
  * task definition per image: awsvpc, EC2, ECS_TASK_CPU / ECS_TASK_MEMORY, all Linux capabilities
    dropped, init process, NO task role, entrypoint wrapped in `timeout -s KILL ECS_TASK_MAX_LIFETIME`,
    awslogs to /acs/swe-ecs.

Knobs: ECS_CLUSTER (acs-swe-ecs)  ECS_NODE_TYPE (m5.2xlarge)  ECS_NODES (4)  ECS_NODES_MIN (0)
       ECS_NODES_MAX (16)  ECS_NODE_DISK_GIB (300)  ECS_TASK_CPU (1536)  ECS_TASK_MEMORY (8192)
       ECS_TASK_MEMORY_RESERVATION (6144)  ECS_TASK_MAX_LIFETIME (7200)  ECS_PREPULL (1)
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

HERE = Path(__file__).resolve().parent
DEMO_DIR = HERE.parent
ECS_DIR = HERE / "ecs"
RENDERED = ECS_DIR / "rendered"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(DEMO_DIR))

import agentcore_deploy as d  # noqa: E402

REGION, PREFIX = d.REGION, d.PREFIX
CLUSTER = os.environ.setdefault("ECS_CLUSTER", f"{PREFIX}-swe-ecs")
CP_NAME = os.environ.setdefault("ECS_CAPACITY_PROVIDER", f"{PREFIX}-swe-ecs-cp")
ASG_NAME = f"{PREFIX}-swe-ecs-asg"
LT_NAME = f"{PREFIX}-swe-ecs-lt"
ROLE_NAME = f"{PREFIX}-swe-ecs-instance-role"
NODE_SG_NAME = f"{PREFIX}-swe-ecs-node"
TASK_SG_NAME = f"{PREFIX}-swe-ecs-task"
LOG_GROUP = "/acs/swe-ecs"
FAMILY_PREFIX = f"{PREFIX}-swe-ecs-"
AMI_PARAM = "/aws/service/ecs/optimized-ami/amazon-linux-2023/recommended/image_id"
IMAGE_MAP = DEMO_DIR / "swebench" / "image_by_task.json"        # deploy_swebench.py: {instance -> URI@digest}
TASKDEF_MAP = DEMO_DIR / "swebench" / "taskdef_by_task.json"    # {instance -> task definition ARN}, run.sh
NETWORK_JSON = DEMO_DIR / "swebench" / "ecs_network.json"       # cluster/CP/subnets/task SG, run.sh
SERVER_CMD = ["/opt/acs/py/bin/python3", "/opt/acs/acs_swe_http_server.py"]   # the wrapper image's CMD

env = os.environ.get


def _c(svc: str):
    return boto3.client(svc, region_name=REGION)


# ----------------------------------------------------------------------------- discovery (read-only)
def discover() -> dict:
    ec2 = _c("ec2")
    vpc_id, _, _ = d.default_vpc()
    subnets = ec2.describe_subnets(Filters=[
        {"Name": "vpc-id", "Values": [vpc_id]}, {"Name": "tag:Name", "Values": [f"{d.EGRESS_TAG}-*"]}])["Subnets"]
    if not subnets:
        d.die(f"no {d.EGRESS_TAG}-* NAT subnets in {vpc_id}. Create them once with any AgentCore demo's "
              "RUNTIME_EGRESS=nat deploy (agentcore_deploy.ensure_egress_subnets).")
    ip = d.trainer_private_ip()
    ami = _c("ssm").get_parameter(Name=AMI_PARAM)["Parameter"]["Value"]
    images = json.loads(IMAGE_MAP.read_text()) if IMAGE_MAP.exists() else {}
    return {"vpc_id": vpc_id, "subnets": sorted(subnets, key=lambda s: s["AvailabilityZone"]),
            "trainer_ip": ip, "trainer_sgs": d.trainer_sgs(ip), "ami": ami, "images": images,
            "caller": _c("sts").get_caller_identity()["Arn"]}


def user_data(images: list[str]) -> str:
    cfg = "\n".join([
        f"ECS_CLUSTER={CLUSTER}",
        "ECS_AWSVPC_BLOCK_IMDS=true",             # tasks never see the node's instance-role credentials
        "ECS_IMAGE_PULL_BEHAVIOR=prefer-cached",  # digest-pinned images: a cached copy is always right
        "ECS_IMAGE_MINIMUM_CLEANUP_AGE=24h",      # keep the pool cached across a whole run
        "ECS_NUM_IMAGES_DELETE_PER_CYCLE=2",
        "ECS_ENGINE_TASK_CLEANUP_WAIT_DURATION=5m",
        "ECS_CONTAINER_STOP_TIMEOUT=2s",
    ])
    script = ["#!/bin/bash", "set -u", f"cat >> /etc/ecs/ecs.config <<'EOF'\n{cfg}\nEOF"]
    if images and env("ECS_PREPULL", "1") == "1":
        registry = images[0].split("/", 1)[0]
        pulls = " ".join(images)
        # In the background: the ECS agent registers (and can take tasks) while the pull runs; a task
        # that lands before its image is cached just pulls it itself.
        script.append(
            "( until docker info >/dev/null 2>&1; do sleep 2; done\n"
            f"  aws ecr get-login-password --region {REGION} | docker login --username AWS --password-stdin {registry}\n"
            f"  for img in {pulls}; do docker pull -q \"$img\" || echo \"prepull failed: $img\"; done\n"
            ") > /var/log/acs-prepull.log 2>&1 &")
    return "\n".join(script) + "\n"


def family(task: str) -> str:
    return FAMILY_PREFIX + task.lower().replace("__", "-").replace(".", "-")


def taskdef(task: str, image: str) -> dict:
    lifetime = env("ECS_TASK_MAX_LIFETIME", "7200")
    return {
        "family": family(task),
        "networkMode": "awsvpc",
        "requiresCompatibilities": ["EC2"],
        "cpu": env("ECS_TASK_CPU", "1536"),
        "memory": env("ECS_TASK_MEMORY", "8192"),                # hard limit (OOM-killed above)
        # no taskRoleArn: the agent gets no AWS credentials. No executionRoleArn: on EC2 the agent pulls
        # from ECR and writes awslogs with the instance role.
        "containerDefinitions": [{
            "name": "sandbox",
            "image": image,                                     # ECR URI pinned by digest
            "essential": True,
            "memoryReservation": int(env("ECS_TASK_MEMORY_RESERVATION", "6144")),
            "portMappings": [{"containerPort": 8080, "protocol": "tcp"}],
            "entryPoint": ["timeout", "-s", "KILL", lifetime],  # caps a leaked task (no idle reclaim)
            "command": SERVER_CMD,
            "linuxParameters": {"capabilities": {"drop": ["ALL"]}, "initProcessEnabled": True},
            "privileged": False,
            "stopTimeout": 2,
            "ulimits": [{"name": "nofile", "softLimit": 65536, "hardLimit": 65536}],
            "logConfiguration": {"logDriver": "awslogs", "options": {
                "awslogs-group": LOG_GROUP, "awslogs-region": REGION, "awslogs-stream-prefix": "swe"}},
        }],
    }


def _spec_hash(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def render(e: dict) -> None:
    RENDERED.mkdir(parents=True, exist_ok=True)
    images = sorted(set(e["images"].values()))
    (RENDERED / "user_data.sh").write_text(user_data(images))
    (RENDERED / "plan.json").write_text(json.dumps({
        "cluster": CLUSTER, "capacity_provider": CP_NAME, "asg": ASG_NAME, "launch_template": LT_NAME,
        "region": REGION, "vpc": e["vpc_id"], "ami": e["ami"],
        "node_type": env("ECS_NODE_TYPE", "m5.2xlarge"), "nodes": env("ECS_NODES", "4"),
        "nodes_min": env("ECS_NODES_MIN", "0"), "nodes_max": env("ECS_NODES_MAX", "16"),
        "subnets": [f"{s['SubnetId']}/{s['AvailabilityZone']}" for s in e["subnets"]],
        "trainer_ip": e["trainer_ip"], "trainer_sgs": e["trainer_sgs"], "pool_images": len(images)},
        indent=2))
    if e["images"]:
        task, image = next(iter(sorted(e["images"].items())))
        (RENDERED / "taskdef-example.json").write_text(json.dumps(taskdef(task, image), indent=2))


# ----------------------------------------------------------------------------- IAM
def ensure_instance_role() -> tuple[str, str]:
    """(role ARN, instance profile ARN)."""
    iam = _c("iam")
    trust = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"},
                                                     "Action": "sts:AssumeRole"}]}
    try:
        role = iam.create_role(RoleName=ROLE_NAME, AssumeRolePolicyDocument=json.dumps(trust),
                               Description="demo_cpu_swebench_ecs container instances")["Role"]
        d.err(f"created IAM role {ROLE_NAME}")
    except iam.exceptions.EntityAlreadyExistsException:
        role = iam.get_role(RoleName=ROLE_NAME)["Role"]
    for pol in ("service-role/AmazonEC2ContainerServiceforEC2Role", "AmazonSSMManagedInstanceCore"):
        iam.attach_role_policy(RoleName=ROLE_NAME, PolicyArn=f"arn:aws:iam::aws:policy/{pol}")
    try:
        iam.create_instance_profile(InstanceProfileName=ROLE_NAME)
        iam.add_role_to_instance_profile(InstanceProfileName=ROLE_NAME, RoleName=ROLE_NAME)
        d.err(f"created instance profile {ROLE_NAME}")
        time.sleep(10)                                  # IAM propagation before a launch template uses it
    except iam.exceptions.EntityAlreadyExistsException:
        pass
    prof = iam.get_instance_profile(InstanceProfileName=ROLE_NAME)["InstanceProfile"]
    # Trunking is opted in per principal: the instances' role. Without it an m5.2xlarge runs 3 awsvpc tasks.
    _c("ecs").put_account_setting(name="awsvpcTrunking", value="enabled", principalArn=role["Arn"])
    return role["Arn"], prof["Arn"]


# ----------------------------------------------------------------------------- security groups
def _find_sg(ec2, vpc_id: str, name: str) -> str | None:
    g = ec2.describe_security_groups(Filters=[{"Name": "vpc-id", "Values": [vpc_id]},
                                              {"Name": "group-name", "Values": [name]}])["SecurityGroups"]
    return g[0]["GroupId"] if g else None


def _ensure_sg(ec2, vpc_id: str, name: str, desc: str) -> tuple[str, bool]:
    sg = _find_sg(ec2, vpc_id, name)
    if sg:
        return sg, False
    sg = ec2.create_security_group(GroupName=name, Description=desc, VpcId=vpc_id, TagSpecifications=[{
        "ResourceType": "security-group", "Tags": [{"Key": "Name", "Value": name}]}])["GroupId"]
    d.err(f"created SG {name} {sg}")
    return sg, True


def _authorize(fn, **kw) -> None:
    try:
        fn(**kw)
    except ClientError as e:
        if "InvalidPermission.Duplicate" not in str(e):
            raise


def ensure_sgs(e: dict) -> tuple[str, str]:
    """(node SG, task SG)."""
    ec2 = _c("ec2")
    node_sg, _ = _ensure_sg(ec2, e["vpc_id"], NODE_SG_NAME, "demo_cpu_swebench_ecs nodes: no ingress")
    task_sg, new = _ensure_sg(ec2, e["vpc_id"], TASK_SG_NAME,
                              "demo_cpu_swebench_ecs sandbox tasks: 8080 from trainer, egress adapter only")
    if new:   # drop the default allow-all egress: a task may only dial the trainer's adapter ports
        _authorize(ec2.revoke_security_group_egress, GroupId=task_sg, IpPermissions=[
            {"IpProtocol": "-1", "IpRanges": [{"CidrIp": "0.0.0.0/0"}]}])
    for tsg in e["trainer_sgs"]:
        _authorize(ec2.authorize_security_group_ingress, GroupId=task_sg, IpPermissions=[{
            "IpProtocol": "tcp", "FromPort": 8080, "ToPort": 8080,
            "UserIdGroupPairs": [{"GroupId": tsg, "Description": "trainer to SWE-bench sandbox tasks"}]}])
        for lo, hi in ((d.ADAPTER_PORT_BASE, d.ADAPTER_PORT_BASE + d.ADAPTER_PORT_SPAN - 1),
                       (d.ADAPTER_PORT, d.ADAPTER_PORT)):
            _authorize(ec2.authorize_security_group_egress, GroupId=task_sg, IpPermissions=[{
                "IpProtocol": "tcp", "FromPort": lo, "ToPort": hi,
                "UserIdGroupPairs": [{"GroupId": tsg, "Description": "sandbox task to trainer adapter"}]}])
    d.open_dialback(task_sg)                             # trainer side of the same dial-back
    return node_sg, task_sg


def revoke_trainer_rules(sg: str) -> None:
    """Drop every trainer-SG rule that references `sg`, or deleting `sg` fails (DependencyViolation)."""
    ec2 = _c("ec2")
    for tsg in d.trainer_sgs(d.trainer_private_ip()):
        for p in ec2.describe_security_groups(GroupIds=[tsg])["SecurityGroups"][0]["IpPermissions"]:
            if any(g["GroupId"] == sg for g in p.get("UserIdGroupPairs", [])):
                ec2.revoke_security_group_ingress(GroupId=tsg, IpPermissions=[{
                    k: v for k, v in p.items() if k in ("IpProtocol", "FromPort", "ToPort")} | {
                    "UserIdGroupPairs": [{"GroupId": sg}]}])
                d.err(f"revoked tcp/{p.get('FromPort')}-{p.get('ToPort')} on {tsg} from {sg}")


# ----------------------------------------------------------------------------- nodes + cluster
def ensure_launch_template(e: dict, profile_arn: str, node_sg: str) -> None:
    ec2 = _c("ec2")
    data = {
        "ImageId": e["ami"], "InstanceType": env("ECS_NODE_TYPE", "m5.2xlarge"),
        "IamInstanceProfile": {"Arn": profile_arn}, "SecurityGroupIds": [node_sg],
        "MetadataOptions": {"HttpTokens": "required", "HttpPutResponseHopLimit": 1, "HttpEndpoint": "enabled"},
        "BlockDeviceMappings": [{"DeviceName": "/dev/xvda", "Ebs": {
            "VolumeSize": int(env("ECS_NODE_DISK_GIB", "300")), "VolumeType": "gp3", "DeleteOnTermination": True}}],
        "UserData": base64.b64encode(user_data(sorted(set(e["images"].values()))).encode()).decode(),
        "TagSpecifications": [{"ResourceType": "instance", "Tags": [{"Key": "Name", "Value": f"{PREFIX}-swe-ecs-node"}]}],
    }
    try:
        ec2.create_launch_template(LaunchTemplateName=LT_NAME, LaunchTemplateData=data)
        d.err(f"created launch template {LT_NAME}")
    except ClientError as ex:
        if "AlreadyExists" not in str(ex):
            raise
        v = ec2.create_launch_template_version(LaunchTemplateName=LT_NAME, LaunchTemplateData=data)
        ec2.modify_launch_template(LaunchTemplateName=LT_NAME,
                                   DefaultVersion=str(v["LaunchTemplateVersion"]["VersionNumber"]))
        d.err(f"launch template {LT_NAME}: new default version {v['LaunchTemplateVersion']['VersionNumber']}")


def _asg():
    got = _c("autoscaling").describe_auto_scaling_groups(AutoScalingGroupNames=[ASG_NAME])["AutoScalingGroups"]
    return got[0] if got else None


def ensure_asg(e: dict) -> str:
    asg = _c("autoscaling")
    kw = dict(AutoScalingGroupName=ASG_NAME,
              LaunchTemplate={"LaunchTemplateName": LT_NAME, "Version": "$Default"},
              MinSize=int(env("ECS_NODES_MIN", "0")), MaxSize=int(env("ECS_NODES_MAX", "16")),
              VPCZoneIdentifier=",".join(s["SubnetId"] for s in e["subnets"]))
    if _asg() is None:
        asg.create_auto_scaling_group(**kw, DesiredCapacity=int(env("ECS_NODES", "4")),
                                      NewInstancesProtectedFromScaleIn=True,   # managed termination protection
                                      Tags=[{"Key": "AmazonECSManaged", "Value": "", "PropagateAtLaunch": True}])
        d.err(f"created ASG {ASG_NAME}")
    else:
        asg.update_auto_scaling_group(**kw)
    return _asg()["AutoScalingGroupARN"]


def ensure_cluster(asg_arn: str) -> None:
    ecs = _c("ecs")
    cps = ecs.describe_capacity_providers(capacityProviders=[CP_NAME])["capacityProviders"]
    if not [c for c in cps if c["status"] == "ACTIVE"]:
        ecs.create_capacity_provider(name=CP_NAME, autoScalingGroupProvider={
            "autoScalingGroupArn": asg_arn, "managedTerminationProtection": "ENABLED",
            "managedScaling": {"status": "ENABLED", "targetCapacity": 100, "minimumScalingStepSize": 1,
                               "maximumScalingStepSize": 4, "instanceWarmupPeriod": 180}})
        d.err(f"created capacity provider {CP_NAME}")
    cl = [c for c in ecs.describe_clusters(clusters=[CLUSTER])["clusters"] if c["status"] == "ACTIVE"]
    if not cl:
        ecs.create_cluster(clusterName=CLUSTER, settings=[{"name": "containerInsights", "value": "disabled"}])
        d.err(f"created cluster {CLUSTER}")
    ecs.put_cluster_capacity_providers(cluster=CLUSTER, capacityProviders=[CP_NAME],
                                       defaultCapacityProviderStrategy=[{"capacityProvider": CP_NAME, "weight": 1}])


def ensure_log_group() -> None:
    logs = _c("logs")
    try:
        logs.create_log_group(logGroupName=LOG_GROUP)
        logs.put_retention_policy(logGroupName=LOG_GROUP, retentionInDays=7)
    except logs.exceptions.ResourceAlreadyExistsException:
        pass


def register_taskdefs(images: dict[str, str]) -> dict[str, str]:
    """One task definition per instance; reuses the latest revision when its spec is unchanged."""
    ecs = _c("ecs")
    out = {}
    for task, image in sorted(images.items()):
        spec = taskdef(task, image)
        h = _spec_hash(spec)
        try:
            cur = ecs.describe_task_definition(taskDefinition=spec["family"], include=["TAGS"])
            if {t["key"]: t["value"] for t in cur.get("tags", [])}.get("acs-spec") == h:
                out[task] = cur["taskDefinition"]["taskDefinitionArn"]
                continue
        except ClientError:
            pass
        r = ecs.register_task_definition(**spec, tags=[{"key": "acs-spec", "value": h}])
        out[task] = r["taskDefinition"]["taskDefinitionArn"]
        d.err(f"[taskdef] {task}: {out[task].rsplit('/', 1)[-1]}")
    return out


# ----------------------------------------------------------------------------- smoke
async def _smoke(td: str) -> dict:
    from agentcore_slime_demo.ecs_task_sandbox import EcsTaskSandbox

    async with EcsTaskSandbox(task_definition=td) as sb:
        _, caps, _ = await sb.exec("grep CapEff /proc/self/status", timeout=30)
        probe = ("/opt/acs/py/bin/python3 -c \"import socket,sys\n"
                 "def ok(h,p):\n  s=socket.socket(); s.settimeout(3)\n"
                 "  try: s.connect((h,p)); return 1\n  except Exception: return 0\n"
                 "print(ok('169.254.169.254',80), ok('169.254.170.2',80), ok('1.1.1.1',443))\"")
        _, net, _ = await sb.exec(probe, timeout=30)
        _, py, _ = await sb.exec("cd /testbed && /opt/miniconda3/envs/testbed/bin/python -c "
                                 "'import sys; print(sys.version.split()[0])'", timeout=60)
        imds, creds, internet = (net.split() + ["?"] * 3)[:3]
        return {"open_s": round(sb.open_recv_epoch - sb.open_sent_epoch, 1), **sb.open_phases,
                "cap_eff": caps.split()[-1] if caps.split() else "?", "imds_reachable": imds,
                "task_creds_endpoint_reachable": creds, "internet_reachable": internet,
                "testbed_python": py.strip()}


def _export_network() -> None:
    net = json.loads(NETWORK_JSON.read_text())
    os.environ.update(ECS_SUBNETS=",".join(net["subnets"]), ECS_SECURITY_GROUPS=",".join(net["security_groups"]))


def smoke() -> dict:
    if not (TASKDEF_MAP.exists() and NETWORK_JSON.exists()):
        d.die(f"{TASKDEF_MAP.name} / {NETWORK_JSON.name} missing — run `ECS_CREATE=1 python3 deploy/ecs_deploy.py up`")
    _export_network()
    res = asyncio.run(_smoke(next(iter(json.loads(TASKDEF_MAP.read_text()).values()))))
    d.err(f"[smoke] {res}")
    if res["cap_eff"].strip("0") or res["imds_reachable"] == "1" or res["internet_reachable"] == "1":
        d.die("smoke task is NOT isolated as intended (capabilities / IMDS / internet) — see [smoke] above")
    return res


# ----------------------------------------------------------------------------- commands
def _cluster_active() -> bool:
    try:
        return any(c["status"] == "ACTIVE" for c in _c("ecs").describe_clusters(clusters=[CLUSTER])["clusters"])
    except ClientError:
        return False


def cmd_render() -> dict:
    e = discover()
    render(e)
    print(f"# rendered deploy/ecs/rendered/ (nothing created)\n"
          f"#   cluster {CLUSTER} ({REGION}), capacity provider {CP_NAME}, ASG {ASG_NAME} "
          f"({env('ECS_NODES', '4')} x {env('ECS_NODE_TYPE', 'm5.2xlarge')}, max {env('ECS_NODES_MAX', '16')})\n"
          f"#   VPC {e['vpc_id']}, subnets {[s['SubnetId'] + '/' + s['AvailabilityZone'] for s in e['subnets']]}\n"
          f"#   AMI {e['ami']}; trainer {e['trainer_ip']} (SGs {e['trainer_sgs']}); caller {e['caller']}\n"
          f"#   exists now: {_cluster_active()}   pool images: {len(e['images'])} ({IMAGE_MAP.name})", file=sys.stderr)
    return e


def cmd_up() -> None:
    e = cmd_render()
    if env("ECS_CREATE") != "1":
        d.die("render only. Creating the substrate is billed (EC2 nodes): re-run with ECS_CREATE=1 once "
              "deploy/ecs/rendered/ looks right.")
    if not e["images"]:
        d.die(f"{IMAGE_MAP} missing or empty — push images first (python3 deploy/deploy_swebench.py)")
    _, profile = ensure_instance_role()
    node_sg, task_sg = ensure_sgs(e)
    ensure_log_group()
    ensure_launch_template(e, profile, node_sg)
    ensure_cluster(ensure_asg(e))
    TASKDEF_MAP.write_text(json.dumps(register_taskdefs(e["images"]), indent=2))
    NETWORK_JSON.write_text(json.dumps({"cluster": CLUSTER, "capacity_provider": CP_NAME,
                                        "subnets": [s["SubnetId"] for s in e["subnets"]],
                                        "security_groups": [task_sg]}, indent=2))
    d.err(f"wrote {TASKDEF_MAP.name} ({len(e['images'])} task definitions) + {NETWORK_JSON.name}")
    smoke()


def _our_tasks(ecs) -> list[str]:
    arns, tok = [], None
    while True:
        r = ecs.list_tasks(cluster=CLUSTER, startedBy="acs-swe", **({"nextToken": tok} if tok else {}))
        arns += r["taskArns"]
        tok = r.get("nextToken")
        if not tok:
            return arns


def cmd_status() -> None:
    if not _cluster_active():
        print(f"cluster {CLUSTER}: not found")
        return
    ecs = _c("ecs")
    cl = ecs.describe_clusters(clusters=[CLUSTER])["clusters"][0]
    asg = _asg()
    print(f"cluster {CLUSTER}: {cl['registeredContainerInstancesCount']} nodes, {cl['runningTasksCount']} running "
          f"/ {cl['pendingTasksCount']} pending tasks; ASG desired={asg and asg['DesiredCapacity']}")
    print(f"{len(_our_tasks(ecs))} sandbox tasks (startedBy=acs-swe)")


def cmd_down() -> None:
    if env("ECS_DELETE") != "1":
        d.die(f"this deletes cluster {CLUSTER}, its nodes and task definitions; re-run with ECS_DELETE=1")
    ecs, asg, ec2 = _c("ecs"), _c("autoscaling"), _c("ec2")
    if _cluster_active():
        for t in _our_tasks(ecs):
            ecs.stop_task(cluster=CLUSTER, task=t, reason="ecs_deploy down")
        ecs.put_cluster_capacity_providers(cluster=CLUSTER, capacityProviders=[], defaultCapacityProviderStrategy=[])
    if any(c["status"] == "ACTIVE" for c in ecs.describe_capacity_providers(capacityProviders=[CP_NAME])["capacityProviders"]):
        ecs.delete_capacity_provider(capacityProvider=CP_NAME)
        d.err(f"deleting capacity provider {CP_NAME}")
    if _asg():
        asg.delete_auto_scaling_group(AutoScalingGroupName=ASG_NAME, ForceDelete=True)
        d.err(f"deleting ASG {ASG_NAME} (terminates the nodes)")
        while _asg():
            time.sleep(15)
    if _cluster_active():
        for ci in ecs.list_container_instances(cluster=CLUSTER)["containerInstanceArns"]:
            ecs.deregister_container_instance(cluster=CLUSTER, containerInstance=ci, force=True)
        ecs.delete_cluster(cluster=CLUSTER)
        d.err(f"deleted cluster {CLUSTER}")
    try:
        ec2.delete_launch_template(LaunchTemplateName=LT_NAME)
    except ClientError:
        pass
    for fam in ecs.list_task_definition_families(familyPrefix=FAMILY_PREFIX, status="ACTIVE")["families"]:
        arns = ecs.list_task_definitions(familyPrefix=fam, status="ACTIVE")["taskDefinitionArns"]
        for a in arns:
            ecs.deregister_task_definition(taskDefinition=a)
        for i in range(0, len(arns), 10):
            ecs.delete_task_definitions(taskDefinitions=arns[i:i + 10])
    vpc_id, _, _ = d.default_vpc()
    for name in (TASK_SG_NAME, NODE_SG_NAME):
        sg = _find_sg(ec2, vpc_id, name)
        if not sg:
            continue
        revoke_trainer_rules(sg)
        for _ in range(40):                                  # task / node ENIs take a while to go away
            try:
                ec2.delete_security_group(GroupId=sg)
                d.err(f"deleted SG {name}")
                break
            except ClientError as ex:
                if "DependencyViolation" not in str(ex):
                    raise
                time.sleep(15)
    iam = _c("iam")
    try:
        iam.remove_role_from_instance_profile(InstanceProfileName=ROLE_NAME, RoleName=ROLE_NAME)
        iam.delete_instance_profile(InstanceProfileName=ROLE_NAME)
    except ClientError:
        pass
    try:
        for p in iam.list_attached_role_policies(RoleName=ROLE_NAME)["AttachedPolicies"]:
            iam.detach_role_policy(RoleName=ROLE_NAME, PolicyArn=p["PolicyArn"])
        iam.delete_role(RoleName=ROLE_NAME)
    except ClientError:
        pass
    try:
        _c("logs").delete_log_group(logGroupName=LOG_GROUP)
    except ClientError:
        pass
    for p in (TASKDEF_MAP, NETWORK_JSON):
        p.unlink(missing_ok=True)
    print(f"# {CLUSTER} torn down (ECR images kept: shared with demo_cpu_swebench_eks)")


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "render"
    {"render": cmd_render, "up": cmd_up, "smoke": smoke, "status": cmd_status, "down": cmd_down}[cmd]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
