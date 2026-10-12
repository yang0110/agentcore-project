#!/usr/bin/env python3
"""Provision / drive / tear down this demo's SageMaker HyperPod cluster (HOST, real AWS $$$).

slime runs on the cluster; the sandboxes stay where demo_terminalbench_kernelbench put them (AgentCore
microVM Runtimes, the AgentCore g6 Instance tier, the m5 EC2 warm pool, the plain-EC2 g6 pool):

    controller  ml.m5.xlarge     x1   Slurm controller (no GPU)
    gpu         ml.g6e.48xlarge  x2   8 x L40S each: the lower-IP node TRAINS (Megatron, 8 GPUs), the
                                      other serves INFERENCE (SGLang, 8 GPUs) + the rollout/harness tier
    FSx for Lustre (/fsx)            the repo mirror (/fsx/work = /work in the container), the HF model,
                                      the base + saved checkpoints, the squashed trainer image

Network (default VPC, ONE AZ, the `<prefix>-egress-<az>` NAT subnet the m5 already uses):

    cluster SG  <-> cluster SG          all traffic (Ray, NCCL, Lustre)
    runtime SG  --tcp 18100-18163-->    cluster SG   (microVM / AgentCore-g6 agent loops dial the adapters)
    m5 / g6 SG  --tcp 18100-18163-->    cluster SG   (same, from the EC2 tiers)
    cluster SG  --tcp 18300, 32768-60999--> m5 / g6 SG  (compose agent + client ports)

    python3 deploy/hyperpod/hyperpod.py image         # build (+EFA, Dockerfile.efa) + push the trainer image to ECR
    python3 deploy/hyperpod/hyperpod.py up            # bucket + IAM role + SG + FSx + lifecycle scripts + cluster -> hyperpod.json
    python3 deploy/hyperpod/hyperpod.py wire          # the SG rules above (idempotent; re-run after a sandbox tier is re-created)
    python3 deploy/hyperpod/hyperpod.py status        # cluster + nodes + FSx
    python3 deploy/hyperpod/hyperpod.py exec '<cmd>'  # run a shell command on the controller (as ubuntu), via SSM
    python3 deploy/hyperpod/hyperpod.py import-image  # enroot-import the ECR image once into /fsx/enroot/<tag>.sqsh
    python3 deploy/hyperpod/hyperpod.py nccl-test     # 2-node all_reduce bus bandwidth: EFA vs TCP sockets
    python3 deploy/hyperpod/hyperpod.py pull          # /fsx/work <- s3://<bucket>/work (after stage_s3.py uploaded it)
    python3 deploy/hyperpod/hyperpod.py submit TAG    # sbatch slurm/train.sbatch -> prints the job id
    python3 deploy/hyperpod/hyperpod.py follow JOB TAG  # stream /fsx/work/.../logs/TAG.log until the job ends
    python3 deploy/hyperpod/hyperpod.py cancel JOB
    python3 deploy/hyperpod/hyperpod.py down          # delete the cluster, then FSx (KEEP_FSX=1 keeps it), SG rules + SG

Knobs: HP_CLUSTER=<prefix>-tbhp  HP_AZ=us-west-2b  HP_GPU_TYPE=ml.g6e.48xlarge  HP_GPU_COUNT=2
HP_CONTROLLER_TYPE=ml.m5.xlarge  HP_EBS_GB=500  HP_FSX_GIB=2400  HP_EFA=1  HP_EFA_VERSION=1.48.0
HP_IMAGE_TAG=agentcore_p4-20261006-efa1.48.0 (…-20261006 with HP_EFA=0).
`exec` needs the AWS session-manager-plugin on this box.
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

HERE = Path(__file__).resolve().parent
DEPLOY = HERE.parent
DEMO_DIR = DEPLOY.parent
sys.path.insert(0, str(DEPLOY))
import agentcore_deploy as d  # noqa: E402
import m5_host as m5  # noqa: E402

STATE = HERE / "hyperpod.json"                   # gitignored: cluster ARN/id, FSx, SG, bucket
LIFECYCLE = HERE / "lifecycle"                   # vendored aws-samples base-config (UPSTREAM.md)
NAME = os.environ.get("HP_CLUSTER", f"{d.PREFIX}-tbhp")
AZ = os.environ.get("HP_AZ", f"{d.REGION}b")
GPU_TYPE = os.environ.get("HP_GPU_TYPE", "ml.g6e.48xlarge")
GPU_COUNT = int(os.environ.get("HP_GPU_COUNT", "2"))
CONTROLLER_TYPE = os.environ.get("HP_CONTROLLER_TYPE", "ml.m5.xlarge")
EBS_GB = int(os.environ.get("HP_EBS_GB", "500"))
FSX_GIB = int(os.environ.get("HP_FSX_GIB", "2400"))   # PERSISTENT_2: 1200 GiB steps; a 9B save is ~200 GB
SRC_IMAGE = os.environ.get("HP_SRC_IMAGE", "slimerl/slime:agentcore_p4-20261006")   # never :latest
# HP_EFA=1 (default): the image is SRC_IMAGE + the EFA userspace (Dockerfile.efa: libfabric + aws-ofi-nccl) and
# the job requires an EFA device on every node; HP_EFA=0: SRC_IMAGE as is, NCCL over TCP sockets.
EFA = os.environ.get("HP_EFA", "1") != "0"
EFA_VERSION = os.environ.get("HP_EFA_VERSION", "1.48.0")
IMAGE_TAG = os.environ.get("HP_IMAGE_TAG", "agentcore_p4-20261006" + (f"-efa{EFA_VERSION}" if EFA else ""))
REPO = f"{d.PREFIX}-tbhp-slime"
ROLE = f"{d.PREFIX}-tbhp-cluster"
SG_NAME = f"{d.PREFIX}-tbhp-cluster"
PARTITION = "rl"
SQSH = f"/fsx/enroot/{IMAGE_TAG}.sqsh"            # one squashfs per image tag (EFA and plain can coexist)
C_DEMO = f"/fsx/work/code/demos/{DEMO_DIR.name}"   # this demo dir on FSx (= /work/code/demos/<demo> in the container)
# Sandbox-tier SGs the cluster must reach / be reached from (the ones demo_terminalbench_kernelbench made).
TIER_SGS = {"m5": (DEPLOY / "m5_host.json", m5.NAME), "g6": (DEPLOY / "g6_host.json", f"{d.PREFIX}-tbh-g6ec2")}


def _c(svc: str):
    return boto3.client(svc, region_name=d.REGION)


def load_state() -> dict:
    if not STATE.exists():
        d.die(f"{STATE} not found — create the cluster first: python3 deploy/hyperpod/hyperpod.py up")
    return json.loads(STATE.read_text())


def save_state(**kv) -> dict:
    st = json.loads(STATE.read_text()) if STATE.exists() else {}
    st.update(kv)
    STATE.write_text(json.dumps(st, indent=2) + "\n")
    return st


# ----------------------------------------------------------------------------- image
def image() -> int:
    """Push the committed trainer snapshot (+ the EFA layer with HP_EFA=1) to ECR; the cluster pulls it once
    with enroot. Only the EFA layer is built: the snapshot itself is never rebuilt."""
    if not shutil.which("docker"):
        d.die("run on the HOST (needs docker)")
    uri = d.ensure_ecr(REPO)
    reg = uri.split("/")[0]
    d.sh(f"aws ecr get-login-password --region {d.REGION} | docker login --username AWS --password-stdin {reg}")
    if EFA:   # empty build context: the Dockerfile copies nothing
        ctx = HERE / ".efa_ctx"
        ctx.mkdir(exist_ok=True)
        d.sh(f"docker build -f {HERE / 'Dockerfile.efa'} --build-arg BASE={SRC_IMAGE} "
             f"--build-arg EFA_INSTALLER_VERSION={EFA_VERSION} -t {uri}:{IMAGE_TAG} {ctx}")
    else:
        d.sh(f"docker tag {SRC_IMAGE} {uri}:{IMAGE_TAG}")
    d.sh(f"docker push {uri}:{IMAGE_TAG}")
    save_state(image=f"{uri}:{IMAGE_TAG}", sqsh=SQSH, efa=EFA)
    d.out("IMAGE", f"{uri}:{IMAGE_TAG}")
    return 0


# ----------------------------------------------------------------------------- up
def _bucket() -> str:
    # "sagemaker-" prefix: AmazonSageMakerClusterInstanceRolePolicy only reads lifecycle scripts from such buckets.
    name = f"sagemaker-{d.PREFIX}-tbhp-{d.account()}-{d.REGION}"
    s3 = _c("s3")
    try:
        s3.head_bucket(Bucket=name)
    except ClientError:
        s3.create_bucket(Bucket=name, CreateBucketConfiguration={"LocationConstraint": d.REGION})
        s3.put_public_access_block(Bucket=name, PublicAccessBlockConfiguration={
            "BlockPublicAcls": True, "IgnorePublicAcls": True, "BlockPublicPolicy": True, "RestrictPublicBuckets": True})
        d.err(f"created bucket {name}")
    return name


def _role(bucket: str) -> str:
    """The node role: HyperPod's managed instance policy + what the rollout process calls (it runs here
    now, not on the box): AgentCore invoke/stop, the EC2 tiers' RunInstances/TerminateInstances (+ PassRole
    on their episode profile), ECR pull, and the staging bucket."""
    iam = _c("iam")
    trust = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {
        "Service": "sagemaker.amazonaws.com"}, "Action": "sts:AssumeRole"}]}
    try:
        arn = iam.create_role(RoleName=ROLE, AssumeRolePolicyDocument=json.dumps(trust),
                              Description=f"{DEMO_DIR.name} HyperPod cluster nodes")["Role"]["Arn"]
        d.err(f"created role {ROLE}")
    except iam.exceptions.EntityAlreadyExistsException:
        arn = iam.get_role(RoleName=ROLE)["Role"]["Arn"]
    iam.attach_role_policy(RoleName=ROLE, PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerClusterInstanceRolePolicy")
    acct = d.account()
    policy = {"Version": "2012-10-17", "Statement": [
        {"Effect": "Allow", "Action": [   # HyperPod's VpcConfig: the nodes' ENIs in our subnet
            "ec2:CreateNetworkInterface", "ec2:CreateNetworkInterfacePermission", "ec2:DeleteNetworkInterface",
            "ec2:DeleteNetworkInterfacePermission", "ec2:DescribeNetworkInterfaces", "ec2:DescribeVpcs",
            "ec2:DescribeDhcpOptions", "ec2:DescribeSubnets", "ec2:DescribeSecurityGroups",
            "ec2:DetachNetworkInterface", "ec2:CreateTags"], "Resource": "*"},
        {"Effect": "Allow", "Action": ["bedrock-agentcore:InvokeAgentRuntime", "bedrock-agentcore:StopRuntimeSession"],
         "Resource": f"arn:aws:bedrock-agentcore:{d.REGION}:{acct}:runtime/{d.PREFIX}_*"},
        {"Effect": "Allow", "Action": ["ec2:RunInstances", "ec2:TerminateInstances", "ec2:Describe*"], "Resource": "*"},
        {"Effect": "Allow", "Action": "iam:PassRole", "Resource": f"arn:aws:iam::{acct}:role/{d.PREFIX}-tbh-m5-episode"},
        {"Effect": "Allow", "Action": ["ecr:GetAuthorizationToken", "ecr:BatchGetImage",
                                       "ecr:GetDownloadUrlForLayer", "ecr:BatchCheckLayerAvailability"], "Resource": "*"},
        {"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket", "s3:DeleteObject"],
         "Resource": [f"arn:aws:s3:::{bucket}", f"arn:aws:s3:::{bucket}/*"]},
    ]}
    iam.put_role_policy(RoleName=ROLE, PolicyName=f"{d.PREFIX}-tbhp-demo", PolicyDocument=json.dumps(policy))
    return arn


def _subnet(vpc_id: str) -> str:
    """The NAT-egress subnet of HP_AZ (pip/HF/ECR from the nodes; no public IP)."""
    return d.ensure_egress_subnets(vpc_id, [AZ])[0]


def _cluster_sg(vpc_id: str) -> str:
    ec2 = _c("ec2")
    got = ec2.describe_security_groups(Filters=[{"Name": "vpc-id", "Values": [vpc_id]},
                                                {"Name": "group-name", "Values": [SG_NAME]}])["SecurityGroups"]
    if got:
        return got[0]["GroupId"]
    sg = ec2.create_security_group(GroupName=SG_NAME, VpcId=vpc_id,
                                   Description=f"{DEMO_DIR.name} HyperPod cluster + FSx")["GroupId"]
    # Self-reference: Ray, NCCL and Lustre (988, 1018-1023) between the nodes and FSx. EFA also needs it.
    ec2.authorize_security_group_ingress(GroupId=sg, IpPermissions=[{
        "IpProtocol": "-1", "UserIdGroupPairs": [{"GroupId": sg, "Description": "cluster self"}]}])
    try:   # EFA requires an explicit self-referencing egress rule, besides the default 0.0.0.0/0
        ec2.authorize_security_group_egress(GroupId=sg, IpPermissions=[{
            "IpProtocol": "-1", "UserIdGroupPairs": [{"GroupId": sg, "Description": "cluster self (EFA)"}]}])
    except ClientError as e:
        if "Duplicate" not in str(e):
            raise
    d.err(f"created SG {sg}")
    return sg


def _fsx(subnet: str, sg: str) -> dict:
    fsx = _c("fsx")
    for fs in fsx.describe_file_systems()["FileSystems"]:
        tags = {t["Key"]: t["Value"] for t in fs.get("Tags", [])}
        if tags.get("Name") == NAME and fs["Lifecycle"] not in ("DELETING", "FAILED"):
            break
    else:
        fs = fsx.create_file_system(
            FileSystemType="LUSTRE", FileSystemTypeVersion="2.15", StorageCapacity=FSX_GIB,
            SubnetIds=[subnet], SecurityGroupIds=[sg],
            LustreConfiguration={"DeploymentType": "PERSISTENT_2", "PerUnitStorageThroughput": 250,
                                 "DataCompressionType": "LZ4"},
            Tags=[{"Key": "Name", "Value": NAME}])["FileSystem"]
        d.err(f"creating FSx {fs['FileSystemId']} ({FSX_GIB} GiB, ~10 min)")
    while fs["Lifecycle"] != "AVAILABLE":
        if fs["Lifecycle"] in ("FAILED", "MISCONFIGURED"):
            d.die(f"FSx {fs['FileSystemId']} {fs['Lifecycle']}: {fs.get('FailureDetails')}")
        time.sleep(30)
        fs = fsx.describe_file_systems(FileSystemIds=[fs["FileSystemId"]])["FileSystems"][0]
    return {"id": fs["FileSystemId"], "dns": fs["DNSName"], "mount": fs["LustreConfiguration"]["MountName"]}


def _upload_lifecycle(bucket: str, fs: dict) -> str:
    s3 = _c("s3")
    prefix = f"lifecycle/{NAME}"
    for p in LIFECYCLE.rglob("*"):
        if p.is_file():
            s3.upload_file(str(p), bucket, f"{prefix}/{p.relative_to(LIFECYCLE)}")
    params = {"version": "1.0.0", "workload_manager": "slurm", "controller_group": "controller",
              "worker_groups": [{"instance_group_name": "gpu", "partition_name": PARTITION}],
              "fsx_dns_name": fs["dns"], "fsx_mountname": fs["mount"]}
    s3.put_object(Bucket=bucket, Key=f"{prefix}/provisioning_parameters.json", Body=json.dumps(params, indent=2).encode())
    return f"s3://{bucket}/{prefix}/"


def _cluster():
    try:
        return _c("sagemaker").describe_cluster(ClusterName=NAME)
    except ClientError as e:
        if "ResourceNotFound" in str(e) or "does not exist" in str(e):
            return None
        raise


def up() -> int:
    vpc_id, _subnets, _sg = d.default_vpc()
    bucket = _bucket()
    role = _role(bucket)
    subnet = _subnet(vpc_id)
    sg = _cluster_sg(vpc_id)
    fs = _fsx(subnet, sg)
    save_state(name=NAME, bucket=bucket, role=role, subnet=subnet, az=AZ, sg=sg, fsx=fs, vpc=vpc_id)
    c = _cluster()
    if c is None:
        s3uri = _upload_lifecycle(bucket, fs)
        time.sleep(10)                                   # a fresh role takes a few seconds to be assumable
        lc = {"SourceS3Uri": s3uri, "OnCreate": "on_create.sh"}
        groups = [
            {"InstanceGroupName": "controller", "InstanceType": CONTROLLER_TYPE, "InstanceCount": 1,
             "LifeCycleConfig": lc, "ExecutionRole": role, "ThreadsPerCore": 1,
             "InstanceStorageConfigs": [{"EbsVolumeConfig": {"VolumeSizeInGB": 100}}]},
            {"InstanceGroupName": "gpu", "InstanceType": GPU_TYPE, "InstanceCount": GPU_COUNT,
             "LifeCycleConfig": lc, "ExecutionRole": role, "ThreadsPerCore": 1,
             "InstanceStorageConfigs": [{"EbsVolumeConfig": {"VolumeSizeInGB": EBS_GB}}]},
        ]
        arn = _c("sagemaker").create_cluster(
            ClusterName=NAME, InstanceGroups=groups, NodeRecovery="Automatic",
            VpcConfig={"SecurityGroupIds": [sg], "Subnets": [subnet]},
            Tags=[{"Key": "Name", "Value": NAME}])["ClusterArn"]
        d.err(f"creating cluster {arn} (~15-25 min)")
        c = _cluster()
    while c["ClusterStatus"] in ("Creating", "Updating", "SystemUpdating"):
        time.sleep(60)
        c = _cluster()
        d.err(f"cluster {c['ClusterStatus']}")
    if c["ClusterStatus"] != "InService":
        d.die(f"cluster {c['ClusterStatus']}: {c.get('FailureMessage', '')}")
    save_state(arn=c["ClusterArn"], id=c["ClusterArn"].rsplit("/", 1)[-1])
    d.err("cluster InService — next: hyperpod.py wire, then import-image")
    return status()


# ----------------------------------------------------------------------------- wire
def wire() -> int:
    """SG rules between the cluster and every sandbox tier. trainer_sgs() honours AGENTCORE_TRAINER_SG,
    so the vendored helpers write the dial-back and m5/g6 rules against the CLUSTER SG, not this box."""
    st = load_state()
    os.environ["AGENTCORE_TRAINER_SG"] = st["sg"]
    vpc_id, _s, runtime_sg = d.default_vpc()
    d.open_dialback(runtime_sg)                        # microVM + AgentCore-g6 sessions -> adapters
    for tier, (state_file, name) in TIER_SGS.items():
        if state_file.exists():
            m5._ensure_sg(vpc_id, name)                # cluster -> agent/client ports, tier -> adapters
        else:
            d.err(f"no {state_file.name}: skip the {tier} tier rules")
    return 0


# ----------------------------------------------------------------------------- nodes / exec
def nodes(group: str | None = None) -> list[dict]:
    st = load_state()
    out, tok = [], None
    while True:
        kw = {"ClusterName": st["name"]} | ({"NextToken": tok} if tok else {})
        if group:
            kw["InstanceGroupNameContains"] = group
        r = _c("sagemaker").list_cluster_nodes(**kw)
        out += r["ClusterNodeSummaries"]
        tok = r.get("NextToken")
        if not tok:
            return out


def _target(group: str = "controller") -> str:
    st = load_state()
    live = [n for n in nodes(group) if n["InstanceGroupName"] == group and n["InstanceStatus"]["Status"] == "Running"]
    if not live:
        d.die(f"no Running node in group {group}")
    return f"sagemaker-cluster:{st['id']}_{group}-{live[0]['InstanceId']}"


def run_remote(cmd: str, group: str = "controller", check: bool = True, capture: bool = False):
    """`cmd` as ubuntu on a node, through SSM (HyperPod has no SSH from outside the VPC)."""
    if not shutil.which("session-manager-plugin"):
        d.die("install the AWS session-manager-plugin on this box first (needed for SSM to HyperPod nodes): "
              "https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html")
    wrapped = f"sudo -u ubuntu -i bash -lc {shlex.quote(cmd)}"
    argv = ["aws", "ssm", "start-session", "--region", d.REGION, "--target", _target(group),
            "--document-name", "AWS-StartNonInteractiveCommand",
            "--parameters", json.dumps({"command": [wrapped]})]
    r = subprocess.run(argv, check=False, text=True, capture_output=capture)
    if check and r.returncode:
        d.die(f"remote command failed ({r.returncode}): {cmd[:200]}")
    return r


def exec_cmd(cmd: str) -> int:
    return run_remote(cmd, check=False).returncode


def import_image() -> int:
    """enroot-import the ECR image to FSx once (~68 GB image -> squashfs); both GPU nodes run from it."""
    st = load_state()
    img = st.get("image") or d.die("no image in hyperpod.json — run: hyperpod.py image")
    reg, path = img.split("/", 1)
    sqsh = st["sqsh"]
    cmd = f"""set -e
sudo mkdir -p /fsx/enroot && sudo chown ubuntu:ubuntu /fsx/enroot && mkdir -p ~/.config/enroot
[ -s {sqsh} ] && {{ echo "{sqsh} exists"; exit 0; }}
printf 'machine {reg} login AWS password %s\\n' "$(aws ecr get-login-password --region {d.REGION})" > ~/.config/enroot/.credentials
chmod 600 ~/.config/enroot/.credentials
srun -p {PARTITION} -N1 --exclusive bash -lc 'enroot import -o {sqsh}.tmp "docker://{reg}#{path}" && mv {sqsh}.tmp {sqsh}'
ls -la {sqsh}"""
    return run_remote(cmd, check=False).returncode


def pull() -> int:
    """/fsx/work <- s3://<bucket>/work (stage_s3.py put the demo + model + base ckpt there)."""
    st = load_state()
    return run_remote(f"sudo mkdir -p /fsx/work && sudo chown ubuntu:ubuntu /fsx/work && aws s3 sync --only-show-errors --no-progress "
                      f"s3://{st['bucket']}/work /fsx/work && du -sh /fsx/work", check=False).returncode


def submit(tag: str) -> int:
    """sbatch slurm/train.sbatch on both GPU nodes; run knobs come through the env (PASS_RE in steps_common.sh)."""
    st = load_state()
    env = os.environ.get("HP_RUN_ENV", "")             # "K=V K=V ..." from 5_run.sh
    cmd = (f"cd {C_DEMO} && mkdir -p logs && env {env} RUN_TAG={shlex.quote(tag)} HP_DEMO={DEMO_DIR.name} HP_BUCKET={st['bucket']} "
           f"HP_SQSH={st['sqsh']} HP_EFA={int(st.get('efa', True))} "
           f"sbatch --parsable -p {PARTITION} -N {GPU_COUNT} --exclusive -o logs/{tag}.slurm.out "
           f"deploy/hyperpod/slurm/train.sbatch")
    r = run_remote(cmd, capture=True)
    job = (r.stdout or "").strip().splitlines()[-1].split(";")[0] if r.stdout else ""
    if not job.isdigit():
        d.die(f"sbatch gave no job id: {r.stdout!r} {r.stderr!r}")
    d.out("JOB", job)
    return 0


def follow(job: str, tag: str, every: int = 60) -> int:
    """Print the run log as it grows (one SSM call per poll) until the Slurm job leaves the queue."""
    seen = 0
    while True:
        r = run_remote(f"f={C_DEMO}/logs/{tag}.log; [ -f $f ] && tail -c +{seen + 1} $f; "
                       f"echo; echo \"@@SIZE $(stat -c %s $f 2>/dev/null || echo {seen})\"; "
                       f"echo \"@@STATE $(squeue -h -j {job} -o %T 2>/dev/null)\"", check=False, capture=True)
        body = r.stdout or ""
        size = next((int(x.split()[1]) for x in body.splitlines() if x.startswith("@@SIZE ")), seen)
        state = next((x[8:].strip() for x in body.splitlines() if x.startswith("@@STATE ")), "")
        text = "\n".join(x for x in body.splitlines() if not x.startswith("@@")).rstrip("\n")
        if size > seen and text:
            print(text, flush=True)
        seen = max(seen, size)
        if not state:
            r = run_remote(f"sacct -n -X -j {job} -o State,ExitCode | head -1", check=False, capture=True)
            res = (r.stdout or "").split()
            print(f"# job {job} finished: {' '.join(res)}", flush=True)
            return 0 if res[:1] == ["COMPLETED"] else 1
        time.sleep(every)


def nccl_test() -> int:
    """2-node all_reduce bus bandwidth, EFA then TCP sockets (slurm/nccl_bw.sbatch), printed here. The weight
    push of a 9B step is ~18 GB bf16, so this bounds the cross-node sync time."""
    st = load_state()
    r = run_remote(f"cd {C_DEMO} && sbatch --parsable --wait -p {PARTITION} -N 2 --exclusive -o logs/nccl_bw.out "
                   f"--export=ALL,HP_SQSH={st['sqsh']},HP_DEMO={DEMO_DIR.name} deploy/hyperpod/slurm/nccl_bw.sbatch; "
                   f"cat logs/nccl_bw.out", check=False)
    return r.returncode


def cancel(job: str) -> int:
    return run_remote(f"scancel {job}", check=False).returncode


# ----------------------------------------------------------------------------- status / down
def status() -> int:
    st = load_state()
    c = _cluster()
    print(f"# cluster {st['name']}: {c['ClusterStatus'] if c else 'absent'}")
    if c:
        for n in nodes():
            print(f"#   {n['InstanceGroupName']:<10} {n['InstanceType']:<16} {n['InstanceId']}  {n['InstanceStatus']['Status']}")
    fs = st.get("fsx") or {}
    if fs:
        try:
            life = _c("fsx").describe_file_systems(FileSystemIds=[fs["id"]])["FileSystems"][0]["Lifecycle"]
        except ClientError:
            life = "absent"
        print(f"# fsx {fs['id']} {life}  {fs['dns']}@tcp:/{fs['mount']} -> /fsx")
    print(f"# sg {st.get('sg')}  subnet {st.get('subnet')} ({st.get('az')})  bucket {st.get('bucket')}  image {st.get('image', '-')}")
    return 0


def down() -> int:
    st = load_state()
    sm = _c("sagemaker")
    if _cluster():
        sm.delete_cluster(ClusterName=st["name"])
        d.err(f"deleting cluster {st['name']}")
        while _cluster():
            time.sleep(30)
        d.err("cluster deleted")
    fs = st.get("fsx")
    if fs and os.environ.get("KEEP_FSX", "0") != "1":
        try:
            _c("fsx").delete_file_system(FileSystemId=fs["id"])
            d.err(f"deleting FSx {fs['id']} (checkpoints on it are gone; KEEP_FSX=1 keeps it)")
            while True:
                time.sleep(30)
                _c("fsx").describe_file_systems(FileSystemIds=[fs["id"]])
        except ClientError as e:
            if "FileSystemNotFound" not in str(e):
                raise
        save_state(fsx={})
    if not (fs and os.environ.get("KEEP_FSX", "0") == "1"):
        _drop_sg(st["sg"])
    d.err(f"kept: bucket {st.get('bucket')}, role {ROLE}, ECR {REPO}")
    return 0


def _drop_sg(sg: str) -> None:
    """Revoke every rule in the tier SGs (and the default runtime SG) that names the cluster SG, then delete it."""
    ec2 = _c("ec2")
    vpc_id = load_state()["vpc"]
    others = ec2.describe_security_groups(Filters=[{"Name": "vpc-id", "Values": [vpc_id]}])["SecurityGroups"]
    for g in others:
        if g["GroupId"] == sg:
            continue
        perms = [p for p in g["IpPermissions"] if any(u.get("GroupId") == sg for u in p.get("UserIdGroupPairs", []))]
        for p in perms:
            p["UserIdGroupPairs"] = [u for u in p["UserIdGroupPairs"] if u.get("GroupId") == sg]
            for k in ("IpRanges", "Ipv6Ranges", "PrefixListIds"):
                p.pop(k, None)
        if perms:
            ec2.revoke_security_group_ingress(GroupId=g["GroupId"], IpPermissions=perms)
            d.err(f"revoked {len(perms)} rule(s) on {g['GroupId']} from {sg}")
    for attempt in range(30):                          # the nodes' ENIs go a little after the cluster
        try:
            ec2.delete_security_group(GroupId=sg)
            d.err(f"deleted SG {sg}")
            return
        except ClientError as e:
            if e.response["Error"]["Code"] == "InvalidGroup.NotFound":
                return
            if e.response["Error"]["Code"] != "DependencyViolation" or attempt == 29:
                raise
            time.sleep(20)


if __name__ == "__main__":
    cmd, args = (sys.argv[1] if len(sys.argv) > 1 else "status"), sys.argv[2:]
    fns = {"image": image, "up": up, "wire": wire, "status": status, "import-image": import_image, "pull": pull,
           "down": down, "exec": lambda: exec_cmd(" ".join(args)), "submit": lambda: submit(args[0]),
           "follow": lambda: follow(args[0], args[1]), "cancel": lambda: cancel(args[0]), "nccl-test": nccl_test}
    if cmd not in fns:
        d.die(f"usage: {sys.argv[0]} {'|'.join(fns)}")
    sys.exit(fns[cmd]())
