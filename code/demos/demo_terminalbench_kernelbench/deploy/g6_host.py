#!/usr/bin/env python3
"""The OPT-IN plain-EC2 g6 tier for KernelBench (TB_KB_SUBSTRATE=ec2; HOST, real AWS $).

The default KernelBench substrate is the AgentCore g6 Runtime (terminalbench/deploy_gpu_tasks.py): one
session = one managed instance, booted per episode. This tier is the m5 warm pool on GPUs instead
(agentcore_slime_demo/ec2_gpu_sandbox.py): warm g6 instances in our own account, one episode per GPU
slot, a fresh container per episode, a verified reset after it. Everything here reuses the m5 tier:
the compose agent (m5_agent/compose_agent.py), its bootstrap, the ECR-pull instance role, the egress
subnets and the SG rules (deploy/m5_host.py).

    python3 deploy/g6_host.py image      # build + push the KernelBench image (torch baked in, no 2 GB cap)
                                         # -> ECR <prefix>-tbhdk-kernelbench-gpu-client + its task record
    python3 deploy/g6_host.py bake-ami   # SG + egress subnets (TB_G6_AZS) + a g6 AMI: Deep Learning Base OSS
                                         # Nvidia AMI (AL2023) + compose agent + the image pre-pulled -> g6_host.json
    python3 deploy/g6_host.py probe      # open/close episodes through the warm pool: cold vs warm open, GPU check
    python3 deploy/g6_host.py status     # live tagged g6 instances
    python3 deploy/g6_host.py gc         # terminate every tagged g6 instance
    python3 deploy/g6_host.py down       # gc + drop the SG + g6_host.json (AMI and ECR image kept)

Then: TB_KB_SUBSTRATE=ec2 bash 5_run.sh. Knobs (bake time, recorded in g6_host.json):
TB_G6_INSTANCE_TYPES=g6.xlarge,g5.xlarge,g6e.xlarge,g6.2xlarge,g5.2xlarge,g6.4xlarge (the launch order: cheapest
1-GPU first, then bigger 1-GPU sizes for capacity; a g6.12xlarge = 4 GPU slots)
TB_G6_AZS=<every AZ with a default subnet>  TB_G6_DISK_GB=100  G6_BAKE_INSTANCE=m5.xlarge (no GPU needed
to bake). Run time: TB_G6_MAX_INSTANCES=16  TB_G6_IDLE_RETIRE_S=900  TB_G6_CAPACITY_RESERVATION_ID
(+ TB_G6_CAPACITY_BLOCK=1). Probe: PROBE_ROUNDS=3 PROBE_CONC=1.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import secrets
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

import boto3

HERE = Path(__file__).resolve().parent
DEMO_DIR = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(DEMO_DIR / "terminalbench"))
sys.path.insert(0, str(DEMO_DIR))
import agentcore_deploy as d  # noqa: E402
import m5_host as m5  # noqa: E402

STATE = HERE / "g6_host.json"                       # gitignored
NAME = f"{d.PREFIX}-tbh-g6ec2"                      # SG name + instance tag
TASK = "kernelbench-gpu"                            # = the KernelBench rows' image (route gpuec2:kernelbench-gpu)
REPO = f"{m5.ECR_PREFIX}-{TASK}-client"             # matches the episode role's <prefix>-tbhdk-* pull grant
DOCKERFILE = HERE / "Dockerfile.kernelbench_ec2"
RECORDS = DEMO_DIR / "terminalbench" / "tasks" / "docker_records"
DLAMI_PARAM = "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-amazon-linux-2023/latest/ami-id"
TYPES = [t.strip() for t in os.environ.get("TB_G6_INSTANCE_TYPES", "g6.xlarge,g5.xlarge,g6e.xlarge,g6.2xlarge,g5.2xlarge,g6.4xlarge").split(",") if t.strip()]
DISK_GB = int(os.environ.get("TB_G6_DISK_GB", "100"))
COMPOSE = (f"services:\n  client:\n    image: acs-tbh/{TASK}-client:latest\n"
           "    container_name: ${T_BENCH_TASK_DOCKER_CLIENT_CONTAINER_NAME}\n"
           "    working_dir: /workspace\n    command: [\"sh\", \"-c\", \"sleep infinity\"]\n")


def _ec2():
    return boto3.client("ec2", region_name=d.REGION)


def _server_files() -> dict[str, bytes]:
    import deploy_docker_tasks as dock  # noqa: E402  (acs_http_server.py + sandbox_server_agent.py)

    return {k: v for k, v in dock._server_files().items() if k != "WRAP_APPENDIX"}


def image() -> int:
    """Build + push the image (linux/amd64), then write its task dir + ECR record where the episode
    sandbox uploads them from (terminalbench/tasks/docker_records/kernelbench-gpu.{json,tar.gz})."""
    files = _server_files()
    h = hashlib.sha256(DOCKERFILE.read_bytes())
    for k, v in sorted(files.items()):
        h.update(k.encode() + v)
    tag = "kb-" + h.hexdigest()[:12]
    uri = d.ensure_ecr(REPO)
    d._ensure_buildx()
    d.sh(f"aws ecr get-login-password --region {d.REGION} | docker login --username AWS --password-stdin {uri.split('/')[0]}")
    with tempfile.TemporaryDirectory(prefix="acs-g6-ctx-") as tmp:
        for k, v in files.items():
            (Path(tmp) / k).write_bytes(v)
        subprocess.run(f"docker buildx build --builder {d.BUILDX_BUILDER} --platform linux/amd64 --provenance=false "
                       f"--sbom=false -f {DOCKERFILE} -t {uri}:{tag} --push {tmp}", shell=True, check=True)
    det = boto3.client("ecr", region_name=d.REGION).describe_images(
        repositoryName=REPO, imageIds=[{"imageTag": tag}])["imageDetails"][0]
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        ti = tarfile.TarInfo("docker-compose.yaml")
        ti.size, ti.mode = len(COMPOSE), 0o644
        tf.addfile(ti, io.BytesIO(COMPOSE.encode()))
    RECORDS.mkdir(parents=True, exist_ok=True)
    (RECORDS / f"{TASK}.tar.gz").write_bytes(buf.getvalue())
    rec = {"task": TASK, "tag": tag, "status": "built", "images": {}, "client": "client",
           "size_mb": round(det["imageSizeInBytes"] / 2**20, 1), "ecr": {"client": f"{uri}:{tag}"}}
    (RECORDS / f"{TASK}.json").write_text(json.dumps(rec, indent=1))
    if STATE.exists():                       # a newer image than the AMI's: episodes pull it once per instance
        st = json.loads(STATE.read_text())
        if st.get("image") != rec["ecr"]["client"]:
            print(f"# note: {STATE.name} AMI pre-pulled {st.get('image')}; re-run bake-ami to pre-pull the new one")
    print(f"# {uri}:{tag} ({rec['size_mb']:.0f} MB compressed) -> {RECORDS / (TASK + '.json')}")
    return 0


def _azs(vpc_id: str) -> list[str]:
    want = [a.strip() for a in os.environ.get("TB_G6_AZS", "").split(",") if a.strip()]
    if want:
        return want
    return sorted({sn["AvailabilityZone"] for sn in _ec2().describe_subnets(Filters=[
        {"Name": "vpc-id", "Values": [vpc_id]}, {"Name": "default-for-az", "Values": ["true"]}])["Subnets"]})


def bake_ami() -> int:
    rec_path = RECORDS / f"{TASK}.json"
    if not rec_path.exists():
        d.die("run `python3 deploy/g6_host.py image` first")
    rec = json.loads(rec_path.read_text())
    profile = m5._ensure_profile(m5.ROLE_EPISODE, m5.PULL)
    vpc_id = d.default_vpc()[0]
    azs = _azs(vpc_id)
    subnets = [{"id": s, "az": a} for s, a in zip(d.ensure_egress_subnets(vpc_id, azs), azs)]
    sg = m5._ensure_sg(vpc_id, NAME, "demo_terminalbench_kernelbench plain-EC2 g6 (KernelBench)")
    ec2 = _ec2()
    base = boto3.client("ssm", region_name=d.REGION).get_parameter(Name=DLAMI_PARAM)["Parameter"]["Value"]
    root = ec2.describe_images(ImageIds=[base])["Images"][0]["RootDeviceName"]
    token = secrets.token_urlsafe(32)
    bake_type = os.environ.get("G6_BAKE_INSTANCE", "m5.xlarge")
    inst = ec2.run_instances(
        ImageId=base, InstanceType=bake_type, MinCount=1, MaxCount=1, UserData=m5._user_data(token),
        IamInstanceProfile={"Name": profile},
        NetworkInterfaces=[{"DeviceIndex": 0, "SubnetId": subnets[0]["id"], "Groups": [sg],
                            "AssociatePublicIpAddress": False}],
        BlockDeviceMappings=[{"DeviceName": root, "Ebs": {"VolumeSize": DISK_GB, "VolumeType": "gp3",
                                                          "DeleteOnTermination": True}}],
        MetadataOptions={"HttpTokens": "required", "HttpPutResponseHopLimit": 1, "HttpEndpoint": "enabled"},
        TagSpecifications=[{"ResourceType": "instance", "Tags": [{"Key": "Name", "Value": f"{NAME}-bake"}]}])["Instances"][0]
    iid = inst["InstanceId"]
    d.err(f"bake: launched {iid} ({bake_type}, DLAMI {base}); waiting for docker + agent")
    try:
        ec2.get_waiter("instance_running").wait(InstanceIds=[iid])
        ip = ec2.describe_instances(InstanceIds=[iid])["Reservations"][0]["Instances"][0]["PrivateIpAddress"]
        st = {"ip": ip, "port": m5.AGENT_PORT, "token": token}
        h = m5._wait_agent(st)
        d.err(f"bake: agent up (docker {h.get('docker')}, gpus {h.get('gpus')}); pre-pulling {rec['ecr']['client']}")
        # One real episode without a GPU: pulls the image into the AMI and checks the server starts in it.
        m5.put_tar(st, (RECORDS / f"{TASK}.tar.gz").read_bytes(), f"task:{TASK}")
        m5.call(st, {"op": "set_record", "task": TASK, "rec": rec})
        up = m5.call(st, {"op": "up", "task": TASK, "episode": "bake"}, timeout=1800)
        with urllib.request.urlopen(f"http://{ip}:{up['port']}/ping", timeout=30) as r:
            assert r.status == 200
        m5.call(st, {"op": "down", "episode": "bake"}, timeout=330)
        chk = m5.call(st, {"op": "reset_check"}, timeout=600)
        if not chk.get("clean"):
            d.die(f"bake: reset_check not clean: {chk.get('problems')}")
        d.err(f"bake: episode ok (pull {up.get('pull_ms')} ms, up {up.get('ready_ms')} ms); imaging")
        img = ec2.create_image(InstanceId=iid, Name=f"{NAME}-{time.strftime('%Y%m%d-%H%M%S')}",
                               Description="demo_terminalbench_kernelbench plain-EC2 g6: DLAMI + compose agent + KernelBench image",
                               TagSpecifications=[{"ResourceType": "image", "Tags": [{"Key": "Name", "Value": NAME}]}])
        ec2.get_waiter("image_available").wait(ImageIds=[img["ImageId"]], WaiterConfig={"Delay": 15, "MaxAttempts": 160})
    finally:
        ec2.terminate_instances(InstanceIds=[iid])
    state = {"ami": img["ImageId"], "root_device": root, "subnets": subnets, "sg": sg, "profile": profile,
             "instance_types": TYPES, "disk_gb": DISK_GB, "agent_port": m5.AGENT_PORT, "tag": NAME,
             "image": rec["ecr"]["client"], "base_ami": base}
    STATE.write_text(json.dumps(state, indent=2))
    print(f"# g6 AMI {img['ImageId']} ({', '.join(TYPES)} in {', '.join(azs)}) -> {STATE}")
    return 0


def _live() -> list[dict]:
    return [i for r in _ec2().describe_instances(Filters=[
        {"Name": "tag:Name", "Values": [NAME]},
        {"Name": "instance-state-name", "Values": ["pending", "running", "stopping", "stopped"]}])["Reservations"]
        for i in r["Instances"]]


def status() -> int:
    for i in _live():
        age = (time.time() - i["LaunchTime"].timestamp()) / 60
        print(f"{i['InstanceId']}  {i['InstanceType']:<12} {i['Placement']['AvailabilityZone']}  "
              f"{i['State']['Name']:<8} {i.get('PrivateIpAddress', '-'):<15} up {age:.0f} min")
    print(f"# {len(_live())} g6 instance(s) tagged {NAME}")
    return 0


def gc() -> int:
    ids = [i["InstanceId"] for i in _live()]
    if ids:
        _ec2().terminate_instances(InstanceIds=ids)
        try:
            _ec2().get_waiter("instance_terminated").wait(InstanceIds=ids, WaiterConfig={"Delay": 10, "MaxAttempts": 30})
        except Exception as e:  # noqa: BLE001
            print(f"# not all terminated after 300 s: {e}")
    print(f"# terminated {len(ids)} g6 instance(s)")
    return 0


def down() -> int:
    gc()
    m5._delete_sg(NAME)
    STATE.unlink(missing_ok=True)
    print(f"# g6 tier removed (kept: AMIs tagged {NAME}, ECR {REPO}, the egress subnets)")
    return 0


def probe() -> int:
    """PROBE_ROUNDS rounds of PROBE_CONC concurrent episodes through the real warm pool: open (cold
    launch the first time, warm after), check the GPU from inside the episode container, close."""
    if not STATE.exists():
        d.die("run bake-ami first")
    os.environ["TB_G6_EC2"] = STATE.read_text()
    os.environ.setdefault("TB_G6_MAX_INSTANCES", "1")
    from agentcore_slime_demo.ec2_gpu_sandbox import Ec2GpuSandbox, _POOL, _terminate_ids

    check = ("nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader; python3 -c \"import torch;"
             "print('cuda', torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0))\"")

    async def one(rnd: int, k: int) -> dict:
        sb = Ec2GpuSandbox(f"gpuec2:{TASK}", region=d.REGION, session_id=f"probe-{secrets.token_hex(16)}")
        await sb.__aenter__()
        try:
            boot = sb.open_reply.get("boot") or {}
            code, out, err = await sb.exec(check, timeout=300)
            return {"round": rnd, "k": k, "open_s": round(sb.open_recv_epoch - sb.open_sent_epoch, 1),
                    "warm": boot.get("warm_reused"), "instance": boot.get("instance_id"),
                    "type": boot.get("instance_type"), "slot": boot.get("gpu_slot"),
                    "check": (out or err).strip().replace("\n", " | ")[:160], "exit": code}
        finally:
            await sb.__aexit__(None, None, None)

    async def main() -> None:
        for rnd in range(int(os.environ.get("PROBE_ROUNDS", "3"))):
            for r in await asyncio.gather(*[one(rnd, k) for k in range(int(os.environ.get("PROBE_CONC", "1")))]):
                print(json.dumps(r), flush=True)

    try:
        asyncio.run(main())
    finally:
        ids = _POOL.all_ids()
        if ids and os.environ.get("PROBE_KEEP") != "1":
            _terminate_ids(d.REGION, ids)
            print(f"# terminated {ids}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    fn = {"image": image, "bake-ami": bake_ami, "status": status, "gc": gc, "down": down, "probe": probe}.get(cmd)
    if fn is None:
        d.die(f"usage: {sys.argv[0]} image|bake-ami|probe|status|gc|down")
    sys.exit(fn())
