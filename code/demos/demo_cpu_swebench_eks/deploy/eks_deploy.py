#!/usr/bin/env python3
"""EKS + gVisor substrate for demo_cpu_swebench_eks: render, create, smoke-test and delete the cluster.

    python3 deploy/eks_deploy.py render     # READ-ONLY: discover VPC/subnets/trainer IP -> deploy/eks/cluster.yaml
                                            #   + deploy/eks/rendered/*.yaml, print the plan. Creates nothing.
    EKS_CREATE=1 python3 deploy/eks_deploy.py up
                                            # eksctl create cluster (~15-20 min, billed: control plane + nodes),
                                            #   SG rules, gVisor installer, RuntimeClass, namespace+NetworkPolicy,
                                            #   image pre-pull, one gVisor smoke pod; prints `export EKS_...`
    python3 deploy/eks_deploy.py smoke      # one pod from the first image: open timings + proof it runs on gVisor
    python3 deploy/eks_deploy.py status
    EKS_DELETE=1 python3 deploy/eks_deploy.py down   # revoke our SG rules, eksctl delete cluster

`up` without EKS_CREATE=1 only renders and stops, so a stray `bash 4_deploy_runtime.sh` cannot start a
cluster. Needs `eksctl` on the HOST for up/down (not in the trainer container; the trainer talks to the
API server itself, agentcore_slime_demo/eks_api.py).

Knobs: EKS_CLUSTER (acs-swe-eks)  EKS_NAMESPACE (acs-swe)  EKS_VERSION (1.35)  EKS_NODE_TYPE (m5.2xlarge)
       EKS_NODES (4)  EKS_NODES_MIN (1)  EKS_NODES_MAX (16)  EKS_NODE_DISK_GIB (300)
       EKS_PUBLIC_CIDRS (default: this box's public egress IP/32)  EKS_PREPULL (1)  GVISOR_RELEASE (latest)
Network: same VPC as the trainer; nodes in the ${NAME_PREFIX}-egress-* NAT subnets (they must already
exist — the AgentCore demos' RUNTIME_EGRESS=nat created them; `render` does not create subnets).
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from string import Template

import boto3
import yaml

HERE = Path(__file__).resolve().parent
DEMO_DIR = HERE.parent
EKS_DIR = HERE / "eks"
RENDERED = EKS_DIR / "rendered"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(DEMO_DIR))

import agentcore_deploy as d  # noqa: E402

REGION, PREFIX = d.REGION, d.PREFIX
CLUSTER = os.environ.setdefault("EKS_CLUSTER", f"{PREFIX}-swe-eks")
NAMESPACE = os.environ.setdefault("EKS_NAMESPACE", f"{PREFIX}-swe")
POOL_MAP = DEMO_DIR / "swebench" / "image_by_task.json"
PAUSE_IMAGE = "registry.k8s.io/pause:3.10"


# ----------------------------------------------------------------------------- discovery (read-only)
def discover() -> dict:
    ec2 = boto3.client("ec2", region_name=REGION)
    vpc_id, _, _ = d.default_vpc()
    subnets = [sn for sn in ec2.describe_subnets(Filters=[
        {"Name": "vpc-id", "Values": [vpc_id]}, {"Name": "tag:Name", "Values": [f"{d.EGRESS_TAG}-*"]}])["Subnets"]]
    if len({sn["AvailabilityZone"] for sn in subnets}) < 2:
        d.die(f"need {d.EGRESS_TAG}-* NAT subnets in >= 2 AZs of {vpc_id} (EKS requires 2); found "
              f"{[sn['SubnetId'] for sn in subnets]}. Create them once with any AgentCore demo's "
              "RUNTIME_EGRESS=nat deploy (agentcore_deploy.ensure_egress_subnets).")
    ip = d.trainer_private_ip()
    public = os.environ.get("EKS_PUBLIC_CIDRS", "")
    if not public:
        try:
            public = urllib.request.urlopen("https://checkip.amazonaws.com", timeout=10).read().decode().strip() + "/32"
        except Exception as e:  # noqa: BLE001
            d.die(f"could not find this box's public IP ({e}); set EKS_PUBLIC_CIDRS")
    return {"vpc_id": vpc_id, "subnets": sorted(subnets, key=lambda s: s["AvailabilityZone"]),
            "trainer_ip": ip, "trainer_sgs": d.trainer_sgs(ip), "public_cidrs": public,
            "caller": boto3.client("sts", region_name=REGION).get_caller_identity()["Arn"]}


def render(env: dict) -> None:
    RENDERED.mkdir(parents=True, exist_ok=True)
    nodes = os.environ.get("EKS_NODES", "4")
    cluster_yaml = Template((EKS_DIR / "cluster.yaml.tmpl").read_text()).substitute(
        CLUSTER=CLUSTER, REGION=REGION, PREFIX=PREFIX, K8S_VERSION=os.environ.get("EKS_VERSION", "1.35"),
        VPC_ID=env["vpc_id"], NODE_TYPE=os.environ.get("EKS_NODE_TYPE", "m5.2xlarge"), NODES=nodes,
        NODES_MIN=os.environ.get("EKS_NODES_MIN", "1"), NODES_MAX=os.environ.get("EKS_NODES_MAX", "16"),
        NODE_DISK_GIB=os.environ.get("EKS_NODE_DISK_GIB", "300"),
        PRIVATE_SUBNETS="\n".join(f"      {sn['AvailabilityZone']}: {{id: {sn['SubnetId']}}}" for sn in env["subnets"]),
        SUBNET_IDS=", ".join(sn["SubnetId"] for sn in env["subnets"]),
        PUBLIC_CIDRS=", ".join(c.strip() for c in env["public_cidrs"].split(",") if c.strip()))
    (EKS_DIR / "cluster.yaml").write_text(cluster_yaml)

    ns_yaml = Template((EKS_DIR / "sandbox-namespace.yaml.tmpl").read_text()).substitute(
        NAMESPACE=NAMESPACE, TRAINER_IP=env["trainer_ip"],
        NODE_CIDR_BLOCKS="".join(f", {{ipBlock: {{cidr: {sn['CidrBlock']}}}}}" for sn in env["subnets"]),
        ADAPTER_PORT_BASE=d.ADAPTER_PORT_BASE, ADAPTER_PORT_END=d.ADAPTER_PORT_BASE + d.ADAPTER_PORT_SPAN - 1,
        ADAPTER_PORT=d.ADAPTER_PORT)
    (RENDERED / "sandbox-namespace.yaml").write_text(ns_yaml)

    inst = (EKS_DIR / "gvisor-installer.yaml").read_text().replace(
        'value: "latest"', f'value: "{os.environ.get("GVISOR_RELEASE", "latest")}"')
    (RENDERED / "gvisor-installer.yaml").write_text(inst)
    shutil.copy(EKS_DIR / "runtimeclass.yaml", RENDERED / "runtimeclass.yaml")
    prepull = prepull_manifest()
    if prepull:
        (RENDERED / "prepull.yaml").write_text(yaml.safe_dump(prepull, sort_keys=False))
    else:
        (RENDERED / "prepull.yaml").unlink(missing_ok=True)


def prepull_manifest() -> dict | None:
    """A DaemonSet whose init containers are the pool's images (each runs /bin/true under gVisor), so every
    node pulls + unpacks all of them once, up front; the first episode on a node then skips the pull.
    Disk: ~3-6 GB per image per node — keep the pool within EKS_NODE_DISK_GIB (kubelet GCs images > 85%)."""
    if os.environ.get("EKS_PREPULL", "1") != "1" or not POOL_MAP.exists():
        return None
    images = sorted(set(json.loads(POOL_MAP.read_text()).values()))
    if not images:
        return None
    small = {"requests": {"cpu": "10m", "memory": "16Mi"}}
    return {
        "apiVersion": "apps/v1", "kind": "DaemonSet",
        "metadata": {"name": "acs-swe-prepull", "namespace": NAMESPACE},
        "spec": {"selector": {"matchLabels": {"app": "acs-swe-prepull"}}, "template": {
            "metadata": {"labels": {"app": "acs-swe-prepull"}},
            "spec": {"runtimeClassName": "gvisor", "automountServiceAccountToken": False,
                     "initContainers": [{"name": f"img{i}", "image": img, "imagePullPolicy": "IfNotPresent",
                                         "command": ["/bin/true"], "resources": small}
                                        for i, img in enumerate(images)],
                     "containers": [{"name": "pause", "image": PAUSE_IMAGE, "resources": small}]}}}}


# ----------------------------------------------------------------------------- apply (no kubectl)
_PATHS = {  # kind -> (api prefix, plural, namespaced)
    "Namespace": ("/api/v1", "namespaces", False), "ServiceAccount": ("/api/v1", "serviceaccounts", True),
    "ConfigMap": ("/api/v1", "configmaps", True), "Pod": ("/api/v1", "pods", True),
    "DaemonSet": ("/apis/apps/v1", "daemonsets", True),
    "NetworkPolicy": ("/apis/networking.k8s.io/v1", "networkpolicies", True),
    "RuntimeClass": ("/apis/node.k8s.io/v1", "runtimeclasses", False),
    "ClusterRole": ("/apis/rbac.authorization.k8s.io/v1", "clusterroles", False),
    "ClusterRoleBinding": ("/apis/rbac.authorization.k8s.io/v1", "clusterrolebindings", False),
}


def _coll(obj: dict) -> str:
    prefix, plural, namespaced = _PATHS[obj["kind"]]
    ns = obj["metadata"].get("namespace", "default")
    return f"{prefix}/namespaces/{ns}/{plural}" if namespaced else f"{prefix}/{plural}"


def apply_file(api, path: Path) -> None:
    from agentcore_slime_demo.eks_api import KubeError

    for obj in (o for o in yaml.safe_load_all(path.read_text()) if o):
        coll, name = _coll(obj), obj["metadata"]["name"]
        try:
            api.request("POST", coll, obj)
            d.err(f"created {obj['kind']}/{name}")
        except KubeError as e:
            if e.status != 409:
                raise
            cur = api.request("GET", f"{coll}/{name}")
            obj["metadata"]["resourceVersion"] = cur["metadata"]["resourceVersion"]
            api.request("PUT", f"{coll}/{name}", obj)
            d.err(f"updated {obj['kind']}/{name}")


def wait_daemonset(api, ns: str, name: str, timeout: float) -> None:
    t0 = time.time()
    while True:
        st = api.request("GET", f"/apis/apps/v1/namespaces/{ns}/daemonsets/{name}").get("status") or {}
        want, ready = st.get("desiredNumberScheduled", 0), st.get("numberReady", 0)
        d.err(f"  daemonset {ns}/{name}: {ready}/{want} ready ({time.time() - t0:.0f}s)")
        if want and ready == want:
            return
        if time.time() - t0 > timeout:
            d.die(f"daemonset {ns}/{name} not ready after {timeout:.0f}s")
        time.sleep(15)


# ----------------------------------------------------------------------------- security groups
def cluster_sg() -> str:
    return boto3.client("eks", region_name=REGION).describe_cluster(name=CLUSTER)[
        "cluster"]["resourcesVpcConfig"]["clusterSecurityGroupId"]


def open_sg_rules(env: dict, sg: str) -> None:
    """trainer -> pod :8080 (ingress on the cluster SG, which managed nodes and their pods carry), and
    pod -> trainer adapter ports (the AgentCore demos' SG->SG dial-back rule, with the cluster SG)."""
    from botocore.exceptions import ClientError

    ec2 = boto3.client("ec2", region_name=REGION)
    for tsg in env["trainer_sgs"]:
        try:
            ec2.authorize_security_group_ingress(GroupId=sg, IpPermissions=[{
                "IpProtocol": "tcp", "FromPort": 8080, "ToPort": 8080,
                "UserIdGroupPairs": [{"GroupId": tsg, "Description": "trainer to SWE-bench sandbox pods"}]}])
            d.err(f"opened tcp/8080 on {sg} from {tsg}")
        except ClientError as e:
            if "InvalidPermission.Duplicate" not in str(e):
                raise
    d.open_dialback(sg)


def revoke_sg_rules(sg: str) -> None:
    """Drop every trainer-SG rule that references the cluster SG, or its deletion fails (DependencyViolation)."""
    ec2 = boto3.client("ec2", region_name=REGION)
    for tsg in d.trainer_sgs(d.trainer_private_ip()):
        perms = ec2.describe_security_groups(GroupIds=[tsg])["SecurityGroups"][0]["IpPermissions"]
        for p in perms:
            pairs = [g for g in p.get("UserIdGroupPairs", []) if g["GroupId"] == sg]
            if pairs:
                ec2.revoke_security_group_ingress(GroupId=tsg, IpPermissions=[{
                    k: v for k, v in p.items() if k in ("IpProtocol", "FromPort", "ToPort")} | {
                    "UserIdGroupPairs": [{"GroupId": sg}]}])
                d.err(f"revoked tcp/{p.get('FromPort')}-{p.get('ToPort')} on {tsg} from {sg}")


# ----------------------------------------------------------------------------- smoke
async def _smoke(image: str) -> dict:
    from agentcore_slime_demo.eks_pod_sandbox import EksPodSandbox

    async with EksPodSandbox(image=image) as sb:
        boot = (sb.open_reply or {}).get("boot") or {}
        _, dmesg, _ = await sb.exec("dmesg 2>/dev/null | head -n 3; uname -r", timeout=30)
        _, py, _ = await sb.exec("cd /testbed && /opt/miniconda3/envs/testbed/bin/python -c 'import sys; print(sys.version.split()[0])'",
                                 timeout=60)
        return {"open_s": round(sb.open_recv_epoch - sb.open_sent_epoch, 1), **sb.open_phases,
                "server_start_after_s": round((boot.get("server_start_epoch") or 0) - sb.open_sent_epoch, 1),
                "gvisor": int("gVisor" in dmesg), "dmesg_uname": dmesg.strip().replace("\n", " | "),
                "testbed_python": py.strip()}


def smoke() -> dict:
    if not POOL_MAP.exists():
        d.die(f"{POOL_MAP} missing — push images first (python3 deploy/deploy_swebench.py)")
    image = next(iter(json.loads(POOL_MAP.read_text()).values()))
    res = asyncio.run(_smoke(image))
    d.err(f"[smoke] {res}")
    if os.environ.get("EKS_RUNTIME_CLASS", "gvisor") and not res["gvisor"]:
        d.die("smoke pod is NOT running under gVisor (dmesg has no 'gVisor') — check the RuntimeClass/installer")
    return res


# ----------------------------------------------------------------------------- commands
def _cluster_exists() -> bool:
    eks = boto3.client("eks", region_name=REGION)
    try:
        eks.describe_cluster(name=CLUSTER)
        return True
    except eks.exceptions.ResourceNotFoundException:
        return False


def cmd_render() -> dict:
    env = discover()
    render(env)
    print(f"# rendered deploy/eks/cluster.yaml + deploy/eks/rendered/ (nothing created)\n"
          f"#   cluster {CLUSTER} ({REGION}), VPC {env['vpc_id']}, node subnets "
          f"{[sn['SubnetId'] + '/' + sn['AvailabilityZone'] for sn in env['subnets']]}\n"
          f"#   trainer {env['trainer_ip']} (SGs {env['trainer_sgs']}), API public CIDRs {env['public_cidrs']}\n"
          f"#   creator/admin identity {env['caller']}\n"
          f"#   exists now: {_cluster_exists()}   pool images: "
          f"{len(json.loads(POOL_MAP.read_text())) if POOL_MAP.exists() else 0} ({POOL_MAP.name})", file=sys.stderr)
    return env


def cmd_up() -> None:
    env = cmd_render()
    if os.environ.get("EKS_CREATE") != "1":
        d.die("render only. Creating the cluster is billed (EKS control plane + nodes, ~15-20 min): "
              "re-run with EKS_CREATE=1 once deploy/eks/cluster.yaml looks right.")
    if not _cluster_exists():
        if not shutil.which("eksctl"):
            d.die("eksctl not found on PATH (https://eksctl.io/installation/)")
        d.sh(f"eksctl create cluster -f {EKS_DIR / 'cluster.yaml'}")
    sg = cluster_sg()
    open_sg_rules(env, sg)

    from agentcore_slime_demo.eks_api import get_api

    api = get_api()
    for f in ("gvisor-installer.yaml", "runtimeclass.yaml", "sandbox-namespace.yaml"):
        apply_file(api, RENDERED / f)
    wait_daemonset(api, "kube-system", "acs-gvisor-installer", 900)
    if (RENDERED / "prepull.yaml").exists():
        apply_file(api, RENDERED / "prepull.yaml")
        wait_daemonset(api, NAMESPACE, "acs-swe-prepull", 3600)
    smoke()
    print(f"export EKS_CLUSTER={CLUSTER}\nexport EKS_NAMESPACE={NAMESPACE}")


def cmd_status() -> None:
    if not _cluster_exists():
        print(f"cluster {CLUSTER}: not found")
        return
    from agentcore_slime_demo.eks_api import get_api

    api = get_api()
    for n in api.request("GET", "/api/v1/nodes")["items"]:
        lab = n["metadata"].get("labels", {})
        print(f"node {n['metadata']['name']}  {lab.get('node.kubernetes.io/instance-type')}  "
              f"gvisor={lab.get('acs/gvisor', '-')}")
    pods = api.request("GET", f"/api/v1/namespaces/{NAMESPACE}/pods?labelSelector=app%3Dacs-swe-sandbox")["items"]
    print(f"{len(pods)} sandbox pods in {NAMESPACE}")


def cmd_down() -> None:
    if not _cluster_exists():
        print(f"cluster {CLUSTER}: not found, nothing to delete")
        return
    if os.environ.get("EKS_DELETE") != "1":
        d.die(f"this deletes cluster {CLUSTER} and its nodes; re-run with EKS_DELETE=1")
    revoke_sg_rules(cluster_sg())
    d.sh(f"eksctl delete cluster --name {CLUSTER} --region {REGION} --wait")


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "render"
    {"render": cmd_render, "up": cmd_up, "smoke": smoke, "status": cmd_status, "down": cmd_down}[cmd]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
