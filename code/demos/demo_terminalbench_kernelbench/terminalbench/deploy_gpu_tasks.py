#!/usr/bin/env python3
"""Provision demo_terminalbench_kernelbench's GPU tier: AgentCore Runtimes on a g6 capacity provider (Instances).

Two kinds of GPU sandbox:
  * KernelBench (TB_KERNELBENCH=1, default): ONE image (deploy/Dockerfile.kernelbench_gpu: python + the
    stdlib server, CUDA torch installed at container start) -> Runtime <prefix>_tbhgpu_kernelbench,
    route key `kernelbench-gpu`. Every KernelBench row names that image.
  * Terminal-Bench tasks named in TB_GPU_TASKS=a,b: their own image, as below. No Terminal-Bench task
    needs a GPU today (none requests one; accelerate-maximal-square, sam-cell-seg say CPU-only).

    python3 terminalbench/deploy_gpu_tasks.py                 # the KernelBench Runtime
    TB_GPU_TASKS=a,b python3 terminalbench/deploy_gpu_tasks.py

For each named task: the task's own Dockerfile + the same wrapper layer as every other image (here the
x86_64 standalone CPython), built for linux/amd64 and pushed to ECR <prefix>-tbgpu-<task>; one Runtime
<prefix>_tbhgpu_<task> on the capacity provider <prefix>_tbh_gpu_cp (TB_GPU_INSTANCE, default g6.xlarge,
NAT-egress subnets so run-tests.sh can install pytest). TB_GPU_INSTANCE may list several types,
but AgentCore did NOT fall back between them: on 5 Oct 2026 (no g6.xlarge in any us-west-2 AZ for
hours) a g6.xlarge,g5.xlarge,g6e.xlarge provider only ever retried g6, while g5.xlarge,g6e.xlarge got
a g5.xlarge (A10G; torch 2.5.1/cu124 runs as is) in 227 s. So list the type that has capacity FIRST.
A capacity provider's types can't be changed after creation, so each type list gets its own provider
(<prefix>_tbh_gpu_<g5_g6e>_cp) and the Runtimes are re-pointed to it. AgentCore maps one runtimeSessionId to one whole
instance, so every episode gets its own g6 (cold boot ~60-80 s + image pull; pre-open hides it, and
TB_GPU_MAX_SESSIONS caps how many run at once). The route is `gpu:<runtime arn>`; the pool opens it with
the same AgentCoreRuntimeSandbox as a microVM and heartbeats it the same way.

Creating the capacity provider launches no instance; the first session does. Writes tasks/images_gpu.csv.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import boto3

HERE = Path(__file__).resolve().parent
DEMO_DIR = HERE.parent
TASKS_DIR = HERE / "tasks"
sys.path.insert(0, str(DEMO_DIR / "deploy"))
sys.path.insert(0, str(HERE))
import agentcore_deploy as d  # noqa: E402
import deploy_docker_tasks as dock  # noqa: E402  (x86_64 CPython + the wrapper's server files)
import deploy_terminalbench as tb  # noqa: E402  (_prepare: task Dockerfile + wrapper layer)

GPU_PREFIX = "gpu:"
# Hard cap on one g6 session (AgentCore stops it, and the instance goes 600 s later). An episode takes
# minutes; the cap only bounds a session nobody stopped. 4 h (d.INSTANCE_MAX_LIFETIME is 24 h).
GPU_SESSION_MAX_LIFETIME = int(os.environ.get("TB_GPU_MAX_LIFETIME", "14400"))
INSTANCES = [s.strip() for s in os.environ.get("TB_GPU_INSTANCE", "g6.xlarge").split(",") if s.strip()]
INSTANCE = ",".join(INSTANCES)
# The original single-g6 provider keeps its name; any other type list gets its own.
CP_NAME = (f"{d.PREFIX}_tbh_gpu_cp" if INSTANCES == ["g6.xlarge"]
           else f"{d.PREFIX}_tbh_gpu_{'_'.join(i.split('.')[0] for i in INSTANCES)}_cp")


def runtime_name(task: str) -> str:
    return tb._runtime_name(task).replace(f"{d.PREFIX}_{tb.RUNTIME_TAG}_", f"{d.PREFIX}_{tb.RUNTIME_TAG}gpu_", 1)


def ensure_cp() -> str:
    """The g6 capacity provider, in the NAT-egress subnets of every AZ the VPC endpoints serve."""
    control = boto3.client("bedrock-agentcore-control", region_name=d.REGION)
    for cp in control.list_capacity_providers().get("capacityProviders", []):
        if cp.get("name") == CP_NAME:
            return cp["capacityProviderArn"]
    net = d.vpc_network("nat")["networkModeConfig"]
    d.open_dialback(net["securityGroups"][0])
    resp = control.create_capacity_provider(
        name=CP_NAME, description=f"demo_terminalbench_kernelbench GPU tier ({INSTANCE})",
        permissionsConfiguration={"capacityProviderOperatorRoleArn": d.operator_role(f"{d.PREFIX}-operator-role")},
        computeConfiguration={"ec2Configuration": {
            "launchTemplateSource": {"launchParameters": {
                "operatingSystem": "LINUX_X86_64", "instanceRequirements": {"allowedInstanceTypes": INSTANCES}}},
            "vpcConfiguration": {"subnets": net["subnets"], "securityGroups": net["securityGroups"]},
            "rootVolume": {"freeSpaceGiB": int(os.environ.get("TB_GPU_ROOT_GIB", "60")), "volumeType": "gp3"},
            "lifecycleConfiguration": {"idleInstanceTimeout": d.CP_IDLE_INSTANCE_TIMEOUT,
                                       "maxLifetime": d.INSTANCE_MAX_LIFETIME}}})
    while (st := control.get_capacity_provider(capacityProviderId=resp["capacityProviderId"]))["status"] != "READY":
        if "FAILED" in st["status"]:
            d.die(f"{CP_NAME} {st['status']}: {st.get('statusReason')}")
        time.sleep(5)
    d.err(f"capacity provider {CP_NAME} READY ({INSTANCE}; launches no instance until a session opens)")
    return resp["capacityProviderArn"]


def build_one(task: str, ctx: Path) -> str:
    """Build + push the task's amd64 image; returns its digest URI."""
    repo = f"{d.PREFIX}-tbgpu-{task}"
    uri = d.ensure_ecr(repo)
    tctx, dockerfile = tb._prepare(TASKS_DIR / task)
    d.sh(f"aws ecr get-login-password --region {d.REGION} | docker login --username AWS --password-stdin {uri.split('/')[0]}")
    subprocess.run(f"docker buildx build --builder {d.BUILDX_BUILDER} --platform linux/amd64 --provenance=false "
                   f"--sbom=false --build-context acs={ctx} -f {dockerfile} -t {uri}:latest --push {tctx}",
                   shell=True, check=True)
    det = boto3.client("ecr", region_name=d.REGION).describe_images(
        repositoryName=repo, imageIds=[{"imageTag": "latest"}])["imageDetails"][0]
    return f"{uri}@{det['imageDigest']}"


KB_IMAGE = "kernelbench-gpu"


def build_kernelbench(ctx: Path) -> str:
    """Build + push the KernelBench sandbox image; returns its digest URI."""
    repo = f"{d.PREFIX}-tbgpu-kernelbench"
    uri = d.ensure_ecr(repo)
    d.sh(f"aws ecr get-login-password --region {d.REGION} | docker login --username AWS --password-stdin {uri.split('/')[0]}")
    subprocess.run(f"docker buildx build --builder {d.BUILDX_BUILDER} --platform linux/amd64 --provenance=false "
                   f"--sbom=false -f {DEMO_DIR / 'deploy' / 'Dockerfile.kernelbench_gpu'} -t {uri}:latest --push {ctx}",
                   shell=True, check=True)
    det = boto3.client("ecr", region_name=d.REGION).describe_images(
        repositoryName=repo, imageIds=[{"imageTag": "latest"}])["imageDetails"][0]
    d.err(f"[gpu] kernelbench image {det['imageSizeInBytes'] / 2**20:.0f} MB (cap 2048)")
    return f"{uri}@{det['imageDigest']}"


def _point_at(arn: str, uri: str, xrole: str, cp_arn: str) -> None:
    """Update an existing Runtime in place if it runs a different image digest, capacity provider or session
    lifecycle (RUNTIME_IDLE_TIMEOUT / TB_GPU_MAX_LIFETIME); wait READY."""
    control = boto3.client("bedrock-agentcore-control", region_name=d.REGION)
    rid = arn.rsplit("/", 1)[1]
    cur = control.get_agent_runtime(agentRuntimeId=rid)
    want_lc = d.runtime_lifecycle(GPU_SESSION_MAX_LIFETIME)
    have_lc = {k: (cur.get("lifecycleConfiguration") or {}).get(k) for k in want_lc}
    if (cur["agentRuntimeArtifact"]["containerConfiguration"]["containerUri"] == uri
            and (cur.get("capacityProviderConfiguration") or {}).get("capacityProviderArn") == cp_arn
            and have_lc == want_lc):
        return
    control.update_agent_runtime(
        agentRuntimeId=rid, roleArn=xrole, agentRuntimeArtifact={"containerConfiguration": {"containerUri": uri}},
        protocolConfiguration={"serverProtocol": "HTTP"}, capacityProviderConfiguration={"capacityProviderArn": cp_arn},
        lifecycleConfiguration=want_lc, environmentVariables={"AWS_REGION": d.REGION})
    d.err(f"[gpu] updated {rid} -> {uri.rsplit('@', 1)[-1][:19]} on {cp_arn.rsplit('/', 1)[-1]} "
          f"(idle {want_lc['idleRuntimeSessionTimeout']} s, max {want_lc['maxLifetime']} s)")
    d._wait_runtime(control, rid)


def main() -> int:
    tasks = [t for t in os.environ.get("TB_GPU_TASKS", "").split(",") if t]
    kernelbench = os.environ.get("TB_KERNELBENCH", "1") == "1"
    if not tasks and not kernelbench:
        print("# GPU tier: nothing to deploy (TB_KERNELBENCH=0, no TB_GPU_TASKS)")
        return 0
    missing = [t for t in tasks if not (TASKS_DIR / t).is_dir()]
    if missing:
        d.die(f"unknown task(s): {missing}")
    dock._ensure_standalone_python()
    d._ensure_buildx()
    cp_arn = ensure_cp()
    xrole = d.exec_role(f"{d.PREFIX}-exec-role")
    rows, pool_path = [], TASKS_DIR / "route_by_image.json"
    with tempfile.TemporaryDirectory(prefix="acs-gpu-ctx-") as tmp:
        ctx = Path(tmp)
        shutil.copytree(dock.PY_DIR, ctx / "py", symlinks=True)
        for name, data in dock._server_files().items():
            (ctx / name).write_bytes(data)
        shutil.copy(DEMO_DIR / "deploy" / "acs_http_server_chunked.py", ctx / "acs_http_server_chunked.py")
        if kernelbench:
            row = {"task": KB_IMAGE, "substrate": "gpu", "instance": INSTANCE, "status": "", "arn": "", "error": ""}
            try:
                uri = build_kernelbench(ctx)
                arn = d.runtime_on_cp(runtime_name("kernelbench"), uri, xrole, cp_arn, {"AWS_REGION": d.REGION})
                _point_at(arn, uri, xrole, cp_arn)        # a reused Runtime must run the image just built
                row.update(status="registered", arn=arn)
            except Exception as e:  # noqa: BLE001
                row.update(status="failed", error=f"{type(e).__name__}: {e}"[-300:])
            d.err(f"[gpu] kernelbench: {row['status']} {row['error'][:140]}")
            rows.append(row)
        for task in tasks:
            row = {"task": task, "substrate": "gpu", "instance": INSTANCE, "status": "", "arn": "", "error": ""}
            try:
                uri = build_one(task, ctx)
                arn = d.runtime_on_cp(runtime_name(task), uri, xrole, cp_arn, {"AWS_REGION": d.REGION})
                _point_at(arn, uri, xrole, cp_arn)
                row.update(status="registered", arn=arn)
            except Exception as e:  # noqa: BLE001 — one bad task must not sink the rest
                row.update(status="failed", error=f"{type(e).__name__}: {e}"[-300:])
            d.err(f"[gpu] {task}: {row['status']} {row['error'][:140]}")
            rows.append(row)
    pool = json.loads(pool_path.read_text()) if pool_path.exists() else {}
    for r in rows:
        if r["arn"]:
            pool[r["task"]] = GPU_PREFIX + r["arn"]
    pool_path.write_text(json.dumps(dict(sorted(pool.items())), indent=2))
    with (TASKS_DIR / "images_gpu.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"# GPU tier: {sum(r['status'] == 'registered' for r in rows)}/{len(rows)} task(s) on {CP_NAME} -> {pool_path}")
    return 0 if all(r["arn"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
