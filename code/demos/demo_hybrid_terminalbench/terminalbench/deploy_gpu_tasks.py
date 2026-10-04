#!/usr/bin/env python3
"""Provision demo_hybrid_terminalbench's GPU tier: AgentCore Runtimes on a g6 capacity provider (Instances).

No Terminal-Bench task needs a GPU today (none requests one; accelerate-maximal-square, sam-cell-seg say
CPU-only), so this tier is EMPTY by default and nothing is created. Name tasks to put on it:

    TB_GPU_TASKS=a,b python3 terminalbench/deploy_gpu_tasks.py

For each named task: the task's own Dockerfile + the same wrapper layer as every other image (here the
x86_64 standalone CPython), built for linux/amd64 and pushed to ECR <prefix>-tbgpu-<task>; one Runtime
<prefix>_tbhgpu_<task> on the capacity provider <prefix>_tbh_gpu_cp (TB_GPU_INSTANCE, default g6.xlarge,
NAT-egress subnets so run-tests.sh can install pytest). AgentCore maps one runtimeSessionId to one whole
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
INSTANCE = os.environ.get("TB_GPU_INSTANCE", "g6.xlarge")
CP_NAME = f"{d.PREFIX}_tbh_gpu_cp"


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
        name=CP_NAME, description=f"demo_hybrid_terminalbench GPU tier ({INSTANCE})",
        permissionsConfiguration={"capacityProviderOperatorRoleArn": d.operator_role(f"{d.PREFIX}-operator-role")},
        computeConfiguration={"ec2Configuration": {
            "launchTemplateSource": {"launchParameters": {
                "operatingSystem": "LINUX_X86_64", "instanceRequirements": {"allowedInstanceTypes": [INSTANCE]}}},
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


def main() -> int:
    tasks = [t for t in os.environ.get("TB_GPU_TASKS", "").split(",") if t]
    if not tasks:
        print("# GPU tier: no TB_GPU_TASKS — nothing to deploy (no Terminal-Bench task needs a GPU)")
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
        for task in tasks:
            row = {"task": task, "substrate": "gpu", "instance": INSTANCE, "status": "", "arn": "", "error": ""}
            try:
                uri = build_one(task, ctx)
                arn = d.runtime_on_cp(runtime_name(task), uri, xrole, cp_arn, {"AWS_REGION": d.REGION})
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
