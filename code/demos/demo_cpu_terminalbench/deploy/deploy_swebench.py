#!/usr/bin/env python3
"""Provision the BENCH=swebench image pool: ONE AgentCore Runtime per SWE-bench instance, on m5 Instances.

Same idea as deploy_terminalbench.py (wrap each task image with a standalone Python + the stdlib
AgentCore HTTP server, one Runtime per image, strict {instance -> ARN} routing), different substrate:

  * Base images are SWE-bench's prebuilt `swebench/sweb.eval.x86_64.<id>` from Docker Hub (repo at
    /testbed on base_commit, conda env `testbed`). They are amd64-only and ~1-2 GB compressed
    (~3-6 GB on disk), so they cannot run on the managed microVM fleet (arm64, <= 2 GB images).
    Each Runtime therefore sits on a CAPACITY PROVIDER of m5 Instances (x86_64, no image-size cap,
    root disk sized for the image): one runtimeSessionId = one whole m5 instance, so an episode is
    still machine-isolated. The capacity provider carries the VPC config (default VPC subnets + SG);
    Runtimes on it take their networking from it. Instances are released CP_IDLE_INSTANCE_TIMEOUT
    (600 s) after their last session ends.
  * Wrapper layer (linux/amd64): standalone CPython 3.12 at /opt/acs/py + acs_http_server.py ->
    sandbox_server_agent.handle as CMD (via acs_swe_http_server.py, which
    also accepts the chunked bodies the Instance data plane sends), WORKDIR /testbed. The testbed env is not touched (grading must
    see exactly the environment SWE-bench validated). The hidden test_patch is NOT baked in: the
    grader (agentcore_slime_demo/swebench.grade_swebench) writes it in after the agent's turns.

    python3 fetch_swebench.py                     # first: swebench/manifest.json
    python3 deploy/deploy_swebench.py             # build + push + (capacity provider) + one Runtime each
    # writes swebench/arn_by_image.json + swebench/images.csv

Knobs: SWE_TASKS=a,b (instance ids)  SWE_LIMIT=N  SWE_BUILD_CONC=4  SWE_REBUILD=1
       SWE_NO_REGISTER=1 (build+push only)   SWE_NO_PUSH=1 (local docker build only, for local tests)
       SWE_INSTANCE=m5.xlarge  SWE_ROOT_GIB=60  NAME_PREFIX=acs (Runtimes ${NAME_PREFIX}_swe_<id>)
Resumable: an image already pushed with this wrapper's recipe tag is not rebuilt; an existing Runtime
is reused (updated in place if it points at another image digest).
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3

HERE = Path(__file__).resolve().parent
DEMO_DIR = HERE.parent
OUT = DEMO_DIR / "swebench"
ACS_SRC = HERE / "acs_ctx"                   # acs_http_server.py lives here (shared with Terminal-Bench)
ACS_CTX = HERE / "acs_ctx_amd64"             # amd64 build context: py/ + server copies (gitignored)
sys.path.insert(0, str(HERE))

import agentcore_deploy as d  # noqa: E402

REGION, PREFIX = d.REGION, d.PREFIX
PLATFORM = "linux/amd64"
INSTANCE = os.environ.get("SWE_INSTANCE", "m5.xlarge")
ROOT_GIB = int(os.environ.get("SWE_ROOT_GIB", "60"))
CP_NAME = f"{PREFIX}_swe_cp"
BUILD_CONC = int(os.environ.get("SWE_BUILD_CONC", "4"))
REBUILD = os.environ.get("SWE_REBUILD") == "1"
NO_REGISTER = os.environ.get("SWE_NO_REGISTER") == "1"
NO_PUSH = os.environ.get("SWE_NO_PUSH") == "1"
PBS_URL = ("https://github.com/astral-sh/python-build-standalone/releases/download/20260901/"
           "cpython-3.12.14%2B20260901-x86_64-unknown-linux-gnu-install_only.tar.gz")

WRAP = """FROM {base}
# ===== demo_cpu_terminalbench BENCH=swebench sandbox wrapper (deploy_swebench.py) =====
USER root
COPY --from=acs py /opt/acs/py
COPY --from=acs acs_http_server.py acs_swe_http_server.py sandbox_server_agent.py /opt/acs/
RUN /opt/acs/py/bin/python3 -c "import http.server, tarfile, json, subprocess" && test -d /testbed/.git \\
 && test -x /opt/miniconda3/envs/testbed/bin/python
WORKDIR /testbed
ENV SANDBOX_WORKDIR=/testbed
EXPOSE 8080
ENTRYPOINT []
CMD ["/opt/acs/py/bin/python3", "/opt/acs/acs_swe_http_server.py"]
"""


def _recipe_tag() -> str:
    h = hashlib.sha256(WRAP.encode())
    for f in ("acs_http_server.py", "acs_swe_http_server.py", "sandbox_server_agent.py"):
        h.update((ACS_CTX / f).read_bytes())
    return "acs-" + h.hexdigest()[:12]


def _ensure_ctx() -> None:
    """amd64 standalone CPython (pruned; libpython debug-stripped) + server copies."""
    py = ACS_CTX / "py"
    if not (py / "bin" / "python3").exists():
        ACS_CTX.mkdir(exist_ok=True)
        d.err("fetching standalone x86_64 CPython (python-build-standalone) -> acs_ctx_amd64/py")
        d.sh(f"curl -sSL -o /tmp/acs_pbs_amd64.tgz '{PBS_URL}' && mkdir -p {py} && "
             f"tar -xzf /tmp/acs_pbs_amd64.tgz -C {py} --strip-components=1")
        lib = py / "lib" / "python3.12"
        for rel in ("include", "share", "lib/pkgconfig"):
            shutil.rmtree(py / rel, ignore_errors=True)
        (py / "lib" / "libpython3.12.a").unlink(missing_ok=True)
        for name in ("test", "idlelib", "tkinter", "turtledemo", "ensurepip", "lib2to3"):
            shutil.rmtree(lib / name, ignore_errors=True)
        for p in list(lib.glob("config-3.12-*")) + list(lib.glob("site-packages/pip*")):
            shutil.rmtree(p, ignore_errors=True)
        for p in list((py / "lib").glob("libtcl*")) + list((py / "lib").glob("libtk*")):
            p.unlink(missing_ok=True)
        # Only libpython is stripped (--strip-debug, -190 MB): `strip` on the PBS python3.12 executable
        # breaks its symbol versioning ("undefined symbol: , version"). No image-size cap here anyway.
        d.sh(f"strip --strip-debug {py}/lib/libpython3.12.so.1.0 && {py}/bin/python3 -c pass")
    shutil.copy(ACS_SRC / "acs_http_server.py", ACS_CTX / "acs_http_server.py")
    shutil.copy(HERE / "acs_swe_http_server.py", ACS_CTX / "acs_swe_http_server.py")   # + chunked bodies
    shutil.copy(DEMO_DIR / "agentcore_slime_demo" / "sandbox_server_agent.py", ACS_CTX / "sandbox_server_agent.py")


def _load_manifest() -> list[dict]:
    mpath = OUT / "manifest.json"
    if not mpath.exists():
        d.die(f"{mpath} not found — run `python3 fetch_swebench.py` first")
    insts = json.loads(mpath.read_text())["instances"]
    only = {t for t in os.environ.get("SWE_TASKS", "").split(",") if t}
    if only:
        insts = [i for i in insts if i["task"] in only]
    if int(os.environ.get("SWE_LIMIT", "0")):
        insts = insts[: int(os.environ["SWE_LIMIT"])]
    if not insts:
        d.die("no instances selected (SWE_TASKS / SWE_LIMIT / manifest)")
    return insts


def repo_name(inst: dict) -> str:
    return f"{PREFIX}-swe-{inst['task'].lower().replace('__', '-')}"


def runtime_name(inst: dict) -> str:
    """[a-zA-Z][a-zA-Z0-9_]{0,47}; long ids get a hash suffix."""
    name = f"{PREFIX}_swe_{inst['task']}".replace("-", "_").replace(".", "_")
    if len(name) > 48:
        name = name[:39] + "_" + hashlib.sha1(inst["task"].encode()).hexdigest()[:8]
    return name


_login_lock = threading.Lock()
_logged_in = False


def _ecr_login(registry: str) -> None:
    global _logged_in
    with _login_lock:
        if not _logged_in:
            d.sh(f"aws ecr get-login-password --region {REGION} | "
                 f"docker login --username AWS --password-stdin {registry}")
            _logged_in = True


def _image_detail(ecr, repo: str, tag: str) -> dict | None:
    try:
        return ecr.describe_images(repositoryName=repo, imageIds=[{"imageTag": tag}])["imageDetails"][0]
    except (ecr.exceptions.ImageNotFoundException, ecr.exceptions.RepositoryNotFoundException):
        return None


def build_one(inst: dict, tag: str, logs: Path) -> dict:
    task, repo = inst["task"], repo_name(inst)
    row = {"task": task, "repo_name": inst["repo"], "base_image": inst["docker_image"], "ecr_repo": repo,
           "status": "", "size_mb": "", "build_s": "", "push_s": "", "digest": "", "error": ""}
    local = f"acs-swe-local:{task.lower()}"
    ecr = boto3.client("ecr", region_name=REGION)
    try:
        uri = None if NO_PUSH else d.ensure_ecr(repo)
        det = None if (REBUILD or NO_PUSH) else _image_detail(ecr, repo, tag)
        if det is None:
            with tempfile.TemporaryDirectory() as ctx, open(logs / f"{task}.log", "w") as lf:
                (Path(ctx) / "Dockerfile").write_text(WRAP.format(base=inst["docker_image"]))
                t0 = time.perf_counter()
                tags = f"-t {local}" + (f" -t {uri}:{tag} -t {uri}:latest" if uri else "")
                subprocess.run(f"docker build --platform {PLATFORM} --provenance=false "
                               f"--build-context acs={ACS_CTX} {tags} {ctx}",
                               shell=True, check=True, stdout=lf, stderr=subprocess.STDOUT)
                row["build_s"] = round(time.perf_counter() - t0, 1)
                if uri:
                    _ecr_login(uri.split("/", 1)[0])
                    t1 = time.perf_counter()
                    subprocess.run(f"docker push -q {uri}:{tag} && docker push -q {uri}:latest", shell=True,
                                   check=True, stdout=lf, stderr=subprocess.STDOUT)
                    row["push_s"] = round(time.perf_counter() - t1, 1)
                    det = _image_detail(ecr, repo, tag)
        if NO_PUSH:
            row["status"] = "built_local"
        else:
            row.update(size_mb=round(det["imageSizeInBytes"] / 2**20, 1), digest=det["imageDigest"],
                       uri=f"{uri}@{det['imageDigest']}", status="built")
    except subprocess.CalledProcessError as e:
        tail = (logs / f"{task}.log").read_text(errors="replace").strip().splitlines()[-3:]
        msg = " | ".join(tail)[-300:] or str(e)
        row.update(status="rate_limited" if "toomanyrequests" in msg.lower() else "build_failed", error=msg)
    except Exception as e:  # noqa: BLE001 — one bad image must not sink the pool
        row.update(status="build_failed", error=f"{type(e).__name__}: {e}"[:300])
    d.err(f"[build] {task}: {row['status']} size_mb={row['size_mb']} build_s={row['build_s']} "
          f"push_s={row['push_s']} {row['error'][:120]}")
    return row


def register_all(rows: list[dict]) -> None:
    xrole = d.exec_role(f"{PREFIX}-exec-role")
    cp_arn = d.capacity_provider(CP_NAME, d.operator_role(f"{PREFIX}-operator-role"), INSTANCE,
                                 root_free_gib=ROOT_GIB)
    d.err(f"capacity provider {CP_NAME} ({INSTANCE}, root {ROOT_GIB} GiB): {cp_arn}")
    control = boto3.client("bedrock-agentcore-control", region_name=REGION)
    env = {"AWS_REGION": REGION}

    def one(row: dict) -> None:
        name = runtime_name({"task": row["task"]})
        t0 = time.perf_counter()
        try:
            arn = d.runtime_on_cp(name, row["uri"], xrole, cp_arn, env)     # create (waits READY) or reuse
            rid = arn.rsplit("/", 1)[-1]
            live = control.get_agent_runtime(agentRuntimeId=rid)
            if live["agentRuntimeArtifact"]["containerConfiguration"]["containerUri"] != row["uri"]:
                d.err(f"[register] {row['task']}: updating {name} to {row['uri'][-20:]}")
                control.update_agent_runtime(
                    agentRuntimeId=rid, roleArn=xrole,
                    agentRuntimeArtifact={"containerConfiguration": {"containerUri": row["uri"]}},
                    protocolConfiguration={"serverProtocol": "HTTP"},
                    capacityProviderConfiguration={"capacityProviderArn": cp_arn},
                    lifecycleConfiguration=d.runtime_lifecycle(d.INSTANCE_MAX_LIFETIME),
                    environmentVariables=env)
                d._wait_runtime(control, rid)
            row.update(status="registered", arn=arn, runtime=name, register_s=round(time.perf_counter() - t0, 1))
        except BaseException as e:  # noqa: BLE001 — incl. SystemExit from d.die inside a worker thread
            row.update(status="register_failed", error=f"{type(e).__name__}: {e}"[:300])
        d.err(f"[register] {row['task']}: {row['status']} {row.get('error', '')[:160]}")

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(one, [r for r in rows if r["status"] == "built"]))


def main() -> int:
    insts = _load_manifest()
    _ensure_ctx()
    tag = _recipe_tag()
    logs = OUT / "build_logs"
    logs.mkdir(parents=True, exist_ok=True)
    d.err(f"building {len(insts)} SWE-bench wrapper images ({PLATFORM}, conc={BUILD_CONC}, tag={tag})")
    t0 = time.perf_counter()
    rows = []
    with ThreadPoolExecutor(max_workers=BUILD_CONC) as ex:
        for f in as_completed([ex.submit(build_one, i, tag, logs) for i in insts]):
            rows.append(f.result())
    rows.sort(key=lambda r: r["task"])
    if not (NO_REGISTER or NO_PUSH):
        register_all(rows)

    pool_path = OUT / "arn_by_image.json"
    pool = json.loads(pool_path.read_text()) if pool_path.exists() else {}
    for r in rows:
        if r.get("arn"):
            pool[r["task"]] = r["arn"]
    pool_path.write_text(json.dumps(dict(sorted(pool.items())), indent=2))
    fields = ["task", "repo_name", "status", "size_mb", "build_s", "push_s", "register_s", "runtime",
              "base_image", "ecr_repo", "digest", "arn", "error"]
    csv_path = OUT / "images.csv"
    old = {}
    if csv_path.exists():
        with csv_path.open() as f:
            old = {r["task"]: r for r in csv.DictReader(f)}
    old.update({r["task"]: r for r in rows})
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(old.values(), key=lambda r: r["task"]))
    by: dict[str, int] = {}
    for r in rows:
        by[r["status"]] = by.get(r["status"], 0) + 1
    print(f"\n# {len(rows)} instances in {time.perf_counter() - t0:.0f}s: {by}")
    print(f"# pool map: {len(pool)} Runtimes -> {pool_path}\n# detail -> {csv_path}")
    return 0 if all(r["status"] in ("registered", "built", "built_local") for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
