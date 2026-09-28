#!/usr/bin/env python3
"""Push the demo_cpu_swebench_eks image pool: ONE wrapped SWE-bench image per instance, in ECR, no Runtimes.

The images are EXACTLY the AgentCore demo's (demo_cpu_swebench/deploy/deploy_swebench.py): the stock
`swebench/sweb.eval.x86_64.<id>` (repo at /testbed, conda env `testbed`) + a standalone CPython 3.12 at
/opt/acs/py + the stdlib HTTP server (acs_swe_http_server.py -> sandbox_server_agent.handle on :8080),
same ECR repo names (${NAME_PREFIX}-swe-<id>) and same recipe tag. So an image the AgentCore demo already
pushed is reused as-is, and both demos grade against byte-identical environments. The difference is
only what consumes them: here each episode's pod (EksPodSandbox) pulls the image onto an EKS node
(cached there for later pods), instead of an AgentCore Runtime per image.

    python3 fetch_swebench.py                     # first: swebench/manifest.json
    python3 deploy/deploy_swebench.py             # build + push -> swebench/image_by_task.json + images.csv

Knobs: SWE_TASKS=a,b (instance ids)  SWE_LIMIT=N  SWE_BUILD_CONC=4  SWE_REBUILD=1
       SWE_NO_PUSH=1 (local docker build only, for local tests)  NAME_PREFIX=acs
Resumable: an image already pushed with this wrapper's recipe tag is not rebuilt.
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
BUILD_CONC = int(os.environ.get("SWE_BUILD_CONC", "4"))
REBUILD = os.environ.get("SWE_REBUILD") == "1"
NO_PUSH = os.environ.get("SWE_NO_PUSH") == "1"
PBS_URL = ("https://github.com/astral-sh/python-build-standalone/releases/download/20260901/"
           "cpython-3.12.14%2B20260901-x86_64-unknown-linux-gnu-install_only.tar.gz")

WRAP = """FROM {base}
# ===== demo_cpu_swebench sandbox wrapper (deploy_swebench.py) =====  (kept verbatim: same recipe tag)
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

    pool_path = OUT / "image_by_task.json"            # {instance -> ECR URI@digest}, read by run.sh
    pool = json.loads(pool_path.read_text()) if pool_path.exists() else {}
    for r in rows:
        if r.get("uri"):
            pool[r["task"]] = r["uri"]
    pool_path.write_text(json.dumps(dict(sorted(pool.items())), indent=2))
    fields = ["task", "repo_name", "status", "size_mb", "build_s", "push_s",
              "base_image", "ecr_repo", "digest", "uri", "error"]
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
    print(f"# pool map: {len(pool)} images -> {pool_path}\n# detail -> {csv_path}")
    return 0 if all(r["status"] in ("built", "built_local") for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
