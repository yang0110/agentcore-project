#!/usr/bin/env python3
"""Provision the demo_cpu_swebench image pool: ONE AgentCore Runtime per SWE-bench instance, on the
managed arm64 microVM fleet (<= 2 GB image cap).

Same idea as demo_cpu_tasks' terminalbench/deploy_terminalbench.py (wrap each task image with a standalone
Python + the stdlib AgentCore HTTP server, one Runtime per image, strict {instance -> ARN} routing):

  * Base images are Epoch AI's arm64 rebuilds of the SWE-bench harness images,
    `ghcr.io/epoch-research/swe-bench.eval.arm64.<id>` (same layout as the official x86_64 ones: repo at
    /testbed on base_commit, conda env `testbed`). 420 of the 500 Verified rows have one, all <= 1.8 GB
    compressed (swebench/verified_image_sizes.csv, survey_image_sizes.py), so the wrapped image fits the
    microVM cap. The cap counts the COMPRESSED ECR size (imageSizeInBytes): 2041.7 MiB was accepted,
    2061.7 MiB rejected with ServiceQuotaExceededException (probe_image_cap.py, swebench/capprobe*.csv).
    An image over MAX_IMAGE_MB is marked too_large and never registered.
  * Wrapper layer (linux/arm64, cross-built with buildx + qemu): standalone CPython 3.12 at /opt/acs/py
    (debug-stripped, ~54 MB) + acs_http_server.py -> sandbox_server_agent.handle as CMD (via
    acs_swe_http_server.py, which accepts both chunked and Content-Length bodies), WORKDIR /testbed. The
    testbed env is not touched (grading must see exactly the environment SWE-bench validated). The hidden
    test_patch is NOT baked in: the grader (agentcore_slime_demo/swebench.grade_swebench) writes it in
    after the agent's turns.
  * Runtimes are plain managed microVM Runtimes in VPC mode (RUNTIME_EGRESS=none by default: the default
    subnets, no internet; =nat for PyPI) with the SG->SG dial-back rule for the in-session loop. One
    session = one microVM, never reused.

    python3 fetch_swebench.py                     # first: swebench/manifest.json
    python3 deploy/deploy_swebench.py             # build + push + one Runtime each
    # writes swebench/arn_by_image.json + swebench/images.csv

Knobs: SWE_TASKS=a,b (instance ids)  SWE_LIMIT=N  SWE_BUILD_CONC=4  SWE_REBUILD=1
       SWE_NO_REGISTER=1 (build+push only)   SWE_NO_PUSH=1 (local arm64 build only, for local tests)
       RUNTIME_EGRESS=none|nat   NAME_PREFIX=acs (Runtimes ${NAME_PREFIX}_swearm_<id>)
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
ACS_CTX = HERE / "acs_ctx"                   # arm64 build context: py/ + server copies (py/ gitignored)
sys.path.insert(0, str(HERE))

import agentcore_deploy as d  # noqa: E402

REGION, PREFIX = d.REGION, d.PREFIX
PLATFORM = "linux/arm64"
MAX_IMAGE_MB = 2048                          # AgentCore quota L-0A9E32B3, compressed MiB (not adjustable)
BUILD_CONC = int(os.environ.get("SWE_BUILD_CONC", "4"))
REBUILD = os.environ.get("SWE_REBUILD") == "1"
NO_REGISTER = os.environ.get("SWE_NO_REGISTER") == "1"
NO_PUSH = os.environ.get("SWE_NO_PUSH") == "1"
PBS_URL = ("https://github.com/astral-sh/python-build-standalone/releases/download/20260901/"
           "cpython-3.12.14%2B20260901-aarch64-unknown-linux-gnu-install_only.tar.gz")

WRAP = """FROM {base}
# ===== demo_cpu_swebench arm64 sandbox wrapper (deploy_swebench.py) =====
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
    """arm64 standalone CPython (pruned + stripped with arm64 binutils) + server copies."""
    py = ACS_CTX / "py"
    if not (py / "bin" / "python3").exists():
        d.err("fetching standalone arm64 CPython (python-build-standalone) -> acs_ctx/py")
        d.sh(f"curl -sSL -o /tmp/acs_pbs_arm64.tgz '{PBS_URL}' && mkdir -p {py} && "
             f"tar -xzf /tmp/acs_pbs_arm64.tgz -C {py} --strip-components=1")
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
        # Strip debug info (~190 MB -> ~54 MB) with arm64 binutils (the host strip is x86-only). Every MB
        # counts against the 2 GB cap.
        d.sh(f"docker run --rm --platform {PLATFORM} -v {py}:/py public.ecr.aws/docker/library/debian:bookworm-slim "
             "bash -c 'apt-get update -qq >/dev/null && apt-get install -y -qq binutils >/dev/null && "
             "strip --strip-unneeded /py/bin/python3.12 /py/lib/libpython3.12.so.1.0 && "
             "find /py/lib -name \"*.so\" -exec strip --strip-unneeded {} + && "
             f"chown -R {os.getuid()}:{os.getgid()} /py'")
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
    """ECR repo; `-arm64-` keeps these apart from the amd64 `acs-swe-*` repos the EKS / ECS demos share."""
    return f"{PREFIX}-swe-arm64-{inst['task'].lower().replace('__', '-')}"


def runtime_name(inst: dict) -> str:
    """[a-zA-Z][a-zA-Z0-9_]{0,47}; long ids get a hash suffix. `_swearm_` never collides with the
    retired m5 Runtimes (`${PREFIX}_swe_<id>`, see swebench/m5_legacy/)."""
    name = f"{PREFIX}_swearm_{inst['task']}".replace("-", "_").replace(".", "_")
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
           "status": "", "size_mb": "", "build_s": "", "digest": "", "error": ""}
    local = f"acs-swe-arm64-local:{task.lower()}"
    ecr = boto3.client("ecr", region_name=REGION)
    try:
        uri = None if NO_PUSH else d.ensure_ecr(repo)
        det = None if (REBUILD or NO_PUSH) else _image_detail(ecr, repo, tag)
        if det is None:
            if uri:
                _ecr_login(uri.split("/", 1)[0])
            with tempfile.TemporaryDirectory() as ctx, open(logs / f"{task}.log", "w") as lf:
                (Path(ctx) / "Dockerfile").write_text(WRAP.format(base=inst["docker_image"]))
                t0 = time.perf_counter()
                out = f"-t {uri}:{tag} -t {uri}:latest --push" if uri else f"-t {local} --load"
                # --provenance/--sbom=false: a plain single-arch image (containerUri rejects attestation lists)
                subprocess.run(f"docker buildx build --builder {d.BUILDX_BUILDER} --platform {PLATFORM} "
                               f"--provenance=false --sbom=false --build-context acs={ACS_CTX} {out} {ctx}",
                               shell=True, check=True, stdout=lf, stderr=subprocess.STDOUT)
                row["build_s"] = round(time.perf_counter() - t0, 1)
                if uri:
                    det = _image_detail(ecr, repo, tag)
        if NO_PUSH:
            row["status"] = "built_local"
        else:
            mb = det["imageSizeInBytes"] / 2**20                 # compressed: what the cap counts
            row.update(size_mb=round(mb, 1), digest=det["imageDigest"], uri=f"{uri}@{det['imageDigest']}",
                       status="too_large" if mb > MAX_IMAGE_MB else "built")
    except subprocess.CalledProcessError as e:
        tail = (logs / f"{task}.log").read_text(errors="replace").strip().splitlines()[-3:]
        msg = " | ".join(tail)[-300:] or str(e)
        row.update(status="rate_limited" if "toomanyrequests" in msg.lower() else "build_failed", error=msg)
    except Exception as e:  # noqa: BLE001 — one bad image must not sink the pool
        row.update(status="build_failed", error=f"{type(e).__name__}: {e}"[:300])
    d.err(f"[build] {task}: {row['status']} size_mb={row['size_mb']} build_s={row['build_s']} {row['error'][:120]}")
    return row


def register_all(rows: list[dict]) -> None:
    """Create (or update to the new digest / network) one microVM Runtime per built image; wait READY."""
    xrole = d.exec_role(f"{PREFIX}-exec-role")
    control = boto3.client("bedrock-agentcore-control", region_name=REGION)
    existing, tok = {}, None
    while True:
        resp = control.list_agent_runtimes(**({"nextToken": tok} if tok else {}), maxResults=100)
        for r in resp.get("agentRuntimes", []):
            existing[r["agentRuntimeName"]] = r
        tok = resp.get("nextToken")
        if not tok:
            break
    network = d.vpc_network()                                  # RUNTIME_EGRESS=none (default) | nat
    d.open_dialback(network["networkModeConfig"]["securityGroups"][0])
    art = lambda uri: {"containerConfiguration": {"containerUri": uri}}  # noqa: E731
    common = dict(roleArn=xrole, protocolConfiguration={"serverProtocol": "HTTP"},
                  networkConfiguration=network,
                  lifecycleConfiguration=d.runtime_lifecycle(d.MICROVM_MAX_LIFETIME),  # RUNTIME_IDLE_TIMEOUT
                  environmentVariables={"AWS_REGION": REGION})

    def one(row: dict) -> None:
        name = runtime_name({"task": row["task"]})
        t0 = time.perf_counter()
        try:
            cur = existing.get(name)
            if cur and "FAILED" in cur.get("status", ""):
                control.delete_agent_runtime(agentRuntimeId=cur["agentRuntimeId"])
                time.sleep(15)
                cur = None
            if cur is None:
                r = control.create_agent_runtime(agentRuntimeName=name, agentRuntimeArtifact=art(row["uri"]), **common)
                rid, arn = r["agentRuntimeId"], r["agentRuntimeArn"]
            else:
                rid, arn = cur["agentRuntimeId"], cur["agentRuntimeArn"]
                live = control.get_agent_runtime(agentRuntimeId=rid)
                live_net = live.get("networkConfiguration", {})
                if (live["agentRuntimeArtifact"]["containerConfiguration"]["containerUri"] != row["uri"]
                        or live_net.get("networkMode") != network["networkMode"]
                        or {k: v for k, v in (live_net.get("networkModeConfig") or {}).items()
                            if k in ("securityGroups", "subnets")} != network["networkModeConfig"]):
                    d.err(f"[register] {row['task']}: updating {name} to {row['uri'][-20:]}")
                    control.update_agent_runtime(agentRuntimeId=rid, agentRuntimeArtifact=art(row["uri"]), **common)
            while True:
                st = control.get_agent_runtime(agentRuntimeId=rid)
                if st["status"] == "READY":
                    break
                if "FAILED" in st["status"]:
                    raise RuntimeError(f"{st['status']}: {st.get('failureReason')}")
                time.sleep(10)
            row.update(status="registered", arn=arn, runtime=name, register_s=round(time.perf_counter() - t0, 1))
        except Exception as e:  # noqa: BLE001 — incl. the quota error of an image that slipped past the gate
            row.update(status="register_failed", error=f"{type(e).__name__}: {e}"[:300])
        d.err(f"[register] {row['task']}: {row['status']} {row.get('error', '')[:160]}")

    with ThreadPoolExecutor(max_workers=8) as ex:     # control-plane mutation quota is 50/s
        list(ex.map(one, [r for r in rows if r["status"] == "built"]))


def main() -> int:
    insts = _load_manifest()
    _ensure_ctx()
    d._ensure_buildx()
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
    if pool or pool_path.exists():    # never leave an EMPTY map: fetch_swebench --data-only would keep 0 rows
        pool_path.write_text(json.dumps(dict(sorted(pool.items())), indent=2))
    fields = ["task", "repo_name", "status", "size_mb", "build_s", "register_s", "runtime",
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
