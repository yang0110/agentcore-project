#!/usr/bin/env python3
"""Provision demo_hybrid_terminalbench's AgentCore half: one custom microVM Runtime per "microvm" task.

The other half (compose / x86-only / > 2 GB tasks on the m5 Docker host) is deploy_docker_tasks.py.
Tasks whose manifest substrate is "docker" are skipped here; a "microvm" task whose arm64 build fails
or exceeds the 2 GB cap is recorded in tasks/images.csv and picked up by deploy_docker_tasks.py.
Runtimes are named <prefix>_<TB_RUNTIME_TAG>_<task> (TB_RUNTIME_TAG=tbh, so this demo never touches
demo_cpu_terminalbench's acs_tb_* Runtimes); the ECR repos <prefix>-tb-<task> and the image recipe are
SHARED with it on purpose: an image already pushed by either demo is reused, not rebuilt.

This is the demo's whole point: a *large* image pool. Terminal-Bench ships one bespoke Docker
environment per task; we build each one, wrap it so it can serve as an AgentCore sandbox, push it
to its own ECR repo, and register ONE managed serverless Runtime per task. The pool map
(`AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE`, {task -> runtime_arn}) then routes each episode to the
Runtime built from ITS task's image — the same data-driven routing the removed demo_cpu_tasks VARIANT=imagepool used
for 3 images, here at ~229-image scale.

Substrate choice (differs from imagepool on purpose):
  * networkMode = VPC with NAT egress (TB_NETWORK=VPC, default). The mini_claude loop runs INSIDE
    the task's session and dials the trainer's adapter back over the VPC (SG->SG rule on the adapter
    port range, opened here), and Terminal-Bench's `run-tests.sh` installs uv + pytest from the
    internet at grade time, so the session also needs egress: it lands in our private
    `<prefix>-egress-<az>` subnets routed to the VPC's EXISTING NAT gateway (outbound only, no public
    IP, no inbound). TB_NETWORK=PUBLIC = the old pull-only shape (harness then falls back to the trainer).
  * arm64, <= 2048 MB per image (the managed serverless fleet cap, AgentCore service quota). Task
    images that only exist for amd64, fail to build for arm64, or exceed the cap are reported in
    tasks/images.csv and SKIPPED — the pool is "every task that built + fits + registered".

Wrapping (the task's environment is left UNTOUCHED): the task's own Dockerfile builds its
environment; we append a layer that copies in a STANDALONE Python (/opt/acs/py, python-build-standalone,
~54 MB) running a stdlib-only AgentCore HTTP server (acs_http_server.py -> sandbox_server_agent.handle)
as the CMD. Nothing is pip-installed into the task's own interpreter (Terminal-Bench images range from
python3.8 to no Python at all; mutating the task's Python would also change what gets graded), and
the image WORKDIR is not changed (the server reports it; tasks use /app, /app/astropy, /home/alice...).

The hidden oracle (tests/ + run-tests.sh) is NOT baked into the image: the grader injects it from the
trainer's tasks/ dir after the agent's turns (op=put_tar), exactly like Terminal-Bench's own harness
copies tests in only at test time. So the root agent cannot read the tests while solving.

    python3 fetch_terminalbench.py            # first: populate tasks/ + manifest.json
    python3 terminalbench/deploy_terminalbench.py  # then (= steps 2+4): build+push+register the whole pool
    # writes tasks/arn_by_image.json + tasks/images.csv (size / build time / status per task)

Knobs (all optional):
    TB_TASKS=a,b,c    only these tasks (by name)       TB_LIMIT=N      only the first N in the manifest
    TB_BUILD_CONC=16  parallel image builds            TB_REBUILD=1    rebuild even if this wrapper's image is pushed
    TB_NO_REGISTER=1  build + push + measure only (no Runtimes)

Resumable: an image already pushed with this wrapper's recipe tag is not rebuilt, and an existing
Runtime is reused (updated in place if it points at a different image digest).
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import yaml

HERE = Path(__file__).resolve().parent
DEMO_DIR = HERE.parent                                   # code/demos/demo_hybrid_terminalbench
TASKS_DIR = HERE / "tasks"
ACS_CTX = HERE / "acs_ctx"                               # named build context `acs` (python + server)
sys.path.insert(0, str(DEMO_DIR / "deploy"))  # the demo's vendored deploy/agentcore_deploy.py

import agentcore_deploy as d  # noqa: E402

REGION = d.REGION
PREFIX = d.PREFIX
PLATFORM = "linux/arm64"
MAX_IMAGE_MB = 2048                                      # AgentCore "Maximum size of a Docker image"
BUILD_CONC = int(os.environ.get("TB_BUILD_CONC", "16"))
REBUILD = os.environ.get("TB_REBUILD") == "1"
NO_REGISTER = os.environ.get("TB_NO_REGISTER") == "1"
RUNTIME_TAG = os.environ.get("TB_RUNTIME_TAG", "tbh")    # Runtime names <prefix>_<tag>_<task>
RETRY_MICROVM = os.environ.get("TB_RETRY_MICROVM") == "1"  # retry tasks images.csv already marked unfit
PBS_URL = ("https://github.com/astral-sh/python-build-standalone/releases/download/20260901/"
           "cpython-3.12.14%2B20260901-aarch64-unknown-linux-gnu-install_only.tar.gz")

# Appended to each task's own Dockerfile. Kept BYTE-IDENTICAL to demo_cpu_terminalbench's (comment line
# included): the recipe tag hashes it, so the same text means the images already in ECR are reused. The last CMD wins (the task's `sleep infinity` compose
# command is not used on AgentCore); ENTRYPOINT is cleared so bases like localstack don't wrap it.
# The RUN is a build-time sanity check that the standalone Python runs on this base (glibc, arm64).
WRAP_APPENDIX = """
# ===== demo_cpu_terminalbench sandbox wrapper (appended by deploy_terminalbench.py) =====
USER root
COPY --from=acs py /opt/acs/py
COPY --from=acs acs_http_server.py sandbox_server_agent.py /opt/acs/
RUN /opt/acs/py/bin/python3 -c "import http.server, tarfile, json, subprocess"
{env_lines}
EXPOSE 8080
ENTRYPOINT []
CMD ["/opt/acs/py/bin/python3", "/opt/acs/acs_http_server.py"]
""".lstrip()


def _recipe_tag() -> str:
    """Image tag that changes whenever the wrapper changes (so a stale image is rebuilt)."""
    h = hashlib.sha256(WRAP_APPENDIX.encode())
    for f in ("acs_http_server.py", "sandbox_server_agent.py"):
        h.update((ACS_CTX / f).read_bytes())
    return "acs-" + h.hexdigest()[:12]


def _ensure_standalone_python() -> None:
    """Fetch + prune + strip the arm64 standalone CPython into acs_ctx/py (once; gitignored)."""
    py = ACS_CTX / "py"
    if (py / "bin" / "python3").exists():
        return
    ACS_CTX.mkdir(exist_ok=True)
    d.err("fetching standalone arm64 CPython (python-build-standalone) -> acs_ctx/py")
    d.sh(f"curl -sSL -o /tmp/acs_pbs.tgz '{PBS_URL}' && mkdir -p {py} && "
         f"tar -xzf /tmp/acs_pbs.tgz -C {py} --strip-components=1")
    lib = py / "lib" / "python3.12"
    for rel in ("include", "share", "lib/pkgconfig", "lib/libpython3.12.a"):
        shutil.rmtree(py / rel, ignore_errors=True) if (py / rel).is_dir() else (py / rel).unlink(missing_ok=True)
    for name in ("test", "idlelib", "tkinter", "turtledemo", "ensurepip", "lib2to3"):
        shutil.rmtree(lib / name, ignore_errors=True)
    for p in list(lib.glob("config-3.12-*")) + list(lib.glob("site-packages/pip*")):
        shutil.rmtree(p, ignore_errors=True)
    for p in list((py / "lib").glob("libtcl*")) + list((py / "lib").glob("libtk*")):
        p.unlink(missing_ok=True)
    # Strip debug info (~190 MB -> ~54 MB) with arm64 binutils (the host strip is x86-only).
    d.sh(f"docker run --rm --platform {PLATFORM} -v {py}:/py public.ecr.aws/docker/library/debian:bookworm-slim "
         "bash -c 'apt-get update -qq >/dev/null && apt-get install -y -qq binutils >/dev/null && "
         "strip --strip-unneeded /py/bin/python3.12 /py/lib/libpython3.12.so.1.0 && "
         "find /py/lib -name \"*.so\" -exec strip --strip-unneeded {} + && "
         f"chown -R {os.getuid()}:{os.getgid()} /py'")


def _prior_unfit() -> dict[str, dict]:
    """images.csv rows of an earlier run that did not fit a microVM (build_failed / too_large)."""
    p = TASKS_DIR / "images.csv"
    if not p.exists() or RETRY_MICROVM:
        return {}
    with p.open() as f:
        return {r["task"]: r for r in csv.DictReader(f) if r.get("status") in ("build_failed", "too_large")}


def _load_manifest() -> list[dict]:
    mpath = TASKS_DIR / "manifest.json"
    if not mpath.exists():
        d.die(f"{mpath} not found — run `python3 terminalbench/fetch_terminalbench.py` first")
    tasks = [t for t in json.loads(mpath.read_text())["tasks"] if t.get("substrate", "microvm") == "microvm"]
    only = {t for t in os.environ.get("TB_TASKS", "").split(",") if t}
    if only:
        tasks = [t for t in tasks if t["task"] in only]
    limit = int(os.environ.get("TB_LIMIT", "0"))
    if limit:
        tasks = tasks[:limit]
    if not tasks:
        d.die("no tasks selected (check TB_TASKS / TB_LIMIT / manifest)")
    return tasks


def _compose_client(task_dir: Path) -> tuple[Path, Path, dict]:
    """(build context, Dockerfile, extra env) of the task's single `client` service."""
    ctx, dockerfile, env = task_dir, task_dir / "Dockerfile", {}
    for name in ("docker-compose.yaml", "docker-compose.yml"):
        cp = task_dir / name
        if not cp.exists():
            continue
        svcs = (yaml.safe_load(cp.read_text()) or {}).get("services") or {}
        svc = svcs.get("client") or (next(iter(svcs.values())) if svcs else {})
        build = svc.get("build") or {}
        if isinstance(build, str):
            build = {"context": build}
        ctx = (task_dir / build.get("context", ".")).resolve()
        dockerfile = ctx / build.get("dockerfile", "Dockerfile")
        raw = svc.get("environment") or []
        items = raw.items() if isinstance(raw, dict) else (e.split("=", 1) for e in raw if "=" in e)
        # T_BENCH_* placeholders are harness plumbing (TEST_DIR is set by the grader instead).
        env = {k: str(v) for k, v in items if "${" not in str(v) and k != "TEST_DIR"}
        break
    return ctx, dockerfile, env


def _prepare(task_dir: Path) -> tuple[Path, Path]:
    """Write <context>/Dockerfile.acs = the task's Dockerfile (amd64 pins dropped) + our wrapper."""
    ctx, dockerfile, env = _compose_client(task_dir)
    base = dockerfile.read_text()
    # A few tasks pin `FROM --platform=linux/amd64`; their bases are multi-arch, so try arm64.
    base = re.sub(r"(?m)^(\s*FROM\s+)--platform=\S+\s+", r"\1", base)
    env_lines = "\n".join(f"ENV {k}={json.dumps(v)}" for k, v in env.items())
    out = ctx / "Dockerfile.acs"
    out.write_text(base.rstrip() + "\n\n" + WRAP_APPENDIX.format(env_lines=env_lines))
    return ctx, out


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
    except ecr.exceptions.ImageNotFoundException:
        return None
    except ecr.exceptions.RepositoryNotFoundException:
        return None


def build_one(t: dict, tag: str, logs: Path) -> dict:
    """Build + push one task image (unless this recipe is already pushed); measure its size."""
    task = t["task"]
    repo = f"{PREFIX}-tb-{t['image']}"
    row = {"task": task, "difficulty": t.get("difficulty", ""), "category": t.get("category", ""),
           "repo": repo, "status": "", "size_mb": "", "layers": "", "build_s": "", "digest": "",
           "error": ""}
    ecr = boto3.client("ecr", region_name=REGION)
    try:
        uri = d.ensure_ecr(repo)
        registry = uri.split("/", 1)[0]
        det = None if REBUILD else _image_detail(ecr, repo, tag)
        if det is None:
            ctx, dockerfile = _prepare(TASKS_DIR / task)
            _ecr_login(registry)
            t0 = time.perf_counter()
            with open(logs / f"{task}.log", "w") as lf:
                subprocess.run(
                    f"docker buildx build --builder {d.BUILDX_BUILDER} --platform {PLATFORM} "
                    f"--provenance=false --sbom=false --build-context acs={ACS_CTX} "
                    f"-f {dockerfile} -t {uri}:{tag} -t {uri}:latest --push {ctx}",
                    shell=True, check=True, stdout=lf, stderr=subprocess.STDOUT)
            row["build_s"] = round(time.perf_counter() - t0, 1)
            det = _image_detail(ecr, repo, tag)
        mb = det["imageSizeInBytes"] / 2**20                  # compressed (what a pull downloads)
        row.update(size_mb=round(mb, 1), digest=det["imageDigest"], uri=f"{uri}@{det['imageDigest']}")
        mf = json.loads(ecr.batch_get_image(
            repositoryName=repo, imageIds=[{"imageDigest": det["imageDigest"]}],
        )["images"][0]["imageManifest"])
        row["layers"] = len(mf.get("layers", []))
        row["status"] = "too_large" if mb > MAX_IMAGE_MB else "built"
    except subprocess.CalledProcessError as e:
        tail = (logs / f"{task}.log").read_text(errors="replace").strip().splitlines()[-3:]
        row.update(status="build_failed", error=" | ".join(tail)[-300:] or str(e))
    except Exception as e:  # noqa: BLE001 — one bad task must not sink the whole pool
        row.update(status="build_failed", error=f"{type(e).__name__}: {e}"[:300])
    d.err(f"[build] {task}: {row['status']} size_mb={row['size_mb']} build_s={row['build_s']} {row['error'][:120]}")
    return row


def _runtime_name(image: str) -> str:
    """AgentCore runtime names: [a-zA-Z][a-zA-Z0-9_]{0,47}. Long task names get a hash suffix."""
    name = f"{PREFIX}_{RUNTIME_TAG}_{image}".replace("-", "_")
    if len(name) > 48:
        name = name[:39] + "_" + hashlib.sha1(image.encode()).hexdigest()[:8]
    return name


def register_all(rows: list[dict], xrole: str) -> None:
    """Create (or update to the new digest) one Runtime per built task; wait READY in parallel."""
    control = boto3.client("bedrock-agentcore-control", region_name=REGION)
    existing, tok = {}, None
    while True:
        resp = control.list_agent_runtimes(**({"nextToken": tok} if tok else {}), maxResults=100)
        for r in resp.get("agentRuntimes", []):
            existing[r["agentRuntimeName"]] = r
        tok = resp.get("nextToken")
        if not tok:
            break
    art = lambda uri: {"containerConfiguration": {"containerUri": uri}}  # noqa: E731
    vpc = os.environ.get("TB_NETWORK", "VPC").upper() == "VPC"
    network = d.vpc_network("nat") if vpc else {"networkMode": "PUBLIC"}
    if vpc:
        d.open_dialback(network["networkModeConfig"]["securityGroups"][0])
    common = dict(roleArn=xrole, protocolConfiguration={"serverProtocol": "HTTP"},
                  networkConfiguration=network,
                  lifecycleConfiguration=d.runtime_lifecycle(d.MICROVM_MAX_LIFETIME),  # RUNTIME_IDLE_TIMEOUT
                  environmentVariables={"AWS_REGION": REGION})

    def one(row: dict) -> None:
        name = _runtime_name(row["repo"].removeprefix(f"{PREFIX}-tb-"))
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
                if (live["agentRuntimeArtifact"]["containerConfiguration"]["containerUri"] != row["uri"]
                        or live.get("networkConfiguration", {}).get("networkMode") != network["networkMode"]
                        or {k: v for k, v in (live.get("networkConfiguration", {}).get("networkModeConfig") or {}).items()
                            if k in ("securityGroups", "subnets")} != (network.get("networkModeConfig") or {})):
                    control.update_agent_runtime(agentRuntimeId=rid, agentRuntimeArtifact=art(row["uri"]), **common)
            while True:
                st = control.get_agent_runtime(agentRuntimeId=rid)
                if st["status"] == "READY":
                    break
                if "FAILED" in st["status"]:
                    raise RuntimeError(f"{st['status']}: {st.get('failureReason')}")
                time.sleep(10)
            row.update(status="registered", arn=arn, register_s=round(time.perf_counter() - t0, 1))
        except Exception as e:  # noqa: BLE001
            row.update(status="register_failed", error=f"{type(e).__name__}: {e}"[:300])
        d.err(f"[register] {row['task']}: {row['status']} {row.get('error', '')[:120]}")

    todo = [r for r in rows if r["status"] == "built"]
    with ThreadPoolExecutor(max_workers=8) as ex:     # control-plane mutation quota is 50/s
        list(ex.map(one, todo))


def main() -> int:
    tasks = _load_manifest()
    _ensure_standalone_python()
    shutil.copy(DEMO_DIR / "agentcore_slime_demo" / "sandbox_server_agent.py", ACS_CTX / "sandbox_server_agent.py")
    tag = _recipe_tag()
    d._ensure_buildx()
    logs = TASKS_DIR / "build_logs"
    logs.mkdir(parents=True, exist_ok=True)

    d.err(f"building {len(tasks)} Terminal-Bench task images ({PLATFORM}, conc={BUILD_CONC}, tag={tag})")
    t0 = time.perf_counter()
    prior = _prior_unfit()
    rows = [prior[t["task"]] for t in tasks if t["task"] in prior]       # known unfit: not rebuilt
    if rows:
        d.err(f"{len(rows)} task(s) already marked unfit for a microVM in images.csv (TB_RETRY_MICROVM=1 to retry)")
    with ThreadPoolExecutor(max_workers=BUILD_CONC) as ex:
        futs = [ex.submit(build_one, t, tag, logs) for t in tasks if t["task"] not in prior]
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if i % 10 == 0:
                d.err(f"  ... {i}/{len(tasks)} images done ({time.perf_counter() - t0:.0f}s)")
    rows.sort(key=lambda r: r["task"])

    if not NO_REGISTER:
        register_all(rows, d.exec_role(f"{PREFIX}-exec-role"))

    # Merge into the pool map (other tasks deployed by an earlier TB_TASKS/TB_LIMIT run are kept).
    pool_path = TASKS_DIR / "route_by_image.json"
    pool = json.loads(pool_path.read_text()) if pool_path.exists() else {}
    for r in rows:
        if str(pool.get(r["task"], "")).startswith(("docker://", "gpu:")):
            continue                              # moved to the m5 / the GPU tier: keep that route
        if r.get("arn"):
            pool[r["task"]] = r["arn"]
        elif (r["task"] in pool and not str(pool[r["task"]]).startswith(("docker://", "gpu:"))
              and r["status"] != "registered" and not (NO_REGISTER and r["status"] == "built")):
            pool.pop(r["task"])                   # rebuilt but no longer registrable: drop it (a build-only run keeps it)
    pool_path.write_text(json.dumps(dict(sorted(pool.items())), indent=2))

    fields = ["task", "difficulty", "category", "status", "size_mb", "layers", "build_s",
              "register_s", "repo", "digest", "arn", "error"]
    with (TASKS_DIR / "images.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    by = {}
    for r in rows:
        by[r["status"]] = by.get(r["status"], 0) + 1
    sizes = sorted(r["size_mb"] for r in rows if isinstance(r["size_mb"], float))
    print(f"\n# {len(rows)} tasks in {time.perf_counter() - t0:.0f}s: {by}")
    if sizes:
        pct = lambda p: sizes[min(len(sizes) - 1, int(p * len(sizes)))]  # noqa: E731
        print(f"# image size MB (compressed): min={sizes[0]} p50={pct(.5)} p95={pct(.95)} max={sizes[-1]}")
    n_dc = sum(1 for v in pool.values() if str(v).startswith("docker://"))
    print(f"# route map: {len(pool) - n_dc} microVM Runtimes + {n_dc} m5 docker tasks -> {pool_path}\n"
          f"# per-task detail -> {TASKS_DIR / 'images.csv'}")
    return 0 if pool or (NO_REGISTER and by.get("built")) else 1   # a build-only run has no pool of its own


if __name__ == "__main__":
    sys.exit(main())
