#!/usr/bin/env python3
"""Provision demo_terminalbench_kernelbench's m5 half: build every task the microVM pool can't run, on the m5.

Which tasks land on the m5 Docker host (data-driven, nothing hardcoded):
  * manifest substrate "docker"   — multi-container compose tasks (fetch_terminalbench.py, 12)
  * images.csv build_failed       — the arm64 build failed (x86-only bases, x86 asm, qemu-system-x86 mirror, ...)
  * images.csv too_large          — over the 2 GB microVM image cap (the CUDA-torch mteb/png images)
  * oracle.csv on a microVM task  — its reference solution FAILED (or errored) on its microVM: the task
                                    needs something a capability-less microVM lacks (raw networking,
                                    KVM, mounts, ...). Retried on the m5 (full root in a container);
                                    validate_oracle.py then decides which substrate keeps it.
                                    TB_DOCKER_ORACLE_FALLBACK=0 turns this off.
  * TB_DOCKER_TASKS=a,b           — force more tasks onto the m5 (e.g. for an A/B against their microVM)

For each one: upload the task dir WITHOUT its hidden oracle and reference solution (tests/, run-tests.sh,
solution.sh/.yaml — the grader injects the tests from this box after the agent stops, exactly as on the
microVM path) plus any minimal bit-rot patch (PATCHES below), then the m5 agent builds the task's compose
images natively for linux/amd64: the client = the task's own Dockerfile + the SAME wrapper layer as the
microVM images (standalone CPython + acs_http_server.py, here the x86_64 build), sidecars as upstream.
Built tasks get a route docker://<m5 ip>:<port>/<task> in tasks/route_by_image.json; per-task detail
goes to tasks/images_docker.csv. Resumable: a task whose files + wrapper are unchanged is not rebuilt.

    python3 deploy/m5_host.py up                     # first: the m5 host (deploy/m5_host.json)
    python3 terminalbench/deploy_docker_tasks.py     # = the docker half of step 2
    TB_TASKS=postgres-csv-clean python3 terminalbench/deploy_docker_tasks.py

Knobs: TB_DOCKER_BUILD_CONC=4 (parallel builds on the m5), TB_TASKS / TB_LIMIT (subset),
TB_DOCKER_TASKS (extra tasks), TB_REBUILD=1 (force).
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEMO_DIR = HERE.parent
TASKS_DIR = HERE / "tasks"
ACS_CTX = HERE / "acs_ctx"
sys.path.insert(0, str(DEMO_DIR / "deploy"))
sys.path.insert(0, str(HERE))
import agentcore_deploy as d  # noqa: E402
import m5_host  # noqa: E402
from deploy_terminalbench import WRAP_APPENDIX  # noqa: E402  (the same wrapper layer as the microVM images)

PLATFORM = "linux/amd64"
PY_DIR = ACS_CTX / "py-x86_64"
PBS_URL = ("https://github.com/astral-sh/python-build-standalone/releases/download/20260901/"
           "cpython-3.12.14%2B20260901-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz")
BUILD_CONC = int(os.environ.get("TB_DOCKER_BUILD_CONC", "2"))   # the builder is an m5.xlarge
REBUILD = os.environ.get("TB_REBUILD") == "1"
HIDDEN = {"tests", "run-tests.sh", "run_tests.sh", "solution.sh", "solution.yaml"}

# Minimal, verified bit-rot fixes (task -> [(file, old, new)]); applied to the upload, tasks/ stays
# pristine. Not fixable, so left to fail and be reported: fix-git (its GitHub repo is gone) and
# leelachess0-pytorch-conversion (the lczero network file now 404s).
PATCHES = {
    "word2vec-from-scratch": [
        ("Dockerfile", "load_dataset('wikitext','wikitext-2-v1')", "load_dataset('Salesforce/wikitext','wikitext-2-v1')"),
    ],
}


def _ensure_standalone_python() -> None:
    """Fetch + prune the x86_64 standalone CPython into acs_ctx/py-x86_64 (once; gitignored).

    Uses the release's PRE-STRIPPED build: running binutils `strip` on this x86_64 build (even
    --strip-debug) corrupts its symbol versioning ("undefined symbol: , version")."""
    if (PY_DIR / "bin" / "python3").exists():
        return
    d.err("fetching standalone x86_64 CPython (python-build-standalone) -> acs_ctx/py-x86_64")
    d.sh(f"curl -sSL -o /tmp/acs_pbs_x86.tgz '{PBS_URL}' && mkdir -p {PY_DIR} && "
         f"tar -xzf /tmp/acs_pbs_x86.tgz -C {PY_DIR} --strip-components=1")
    lib = PY_DIR / "lib" / "python3.12"
    for rel in ("include", "share", "lib/pkgconfig", "lib/libpython3.12.a"):
        p = PY_DIR / rel
        shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
    for name in ("test", "idlelib", "tkinter", "turtledemo", "ensurepip", "lib2to3"):
        shutil.rmtree(lib / name, ignore_errors=True)
    for p in list(lib.glob("config-3.12-*")) + list(lib.glob("site-packages/pip*")):
        shutil.rmtree(p, ignore_errors=True)
    d.sh(f"{PY_DIR}/bin/python3 -c 'import http.server, tarfile, json, subprocess'")   # runs here (x86_64 host)


def _server_files() -> dict[str, bytes]:
    return {"acs_http_server.py": (ACS_CTX / "acs_http_server.py").read_bytes(),
            "sandbox_server_agent.py": (DEMO_DIR / "agentcore_slime_demo" / "sandbox_server_agent.py").read_bytes(),
            "WRAP_APPENDIX": WRAP_APPENDIX.encode()}


def _ctx_tag() -> str:
    h = hashlib.sha256((PLATFORM + PBS_URL).encode())          # a new Python build re-uploads the context
    for k, v in sorted(_server_files().items()):
        h.update(k.encode() + v)
    return "ctx-" + h.hexdigest()[:12]


def _ctx_tar(tag: str) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        tf.add(PY_DIR, arcname="py")
        for name, data in list(_server_files().items()) + [("TAG", tag.encode())]:
            ti = tarfile.TarInfo(name)
            ti.size, ti.mode = len(data), 0o644
            tf.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


def _task_tar(task: str) -> tuple[bytes, str, int]:
    """(tar.gz of the task dir minus its hidden oracle/solution, with patches applied; content hash; #patches)."""
    tdir = TASKS_DIR / task
    patches = {f: [(o, n) for ff, o, n in PATCHES.get(task, []) if ff == f] for f, _o, _n in PATCHES.get(task, [])}
    h, applied, buf = hashlib.sha256(), 0, io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for p in sorted(tdir.rglob("*")):
            rel = p.relative_to(tdir)
            if rel.parts[0] in HIDDEN or p.name == "Dockerfile.acs":
                continue
            if p.is_dir():                      # kept even when empty (a Dockerfile may COPY it)
                ti = tarfile.TarInfo(str(rel))
                ti.type, ti.mode = tarfile.DIRTYPE, 0o755
                tf.addfile(ti)
                continue
            if not p.is_file():                 # dangling symlink, socket, ...
                continue
            data = p.read_bytes()
            for old, new in patches.get(str(rel), []):
                if old.encode() not in data:
                    raise RuntimeError(f"patch for {task}/{rel} no longer applies (upstream changed)")
                data, applied = data.replace(old.encode(), new.encode()), applied + 1
            h.update(str(rel).encode() + b"\0" + data)
            ti = tarfile.TarInfo(str(rel))
            ti.size, ti.mode, ti.mtime = len(data), p.stat().st_mode & 0o777, 0
            tf.addfile(ti, io.BytesIO(data))
    return buf.getvalue(), h.hexdigest()[:10], applied


def _select() -> list[tuple[str, str]]:
    """[(task, reason)] for every task that belongs on the m5."""
    manifest = json.loads((TASKS_DIR / "manifest.json").read_text())["tasks"]
    reason = {t["task"]: "multi_container" for t in manifest if t.get("substrate") == "docker"}
    p = TASKS_DIR / "images.csv"
    if p.exists():
        with p.open() as f:
            for r in csv.DictReader(f):
                if r.get("status") in ("build_failed", "too_large"):
                    reason.setdefault(r["task"], f"microvm_{r['status']}")
    o = TASKS_DIR / "oracle.csv"
    if o.exists() and os.environ.get("TB_DOCKER_ORACLE_FALLBACK", "1") == "1":
        pool_path = TASKS_DIR / "route_by_image.json"
        pool = json.loads(pool_path.read_text()) if pool_path.exists() else {}
        with o.open() as f:
            for r in csv.DictReader(f):
                # oracle failed or errored; a task whose UNTOUCHED env already passes has no signal anywhere
                if r.get("usable") != "1" and r.get("null") != "1" and str(pool.get(r["task"], "")).startswith("arn:"):
                    reason.setdefault(r["task"], "microvm_oracle_fail")
    rp = TASKS_DIR / "route_by_image.json"
    for t, r in (json.loads(rp.read_text()) if rp.exists() else {}).items():
        if str(r).startswith("docker://"):
            reason.setdefault(t, "routed_m5")           # already on the m5 (e.g. a validated oracle fallback)
    for t in filter(None, os.environ.get("TB_DOCKER_TASKS", "").split(",")):
        reason.setdefault(t, "forced")
    only = {t for t in os.environ.get("TB_TASKS", "").split(",") if t}
    out = [(t, r) for t, r in sorted(reason.items()) if (TASKS_DIR / t).is_dir() and (not only or t in only)]
    limit = int(os.environ.get("TB_LIMIT", "0"))
    return out[:limit] if limit else out


def build_one(state: dict, task: str, why: str, ctx_tag: str) -> dict:
    row = {"task": task, "substrate": "docker", "reason": why, "status": "", "size_mb": "", "build_s": "",
           "patched": 0, "tag": "", "error": ""}
    try:
        data, digest, row["patched"] = _task_tar(task)
        tag = f"{ctx_tag[4:12]}-{digest}"
        row["tag"] = tag
        rec = m5_host.call(state, {"op": "build_status", "task": task})
        if REBUILD or rec.get("status") != "built" or rec.get("tag") != tag:
            m5_host.put_tar(state, data, f"task:{task}")
            m5_host.call(state, {"op": "build", "task": task, "tag": tag, "force": REBUILD})
            end = time.time() + 7200
            while True:
                time.sleep(15)
                rec = m5_host.call(state, {"op": "build_status", "task": task})
                if rec.get("status") != "running" or time.time() > end:
                    break
        row.update(status=rec.get("status", "unknown"), size_mb=rec.get("size_mb", ""),
                   build_s=rec.get("build_s", ""), error=(rec.get("error") or "")[-300:])
        if row["status"] == "built" and state.get("ecr_prefix") and os.environ.get("TB_M5_PUBLISH", "1") == "1":
            publish(state, task, rec, data)
            row["ecr"] = 1
    except Exception as e:  # noqa: BLE001 — one bad task must not sink the rest
        row.update(status="build_failed", error=f"{type(e).__name__}: {e}"[-300:])
    d.err(f"[docker-build] {task} ({why}): {row['status']} size_mb={row['size_mb']} build_s={row['build_s']} "
          f"{row['error'][:140]}")
    return row


RECORDS = TASKS_DIR / "docker_records"


def publish(state: dict, task: str, rec: dict, task_tar: bytes) -> None:
    """Per-episode mode: the builder pushes every image of the task to ECR (<prefix>-tbhdk-<task>-<svc>),
    and this box keeps what a fresh instance needs: the ECR build record + the task dir tarball (no
    tests/solution) in tasks/docker_records/."""
    RECORDS.mkdir(exist_ok=True)
    rj = RECORDS / f"{task}.json"
    old = json.loads(rj.read_text()) if rj.exists() else {}
    if REBUILD or old.get("tag") != rec.get("tag") or not old.get("ecr"):
        rep = m5_host.call(state, {"op": "push", "task": task, "repo_prefix": state["ecr_prefix"]}, timeout=7200)
        rec = dict(rec, ecr=rep["ecr"])
        rj.write_text(json.dumps(rec, indent=1))
        d.err(f"[docker-publish] {task}: {len(rep['ecr'])} image(s) -> ECR in {rep.get('push_s')}s")
    (RECORDS / f"{task}.tar.gz").write_bytes(task_tar)


def main() -> int:
    state = m5_host.load_state()
    h = m5_host.call(state, {"op": "health"}, timeout=30)
    d.err(f"m5 host {state['instance_id']} {state['ip']}: docker {h.get('docker')}, {h.get('disk_free_gb')} GB free")
    tasks = _select()
    if not tasks:
        d.die("no tasks for the m5 (fetch the corpus, run the microVM build first, or set TB_DOCKER_TASKS)")
    _ensure_standalone_python()
    ctx_tag = _ctx_tag()
    if h.get("ctx_tag") != ctx_tag:
        d.err(f"uploading the wrapper build context ({ctx_tag}) to the m5")
        m5_host.put_tar(state, _ctx_tar(ctx_tag), "ctx")

    d.err(f"building {len(tasks)} task(s) on the m5 ({PLATFORM}, conc={BUILD_CONC})")
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=BUILD_CONC) as ex:
        rows = list(ex.map(lambda tw: build_one(state, tw[0], tw[1], ctx_tag), tasks))

    pool_path = TASKS_DIR / "route_by_image.json"
    pool = json.loads(pool_path.read_text()) if pool_path.exists() else {}
    fallback = TASKS_DIR / "docker_fallback.json"        # microvm_oracle_fail tasks -> their microVM ARN
    keep_arn = json.loads(fallback.read_text()) if fallback.exists() else {}
    for r in rows:
        if r["status"] == "built":
            if r["reason"] == "microvm_oracle_fail" and str(pool.get(r["task"], "")).startswith("arn:"):
                keep_arn[r["task"]] = pool[r["task"]]   # validate_oracle.py may route it back
            pool[r["task"]] = f"docker://{state['ip']}:{state['port']}/{r['task']}"
        elif str(pool.get(r["task"], "")).startswith("docker://"):
            if r["task"] in keep_arn:
                pool[r["task"]] = keep_arn.pop(r["task"])
            else:
                pool.pop(r["task"])
    fallback.write_text(json.dumps(dict(sorted(keep_arn.items())), indent=2))
    pool_path.write_text(json.dumps(dict(sorted(pool.items())), indent=2))

    csv_path = TASKS_DIR / "images_docker.csv"
    prev = {}
    if csv_path.exists():
        with csv_path.open() as f:
            prev = {r["task"]: r for r in csv.DictReader(f)}
    prev.update({r["task"]: r for r in rows})
    fields = ["task", "substrate", "reason", "status", "size_mb", "build_s", "patched", "tag", "ecr", "error"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(prev.values(), key=lambda r: r["task"]))

    by: dict[str, int] = {}
    for r in rows:
        by[r["status"]] = by.get(r["status"], 0) + 1
    n_dc = sum(1 for v in pool.values() if str(v).startswith("docker://"))
    print(f"\n# {len(rows)} m5 tasks in {time.perf_counter() - t0:.0f}s: {by}")
    print(f"# route map: {len(pool) - n_dc} microVM Runtimes + {n_dc} m5 docker tasks -> {pool_path}")
    print(f"# per-task detail -> {csv_path}")
    return 0 if by.get("built") else 1


if __name__ == "__main__":
    sys.exit(main())
