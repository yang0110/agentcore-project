#!/usr/bin/env python3
"""Host agent on the m5 Docker host: one `docker compose` project per Terminal-Bench episode.

demo_terminalbench_kernelbench runs the tasks AgentCore microVMs can't (multi-container compose, x86-only,
images over 2 GB) on a plain EC2 instance with Docker. The trainer never runs docker itself: it calls
this agent over the VPC (SG->SG, shared-secret header), and then talks to the episode's `client`
container directly, which runs the SAME stdlib server as the microVM images (acs_http_server.py on
:8080, published on a random host port). So everything above the sandbox (in-session agent loop,
hidden-oracle grader, reward) is the code the microVM pool already uses.

    POST /op  {"op": ...}  header X-ACS-Token: <token>

    health                          docker up, live projects, free disk, built tasks
    put_tar {dest, key, b64, more}  upload a tar.gz in chunks; dest = "ctx" (the /opt/acs build context)
                                    or "task:<name>" (the task dir, WITHOUT tests/ or the solution)
    build {task, tag, no_cache}     start building the task's images in the background (no_cache: fresh
                                    layers + base pull, e.g. a stale `apt-get update` layer)
    build_status {task}             {status: running|built|build_failed, ...}
    up {task, episode}              fresh project ep_<episode>: copy the task dir, rewrite the compose
                                    file (below), `up -d`, wait for the client's server -> {port, ...}
    down {episode}                  `down -v --remove-orphans` + delete the episode dir
    list / gc {keep}                live projects / tear down every ep_* project not in keep
    push {task, repo_prefix}        (builder) push every image of a built task — built AND sidecar — to
                                    ECR <repo_prefix>-<task>-<service>, record the URIs (per-episode mode)
    set_record {task, rec}          (episode instance) install a build record whose images live in ECR
    up {task, episode, gpu, cpus, mem, shm}
                                    (g6 GPU instance) as `up`, with the client pinned to ONE host GPU
                                    (compose device reservation, device_ids=[gpu]) and per-slot limits
    reset_check                     (warm instance) after `down`: verify NOTHING of the episode is left —
                                    no containers, no volumes, no non-default networks, empty episode dir —
                                    evicting cached images if free disk < ACS_MIN_FREE_GB -> {clean, problems}
    reset_check {episode, gpu}      (g6 GPU instance, other slots still busy) the same check SCOPED to one
                                    project (compose labels) + no process left on that GPU
    logs {episode, tail}            `docker compose logs` of one episode (debugging)
    self_update {b64}               replace this file and exit (systemd restarts it)

Per-episode compose rewrite (the upstream compose files assume ONE episode per host):
  * every service with a fixed container_name gets <project>-<service> instead, and keeps the old
    name as a network alias, so hostnames like `master_service` still resolve inside the project;
  * sidecar `ports:` are dropped (nothing outside the project needs them, and publishing the
    deliberately vulnerable services of the security tasks would expose them to the VPC); the client
    publishes only :8080 on a random host port;
  * the client runs acs_http_server.py instead of `sleep infinity` (a prelude such as the
    `echo ... >> /etc/hosts &&` of two security tasks is kept);
  * built services use the images prebuilt by `build` (no build at episode time), bind mounts point
    into the episode's own copy of the task dir, and anything that could reach the host (privileged,
    host network/pid, devices, docker.sock, bind mounts outside the episode dir) is refused.

Stdlib only, Python >= 3.9 (Amazon Linux 2023's system python3). Config via env (systemd unit):
ACS_ROOT=/srv/acs, ACS_PORT=18300, ACS_BIND=0.0.0.0, ACS_MAX_PROJECTS=16, ACS_CLIENT_MEM=8g, ACS_CLIENT_CPUS=4,
ACS_TOKEN_FILE=$ACS_ROOT/token.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.environ.get("ACS_ROOT", "/srv/acs")
PORT = int(os.environ.get("ACS_PORT", "18300"))
BIND = os.environ.get("ACS_BIND", "0.0.0.0")
MAX_PROJECTS = int(os.environ.get("ACS_MAX_PROJECTS", "16"))
CLIENT_MEM = os.environ.get("ACS_CLIENT_MEM", "8g")
CLIENT_CPUS = os.environ.get("ACS_CLIENT_CPUS", "4")
TOKEN = open(os.environ.get("ACS_TOKEN_FILE", os.path.join(ROOT, "token"))).read().strip()
CTX, TASKS, BUILT, EPS = (os.path.join(ROOT, d) for d in ("ctx", "tasks", "built", "ep"))
SERVER_PORT = 8080                  # records built before the launcher (and the KernelBench GPU image)
SERVER_CMD = "exec /opt/acs/py/bin/python3 /opt/acs/acs_http_server.py"
# Records with `launcher` (built since 2026-10-13): the image's own start-up runs, as its own USER, and our
# server moves off :8080, which nginx / pypi-server / mlflow tasks need themselves (acs_m5_launch.py).
LAUNCH_PORT = 18080
LAUNCH_CMD = "exec /opt/acs/py/bin/python3 /opt/acs/acs_m5_launch.py"
_IDLE_STARTS = {"sleep infinity", "sh -c sleep infinity", "bash -c sleep infinity", "/bin/sh -c sleep infinity",
                "/bin/bash -c sleep infinity", "bash", "/bin/bash", "sh", "/bin/sh", ""}
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,80}$")
AGENT_VERSION = "2026-10-13.3"      # health reports it; the trainer self_updates an older agent (baked AMI)
MIN_FREE_GB = float(os.environ.get("ACS_MIN_FREE_GB", "8"))
# Dead-man switch for episode instances (0 = off, the builder): no authenticated trainer call for this
# long -> `shutdown -h now`, and the instance (launched with InstanceInitiatedShutdownBehavior=terminate)
# is terminated. Covers a trainer box that died or a run killed before it could terminate its instances.
IDLE_SHUTDOWN_S = float(os.environ.get("ACS_IDLE_SHUTDOWN_S", "0"))
_last_call = [time.time()]

_lock = threading.Lock()
_live: dict = {}          # project -> {"task", "since"}
_builds: dict = {}        # task -> thread


def _sh(args: list, timeout: float = 600, cwd: str | None = None) -> tuple:
    p = subprocess.run(args, capture_output=True, text=True, timeout=timeout, cwd=cwd)
    return p.returncode, p.stdout, p.stderr


def _name(v: str, what: str) -> str:
    if not isinstance(v, str) or not NAME_RE.match(v):
        raise ValueError(f"bad {what}: {v!r}")
    return v


def _project(episode: str) -> str:
    return "ep_" + _name(episode.lower(), "episode").replace(".", "_")


def _record(task: str) -> dict:
    p = os.path.join(BUILT, f"{task}.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def _save_record(task: str, rec: dict) -> None:
    os.makedirs(BUILT, exist_ok=True)
    tmp = os.path.join(BUILT, f".{task}.json")
    with open(tmp, "w") as f:
        json.dump(rec, f, indent=1)
    os.replace(tmp, os.path.join(BUILT, f"{task}.json"))


def _env(project: str, task: str, ep_dir: str) -> dict:
    """The T_BENCH_* variables the upstream compose files interpolate (Terminal-Bench's harness sets them)."""
    env = dict(os.environ)
    env.update({
        "T_BENCH_TASK_DOCKER_CLIENT_IMAGE_NAME": f"acs-tbh/{task}-client:latest",
        "T_BENCH_TASK_DOCKER_CLIENT_CONTAINER_NAME": f"{project}-client",
        "T_BENCH_TASK_DOCKER_NAME_PREFIX": project,
        "T_BENCH_TEST_DIR": "/tests",
        "T_BENCH_TASK_LOGS_PATH": os.path.join(ep_dir, "logs"),
        "T_BENCH_CONTAINER_LOGS_PATH": "/logs",
        "T_BENCH_TASK_AGENT_LOGS_PATH": os.path.join(ep_dir, "agent-logs"),
        "T_BENCH_CONTAINER_AGENT_LOGS_PATH": "/agent-logs",
    })
    return env


def _resolved(compose_dir: str, project: str, env: dict) -> dict:
    """The task's compose model, interpolated and with absolute paths (`docker compose config`)."""
    cf = next((os.path.join(compose_dir, n) for n in ("docker-compose.yaml", "docker-compose.yml")
               if os.path.exists(os.path.join(compose_dir, n))), None)
    if cf is None:
        raise RuntimeError(f"no docker-compose.yaml in {compose_dir}")
    p = subprocess.run(["docker", "compose", "-p", project, "-f", cf, "config", "--format", "json"],
                       capture_output=True, text=True, env=env, cwd=compose_dir, timeout=120)
    if p.returncode != 0:
        raise RuntimeError(f"compose config failed: {p.stderr[-400:]}")
    model = json.loads(p.stdout)
    model.pop("name", None)
    return model


def _client_service(model: dict, project: str) -> str:
    svcs = model.get("services") or {}
    for name, s in svcs.items():
        if s.get("container_name") == f"{project}-client":
            return name
    if "client" in svcs:
        return "client"
    raise RuntimeError(f"no client service among {sorted(svcs)}")


def _image_tag(task: str, svc: str, tag: str) -> str:
    return f"acs-tbh/{task}-{re.sub(r'[^a-z0-9_.-]', '-', svc.lower())}:{tag}"


def _client_command(cmd) -> list:
    """The client's command with `sleep infinity` replaced by our server (prelude kept)."""
    script = None
    if isinstance(cmd, list) and len(cmd) == 3 and cmd[0] in ("sh", "bash", "/bin/sh", "/bin/bash") and cmd[1] == "-c":
        script = cmd[2]
    elif isinstance(cmd, str):
        script = cmd
    if script and "sleep infinity" in script:
        return ["sh", "-c", script.replace("sleep infinity", SERVER_CMD)]
    return ["sh", "-c", SERVER_CMD]


def _check_safe(model: dict, allowed_root: str) -> None:
    for name, s in (model.get("services") or {}).items():
        bad = [k for k in ("privileged", "cap_add", "devices", "security_opt") if s.get(k)]
        if s.get("network_mode", "").startswith(("host", "container:", "service:")):
            bad.append(f"network_mode={s['network_mode']}")
        if s.get("pid") == "host" or s.get("ipc") == "host":
            bad.append("host pid/ipc")
        for v in s.get("volumes") or []:
            src = v.get("source", "") if isinstance(v, dict) else ""
            if isinstance(v, dict) and v.get("type") == "bind":
                if "docker.sock" in src or not os.path.realpath(src).startswith(os.path.realpath(allowed_root) + os.sep):
                    bad.append(f"bind {src}")
        if bad:
            raise RuntimeError(f"refused: service {name!r} asks for {bad}")


def _ecr_login(uri: str) -> None:
    """docker login to the ECR registry of `uri` with this instance's role (host only: containers can't
    reach instance metadata)."""
    registry = uri.split("/", 1)[0]
    region = registry.split(".")[3]
    p = subprocess.run(f"aws ecr get-login-password --region {region} | docker login --username AWS "
                       f"--password-stdin {registry}", shell=True, capture_output=True, text=True, timeout=120)
    if p.returncode != 0:
        raise RuntimeError(f"ecr login {registry}: {(p.stdout + p.stderr)[-300:]}")


def _rewrite(model: dict, task: str, project: str, rec: dict, ep_dir: str, slot: dict | None = None) -> tuple:
    """(model, client service): unique names, no sidecar ports, our server in the client, prebuilt images.
    `slot` (g6 GPU instance): {gpu, cpus, mem, shm} — the client gets that ONE GPU and those limits."""
    svcs = model.get("services") or {}
    ecr = rec.get("ecr") or {}          # per-episode instance: EVERY image comes from ECR
    client = _client_service(model, project)
    for name, s in svcs.items():
        old = s.get("container_name")
        s["container_name"] = f"{project}-{name}"
        nets = s.get("networks") or {"default": None}
        if old and old != s["container_name"]:
            for net in list(nets):
                cfg = nets[net] or {}
                cfg["aliases"] = sorted(set((cfg.get("aliases") or []) + [old]))
                nets[net] = cfg
        s["networks"] = nets
        if name in ecr:
            s.pop("build", None)
            s["image"] = ecr[name]
            s["pull_policy"] = "missing"
        elif "build" in s:
            s.pop("build")
            s["image"] = rec["images"][name]
            s["pull_policy"] = "never"
        if name == client and rec.get("launcher"):
            orig = rec.get("orig") or {}
            cmd = s.get("command")
            if isinstance(cmd, str):
                cmd = ["sh", "-c", cmd]
            # compose overrides the image only when it sets one (`compose config` emits "entrypoint": null)
            ep = s["entrypoint"] if s.get("entrypoint") is not None else orig.get("entrypoint")
            if isinstance(ep, str):
                ep = ["sh", "-c", ep]
            start = list(ep or []) + list(cmd if cmd else orig.get("cmd") or [])
            env = s.get("environment") or {}
            if isinstance(env, list):
                env = dict(e.split("=", 1) if "=" in e else (e, "") for e in env)
            env.update(ACS_SERVER_PORT=str(LAUNCH_PORT), ACS_EXEC_INTERACTIVE="1")
            if " ".join(start) not in _IDLE_STARTS:
                env["ACS_ORIG_START"] = json.dumps(start)
            user = orig.get("user", "").split(":", 1)[0]
            if user not in ("", "root", "0") and not s.get("user"):
                env["ACS_EXEC_USER"] = user
            s["environment"] = env
            s["entrypoint"] = []
            s["command"] = ["sh", "-c", LAUNCH_CMD]
            s["init"] = True                # tini reaps what the image's start-up orphans (qemu -daemonize)
            s["ports"] = [{"target": int(rec.get("server_port", LAUNCH_PORT)), "protocol": "tcp", "mode": "ingress"}]
        elif name == client:
            s["ports"] = [{"target": SERVER_PORT, "protocol": "tcp", "mode": "ingress"}]
            s["command"] = _client_command(s.get("command"))
        if name == client:
            s.setdefault("mem_limit", CLIENT_MEM)
            s.setdefault("cpus", float(CLIENT_CPUS))
            s.setdefault("pids_limit", 8192)
            if slot:
                if slot.get("cpus"):
                    s["cpus"] = float(slot["cpus"])
                if slot.get("mem"):
                    s["mem_limit"] = str(slot["mem"])
                if slot.get("shm"):
                    s["shm_size"] = str(slot["shm"])
                if slot.get("gpu") is not None:
                    s.setdefault("deploy", {}).setdefault("resources", {}).setdefault("reservations", {})["devices"] = [
                        {"driver": "nvidia", "device_ids": [str(int(slot["gpu"]))], "capabilities": ["gpu"]}]
        else:
            s.pop("ports", None)
    if "default" not in (model.get("networks") or {}) and any("default" in (s.get("networks") or {}) for s in svcs.values()):
        model.setdefault("networks", {})["default"] = {}
    _check_safe(model, ep_dir)
    return model, client


# ----------------------------------------------------------------------------- build
def _build(task: str, tag: str, no_cache: bool = False) -> None:
    tdir = os.path.join(TASKS, task)
    log_path = os.path.join(BUILT, f"{task}.log")
    rec = {"task": task, "tag": tag, "status": "running", "started": time.time()}
    _save_record(task, rec)
    t0 = time.time()
    try:
        project = f"build_{task}".replace(".", "_")
        model = _resolved(tdir, project, _env(project, task, os.path.join(EPS, "_build")))
        client = _client_service(model, project)
        images = {}
        for name, s in (model.get("services") or {}).items():
            if "build" in s:
                images[name] = _image_tag(task, name, tag)
                s["image"] = images[name]
        if client not in images:
            raise RuntimeError(f"client service {client!r} has no build section (an image-only client can't be wrapped)")
        # The client is built AS SHIPPED first (its own Dockerfile, untouched), so its ENTRYPOINT / CMD /
        # USER can be read back; the wrapper is then a layer FROM it (build cache: no second build).
        orig = _image_tag(task, client + "-orig", tag)
        model["services"][client]["image"] = orig
        bf = os.path.join(BUILT, f"{task}.build.json")
        with open(bf, "w") as f:
            json.dump(model, f)
        with open(log_path, "w") as lf:
            for args in (["docker", "compose", "-p", project, "-f", bf, "pull", "--ignore-buildable"],
                         ["docker", "compose", "-p", project, "-f", bf, "build"]
                         + (["--no-cache", "--pull"] if no_cache else [])):   # a stale cached `apt-get update` layer
                lf.write(f"$ {' '.join(args)}\n")
                lf.flush()
                p = subprocess.run(args, stdout=lf, stderr=subprocess.STDOUT, timeout=7200, cwd=tdir)
                if p.returncode != 0:
                    raise RuntimeError(f"compose {args[6]} exit {p.returncode}")
            code, out, err = _sh(["docker", "image", "inspect", "-f", "{{json .Config}}", orig], timeout=60)
            if code != 0:
                raise RuntimeError(f"inspect {orig}: {err[-300:]}")
            cfg = json.loads(out)
            with open(os.path.join(CTX, "WRAP_APPENDIX")) as f:
                appendix = f.read().replace("{env_lines}", "")
            wrap_dir = os.path.join(BUILT, f"{task}.wrap")
            shutil.rmtree(wrap_dir, ignore_errors=True)
            os.makedirs(wrap_dir)
            with open(os.path.join(wrap_dir, "Dockerfile.acs"), "w") as f:
                f.write(f"FROM {orig}\n\n{appendix.rstrip()}\nCOPY --from=acs acs_m5_launch.py /opt/acs/\n")
            args = ["docker", "build", "--build-context", f"acs={CTX}", "-f", os.path.join(wrap_dir, "Dockerfile.acs"),
                    "-t", images[client], wrap_dir]
            lf.write(f"$ {' '.join(args)}\n")
            lf.flush()
            p = subprocess.run(args, stdout=lf, stderr=subprocess.STDOUT, timeout=3600, cwd=wrap_dir)
            if p.returncode != 0:
                raise RuntimeError(f"wrapper build exit {p.returncode}")
        size = 0
        for img in images.values():
            code, out, _ = _sh(["docker", "image", "inspect", "-f", "{{.Size}}", img], timeout=60)
            size += int(out.strip() or 0) if code == 0 else 0
        rec.update(status="built", images=images, client=client, size_mb=round(size / 2**20, 1),
                   build_s=round(time.time() - t0, 1), launcher=1, server_port=LAUNCH_PORT,
                   orig={"entrypoint": cfg.get("Entrypoint") or [], "cmd": cfg.get("Cmd") or [],
                         "user": cfg.get("User") or ""})
    except Exception as e:  # noqa: BLE001 — recorded, reported by build_status
        tail = ""
        if os.path.exists(log_path):
            tail = " | ".join(open(log_path, errors="replace").read().strip().splitlines()[-3:])
        rec.update(status="build_failed", error=f"{type(e).__name__}: {e} :: {tail}"[-600:],
                   build_s=round(time.time() - t0, 1))
    _save_record(task, rec)


# ----------------------------------------------------------------------------- episodes
def _wait_server(port: int, deadline_s: float) -> None:
    end = time.time() + deadline_s
    while True:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/ping", timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:  # noqa: BLE001 — not up yet
            pass
        if time.time() > end:
            raise RuntimeError(f"client server on :{port} not up after {deadline_s:.0f}s")
        time.sleep(0.5)


def _up(task: str, episode: str, slot: dict | None = None) -> dict:
    project = _project(episode)
    rec = _record(task)
    if rec.get("status") != "built":
        raise RuntimeError(f"task {task!r} is not built here (status={rec.get('status')!r})")
    with _lock:
        if project in _live:
            raise RuntimeError(f"{project} already exists")
        if len(_live) >= MAX_PROJECTS:
            return {"busy": True, "live": len(_live), "max_projects": MAX_PROJECTS}
        _live[project] = {"task": task, "since": time.time()}
    t0 = time.time()
    ep_dir = os.path.join(EPS, project)
    try:
        os.makedirs(os.path.join(ep_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(ep_dir, "agent-logs"), exist_ok=True)
        # The episode's OWN copy of the task dir: bind mounts (./:/app/data, .:/home/user) must not
        # leak one episode's edits into the next, nor into the build copy.
        code, _, err = _sh(["cp", "-a", "--reflink=auto", os.path.join(TASKS, task), os.path.join(ep_dir, "task")])
        if code != 0:
            raise RuntimeError(f"copy task dir: {err[-300:]}")
        env = _env(project, task, ep_dir)
        model, client = _rewrite(_resolved(os.path.join(ep_dir, "task"), project, env), task, project, rec, ep_dir, slot)
        cf = os.path.join(ep_dir, "compose.json")
        with open(cf, "w") as f:
            json.dump(model, f)
        missing = [u for u in (rec.get("ecr") or {}).values()
                   if _sh(["docker", "image", "inspect", u], timeout=60)[0] != 0]
        if missing:                     # a warm instance keeps a task's images: pull only what it lacks
            _ecr_login(missing[0])
            for u in missing:
                code, out, err = _sh(["docker", "pull", "--quiet", u], timeout=1800)
                if code != 0:
                    raise RuntimeError(f"pull {u}: {(out + err)[-400:]}")
        pulled_ms = round((time.time() - t0) * 1000.0, 1)
        code, out, err = _sh(["docker", "compose", "-p", project, "-f", cf, "up", "-d", "--pull", "missing",
                              "--no-build", "--quiet-pull"], timeout=900, cwd=ep_dir)
        if code != 0:
            raise RuntimeError(f"compose up exit {code}: {(out + err)[-600:]}")
        code, out, err = _sh(["docker", "compose", "-p", project, "-f", cf, "port", client,
                              str(rec.get("server_port", SERVER_PORT))], timeout=60)
        if code != 0 or ":" not in out:
            raise RuntimeError(f"compose port: {(out + err)[-300:]}")
        port = int(out.strip().rsplit(":", 1)[1])
        up_ms = round((time.time() - t0) * 1000.0, 1)
        _wait_server(port, float(os.environ.get("ACS_SERVER_WAIT_S", "180")))
        with _lock:
            _live[project].update(port=port)
        return {"ok": True, "project": project, "port": port, "client": client, "pull_ms": pulled_ms,
                "up_ms": up_ms, "ready_ms": round((time.time() - t0) * 1000.0, 1)}
    except BaseException:
        _down(episode)
        raise


def _down(episode: str) -> dict:
    project = _project(episode)
    ep_dir = os.path.join(EPS, project)
    cf = os.path.join(ep_dir, "compose.json")
    args = ["docker", "compose", "-p", project] + (["-f", cf] if os.path.exists(cf) else []) + \
           ["down", "-v", "--remove-orphans", "--timeout", "5"]
    code, out, err = _sh(args, timeout=300)
    shutil.rmtree(ep_dir, ignore_errors=True)
    with _lock:
        _live.pop(project, None)
    return {"ok": code == 0, "project": project, **({"error": (out + err)[-300:]} if code else {})}


_GPUS: list = []


def _gpu_count() -> int:
    """Host GPUs (nvidia-smi -L), counted once; 0 on the m5."""
    if not _GPUS:
        n = 0
        if shutil.which("nvidia-smi"):
            code, out, _ = _sh(["nvidia-smi", "-L"], timeout=60)
            n = sum(ln.startswith("GPU ") for ln in out.splitlines()) if code == 0 else 0
        _GPUS.append(n)
    return _GPUS[0]


def _reset_check(episode: str | None = None, gpu=None) -> dict:
    """A warm instance's post-episode reset, VERIFIED: the episode only ever touched its own compose
    project (containers can't reach the host), so after `down -v` nothing of it may remain.

    With `episode` (a g6 instance whose other GPU slots may still be running episodes) the check is
    SCOPED to that project — its compose-labelled containers / volumes / networks and its episode dir —
    plus, with `gpu`, no compute process left on that GPU."""
    problems = []
    scope = ["--filter", f"label=com.docker.compose.project={_project(episode)}"] if episode else []
    with _lock:
        live = [p for p in _live if not episode or p == _project(episode)]
    if live:
        problems.append(f"live projects {sorted(live)}")
    for args, what in ((["docker", "ps", "-aq"] + scope, "containers"),
                       (["docker", "volume", "ls", "-q"] + scope, "volumes")):
        code, out, _ = _sh(args, timeout=60)
        if code != 0 or out.strip():
            problems.append(f"{what} left: {out.split()[:5]}")
    code, out, _ = _sh(["docker", "network", "ls", "--format", "{{.Name}}"] + scope, timeout=60)
    extra = sorted(set(out.split()) - {"bridge", "host", "none"})
    if code != 0 or extra:
        problems.append(f"networks left: {extra[:5]}")
    left = os.listdir(EPS) if os.path.isdir(EPS) else []
    if episode:
        left = [d for d in left if d == _project(episode)]
    if left:
        problems.append(f"episode dirs left: {left[:5]}")
    if gpu is not None:
        code, out, _ = _sh(["nvidia-smi", "-i", str(int(gpu)), "--query-compute-apps=pid",
                            "--format=csv,noheader"], timeout=60)
        if code != 0 or out.strip():
            problems.append(f"gpu {gpu} processes left: {out.split()[:5] if code == 0 else 'nvidia-smi failed'}")
    st = os.statvfs(ROOT)
    free = st.f_bavail * st.f_frsize / 2**30
    if free < MIN_FREE_GB:              # image cache, not episode state: evict it and re-measure
        _sh(["docker", "image", "prune", "-af"], timeout=600)
        st = os.statvfs(ROOT)
        free = st.f_bavail * st.f_frsize / 2**30
        if free < MIN_FREE_GB:
            problems.append(f"disk free {free:.1f} GB < {MIN_FREE_GB} after eviction")
    return {"ok": True, "clean": not problems, "problems": problems, "disk_free_gb": round(free, 1)}


def _docker_projects() -> list:
    code, out, _ = _sh(["docker", "compose", "ls", "-a", "--format", "json"], timeout=60)
    try:
        return [p["Name"] for p in json.loads(out or "[]") if p["Name"].startswith("ep_")] if code == 0 else []
    except ValueError:
        return []


def _gc(keep: list) -> dict:
    keep_p = {_project(k) for k in keep}
    gone = []
    for project in set(_docker_projects()) | set(os.listdir(EPS) if os.path.isdir(EPS) else []):
        if project.startswith("ep_") and project not in keep_p:
            _down(project[3:])
            gone.append(project)
    return {"ok": True, "removed": sorted(gone)}


def _push(task: str, repo_prefix: str) -> dict:
    """Builder: push every image the task's compose project uses to ECR (one repo per service, which the
    caller has created), so a per-episode instance pulls ONLY from ECR — never Docker Hub, whose
    anonymous pull limit a fleet behind one NAT IP would exhaust."""
    rec = _record(task)
    if rec.get("status") != "built":
        raise RuntimeError(f"task {task!r} is not built here")
    model = json.load(open(os.path.join(BUILT, f"{task}.build.json")))
    ecr, t0 = {}, time.time()
    _ecr_login(f"{repo_prefix}-x")
    for name, s in (model.get("services") or {}).items():
        src = rec["images"].get(name) or s.get("image")
        if not src:
            raise RuntimeError(f"service {name!r} has no image")
        repo = f"{repo_prefix}-{task}-{re.sub(r'[^a-z0-9_.-]', '-', name.lower())}"
        dst = f"{repo}:{rec['tag']}"
        region = repo.split(".")[3]
        p = subprocess.run(["aws", "ecr", "create-repository", "--region", region,
                            "--repository-name", repo.split("/", 1)[1]], capture_output=True, text=True, timeout=60)
        if p.returncode != 0 and "RepositoryAlreadyExists" not in p.stderr:
            raise RuntimeError(f"create-repository {repo}: {p.stderr[-300:]}")
        for args in (["docker", "tag", src, dst], ["docker", "push", "--quiet", dst]):
            code, out, err = _sh(args, timeout=3600)
            if code != 0:
                raise RuntimeError(f"{' '.join(args[:2])} {dst}: {(out + err)[-300:]}")
        ecr[name] = dst
    rec.update(ecr=ecr, push_s=round(time.time() - t0, 1))
    _save_record(task, rec)
    return {"ok": True, "ecr": ecr, "push_s": rec["push_s"]}


# ----------------------------------------------------------------------------- uploads
def _put_tar(req: dict) -> dict:
    dest = req["dest"]
    if dest == "ctx":
        target = CTX
    elif dest.startswith("task:"):
        target = os.path.join(TASKS, _name(dest[5:], "task"))
    else:
        raise ValueError(f"bad dest {dest!r}")
    stage = os.path.join(ROOT, f".put_{_name(req.get('key', 'tar'), 'key')}")
    with open(stage, "ab") as f:
        f.write(base64.b64decode(req.get("b64", "")))
    if req.get("more"):
        return {"ok": True, "staged": os.path.getsize(stage)}
    new = target + ".new"
    shutil.rmtree(new, ignore_errors=True)
    os.makedirs(new)
    with open(stage, "rb") as f, tarfile.open(fileobj=io.BytesIO(f.read()), mode="r:gz") as tf:
        for m in tf.getmembers():
            if m.name.startswith("/") or ".." in m.name.split("/"):
                raise ValueError(f"unsafe member {m.name!r}")
        tf.extractall(new)
    os.remove(stage)
    shutil.rmtree(target, ignore_errors=True)
    os.replace(new, target)
    return {"ok": True}


def _health() -> dict:
    code, out, _ = _sh(["docker", "info", "--format", "{{.ServerVersion}}"], timeout=30)
    st = os.statvfs(ROOT)
    built = sorted(f[:-5] for f in os.listdir(BUILT) if f.endswith(".json") and not f.endswith(".build.json")
                   and _record(f[:-5]).get("status") == "built") if os.path.isdir(BUILT) else []
    with _lock:
        live = {p: dict(v) for p, v in _live.items()}
    return {"ok": code == 0, "version": AGENT_VERSION, "docker": out.strip(), "live": len(live), "projects": live,
            "gpus": _gpu_count(),
            "max_projects": MAX_PROJECTS, "disk_free_gb": round(st.f_bavail * st.f_frsize / 2**30, 1),
            "built": built, "ctx": os.path.exists(os.path.join(CTX, "py", "bin", "python3")),
            "ctx_tag": open(os.path.join(CTX, "TAG")).read().strip() if os.path.exists(os.path.join(CTX, "TAG")) else ""}


def handle(req: dict) -> dict:
    op = req.get("op")
    if op == "health":
        return _health()
    if op == "set_idle_shutdown":          # the trainer arms the switch on the instances it launches
        global IDLE_SHUTDOWN_S
        IDLE_SHUTDOWN_S = float(req.get("seconds", 0))
        return {"ok": True, "idle_shutdown_s": IDLE_SHUTDOWN_S}
    if op == "put_tar":
        return _put_tar(req)
    if op == "build":
        task, tag = _name(req["task"], "task"), _name(req["tag"], "tag")
        with _lock:
            th = _builds.get(task)
            if th is not None and th.is_alive():
                return {"ok": True, "status": "running"}
            rec = _record(task)
            if rec.get("status") == "built" and rec.get("tag") == tag and not req.get("force"):
                return {"ok": True, "status": "built", "cached": True}
            th = threading.Thread(target=_build, args=(task, tag, bool(req.get("no_cache"))), daemon=True)
            _builds[task] = th
            th.start()
        return {"ok": True, "status": "running"}
    if op == "build_status":
        return _record(_name(req["task"], "task")) or {"status": "unknown"}
    if op == "up":
        slot = {k: req[k] for k in ("gpu", "cpus", "mem", "shm") if req.get(k) is not None} or None
        return _up(_name(req["task"], "task"), req["episode"], slot)
    if op == "push":
        return _push(_name(req["task"], "task"), req["repo_prefix"])
    if op == "set_record":
        rec = dict(req["rec"])
        if rec.get("status") != "built" or not rec.get("ecr"):
            raise ValueError("set_record needs a built record with ECR images")
        _save_record(_name(req["task"], "task"), rec)
        return {"ok": True}
    if op == "down":
        return _down(req["episode"])
    if op == "reset_check":
        return _reset_check(req.get("episode"), req.get("gpu"))
    if op == "list":
        with _lock:
            return {"ok": True, "live": sorted(_live), "docker": _docker_projects()}
    if op == "gc":
        return _gc(list(req.get("keep") or []))
    if op == "logs":
        project = _project(req["episode"])
        cf = os.path.join(EPS, project, "compose.json")
        code, out, err = _sh(["docker", "compose", "-p", project, "-f", cf, "logs", "--no-color",
                              "--tail", str(int(req.get("tail", 200)))], timeout=60)
        return {"ok": code == 0, "logs": (out + err)[-20000:]}
    if op == "self_update":
        src = base64.b64decode(req["b64"]).decode()
        compile(src, "compose_agent.py", "exec")
        with open(os.path.abspath(__file__) + ".new", "w") as f:
            f.write(src)
        os.replace(os.path.abspath(__file__) + ".new", os.path.abspath(__file__))
        threading.Timer(0.5, lambda: os._exit(0)).start()      # systemd Restart=always
        return {"ok": True, "restarting": True}
    raise ValueError(f"unknown op {op!r}")


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):  # noqa: D401 — one line per op on stderr (journald)
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % a))

    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 — unauthenticated liveness only
        self._send(200, {"ok": True}) if self.path == "/ping" else self._send(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        if self.path != "/op":
            return self._send(404, {"error": "not found"})
        if self.headers.get("X-ACS-Token", "") != TOKEN:
            return self._send(403, {"error": "bad token"})
        n = int(self.headers.get("Content-Length") or 0)
        _last_call[0] = time.time()
        try:
            out = handle(json.loads(self.rfile.read(n) or b"{}"))
            self._send(429 if out.get("busy") else 200, out)
        except Exception as e:  # noqa: BLE001 — report, never crash the agent
            self._send(500, {"error": f"{type(e).__name__}: {e}"[-1500:]})


def main() -> int:
    for d in (CTX, TASKS, BUILT, EPS):
        os.makedirs(d, exist_ok=True)
    # Projects left by a previous agent process are unknown to it: the trainer's gc removes them.
    def _watchdog() -> None:
        while True:
            time.sleep(30)
            if IDLE_SHUTDOWN_S > 0 and time.time() - _last_call[0] > IDLE_SHUTDOWN_S:
                sys.stderr.write(f"no trainer call for {IDLE_SHUTDOWN_S:.0f}s: shutting down (-> terminate)\n")
                subprocess.run(["shutdown", "-h", "now"])

    threading.Thread(target=_watchdog, daemon=True).start()
    ThreadingHTTPServer.daemon_threads = True
    srv = ThreadingHTTPServer((BIND, PORT), _Handler)
    sys.stderr.write(f"compose_agent on :{PORT} root={ROOT} max_projects={MAX_PROJECTS}\n")
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
