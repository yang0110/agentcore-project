"""`sandbox_server_agent` — the deployed AgentCore Runtime that IS the sandbox.

STATUS: untested scaffolding for the two-Runtime target (docs/harness-sandbox-decoupling.md §7).
This is the *server* side of `AgentCoreRuntimeSandbox`: a `BedrockAgentCoreApp` you deploy on an
AgentCore Runtime (Code Interpreter-class for A3, or a **GPU capacity provider** for B3 — the real
AgentCore GPU instance the B rows target). It executes the harness's tool actions and the reward
checker in one persistent workspace, then returns structured results.

It speaks the tiny op protocol `AgentCoreRuntimeSandbox` sends:
    {"op":"ping","workdir":...}                       -> {"ok":true}
    {"op":"exec","command":...,"user":..,"env":{},"timeout":N,"workdir":..} -> {exit_code,stdout,stderr}
    {"op":"write_file","path":..,"text":..}           -> {"ok":true}
    {"op":"read_file","path":..}                      -> {"text":..}
    {"mode":"prepare"}   (GPU/B3 image only)          -> installs the torch stack onto the volume

Deploy (mirrors agentcore-usecases 08/09):
    - build an image from a Dockerfile that installs python + (for B) CUDA torch, copies this file,
      and runs `python -m agentcore_slime_demo.sandbox_server_agent`;
    - `gpu_deploy.py runtime-on-cp <name> <image> <exec_arn> <cp_arn> ...` to place it on a GPU
      capacity provider (B), or a normal Runtime (A);
    - export the resulting ARN as AGENTCORE_SANDBOX_RUNTIME_ARN for the client.

GPU note: on the GPU capacity provider, set LD_LIBRARY_PATH=/usr/lib64 so torch finds
libcuda.so.1 (the exact fix use case 08 needed). This process holds the workspace and the GPU
context for the life of the runtime session, so many invokes reuse one warm CUDA context — that is
what makes a *pool of long-lived sessions* (not per-episode instance launches) the viable design.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys

try:
    from bedrock_agentcore.runtime import BedrockAgentCoreApp
except Exception:  # pragma: no cover - only importable inside the deployed container
    BedrockAgentCoreApp = None  # type: ignore


DEFAULT_WORKDIR = os.environ.get("SANDBOX_WORKDIR", "/workspace")

# GPU (B3) only: the CUDA torch stack is >2 GB, over the Runtime image cap, so it is installed onto
# a persistent EBS volume (filesystemConfigurations) at mode=prepare — exactly like AgentCore
# use case 08 — and reused across sessions. SANDBOX_PYLIB is set (via ENV) ONLY in the GPU sandbox
# image; on the CPU (A3) image it is unset and none of this runs (the MBPP checker is pure Python).
PYLIB = os.environ.get("SANDBOX_PYLIB")                     # e.g. /mnt/models/pylib (GPU image only)
TORCH_SPEC = os.environ.get("SANDBOX_TORCH_SPEC", "torch==2.5.1")
TORCH_INDEX = os.environ.get("SANDBOX_TORCH_INDEX", "https://download.pytorch.org/whl/cu124")


def _ensure_workdir(workdir: str) -> None:
    os.makedirs(workdir or DEFAULT_WORKDIR, exist_ok=True)


def _gpu_env(base: dict | None = None) -> dict:
    """Env that makes the volume-installed torch import and find the injected driver.

    LD_LIBRARY_PATH=/usr/lib64 is also baked into the Dockerfile ENV (so it is in effect at
    PROCESS START — the loader caches its search path, so setting it late is too late), but we
    re-assert it here for subprocesses and prepend PYLIB to PYTHONPATH so the checker's torch
    resolves off the volume. No-op when PYLIB is unset (CPU/A3 image)."""
    env = dict(base if base is not None else os.environ)
    if PYLIB:
        env["PYTHONPATH"] = PYLIB + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        lib = "/usr/lib64"
        env["LD_LIBRARY_PATH"] = lib + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    return env


def _gpu_prepare(req: dict) -> dict:
    """Install the CUDA torch stack onto the volume once (GPU sandbox / B3 only).

    Mirrors use case 08: a SINGLE pinned pip pass against the cu124 index (PyPI as fallback for
    the non-torch deps). Pinning + one pass is what avoids pip re-resolving a mismatched newer
    torch. Idempotent: skips if torch is already on the volume."""
    if not PYLIB:
        return {"error": "prepare called but SANDBOX_PYLIB unset (not a GPU sandbox image)"}
    os.makedirs(PYLIB, exist_ok=True)
    if not os.path.exists(os.path.join(PYLIB, "torch")):
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--target", PYLIB,
             TORCH_SPEC, "numpy",
             "--index-url", TORCH_INDEX, "--extra-index-url", "https://pypi.org/simple"],
            check=True, env=_gpu_env(),
        )
    # Verify torch imports + sees the GPU, so a broken prepare fails loud here, not mid-rollout.
    check = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.__version__, torch.cuda.is_available())"],
        capture_output=True, text=True, env=_gpu_env(),
    )
    return {"status": "prepared", "pylib": PYLIB, "torch_probe": check.stdout.strip(),
            "torch_probe_err": check.stderr.strip(), "ok": check.returncode == 0}


def _do_exec(req: dict) -> dict:
    workdir = req.get("workdir") or DEFAULT_WORKDIR
    _ensure_workdir(workdir)
    cmd = req.get("command", "")
    user = req.get("user", "root")
    timeout = int(req.get("timeout", 120))
    # On the GPU (B3) image this prepends the volume PYLIB to PYTHONPATH and /usr/lib64 to
    # LD_LIBRARY_PATH so the checker's torch resolves + finds the injected driver; no-op on CPU.
    env = _gpu_env({**os.environ, **(req.get("env") or {})})

    exports = "".join(f"export {k}={shlex.quote(str(v))}; " for k, v in (req.get("env") or {}).items())
    inner = f"cd {shlex.quote(workdir)}; {exports}bash -lc {shlex.quote(cmd)}"
    # NOTE: `user` is intentionally IGNORED. AgentCore runs this container as uid 0 with an EMPTY
    # Linux capability set (CapEff/CapBnd=0), so CAP_SETUID/SETGID are unavailable and `su <user>`
    # fails with "cannot set groups: Operation not permitted". The sandbox is therefore root-only;
    # every command (including the Claude Code CLI, which tolerates root via IS_SANDBOX=1) runs as
    # root. The image makes /workspace + /home/agent world-writable so root writes succeed anyway.
    full = inner
    try:
        proc = subprocess.run(
            full, shell=True, capture_output=True, text=True, timeout=timeout, env=env,
        )
        return {"exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
    except subprocess.TimeoutExpired as e:
        return {"exit_code": 124, "stdout": e.stdout or "", "stderr": "timeout"}


def _do_write(req: dict) -> dict:
    path = req["path"]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(req.get("text", ""))
    # No chown: the container has no CAP_CHOWN (AgentCore zero-cap root) and the sandbox is root-only,
    # so ownership handoff is both impossible and unnecessary. `user` is accepted but ignored.
    return {"ok": True}


def _do_read(req: dict) -> dict:
    try:
        with open(req["path"]) as f:
            return {"text": f.read()}
    except FileNotFoundError:
        return {"text": "", "error": "not found"}


# ---- Demo B4: GPU-shared farm path -----------------------------------------------------------
# When a request carries a "sub_session", it is a B4 op: several episodes share THIS one g6
# instance (all invoked under one runtimeSessionId), and each sub_session is multiplexed onto a
# fractional slice of the single GPU by a Ray farm running in-process here (sandbox_ray_farm).
# Requests WITHOUT a sub_session take the legacy single-session path below (B2/B3), so one image
# serves all of series B — B4 just also carries a sub_session and needs `ray` installed. The farm
# is built lazily on the first B4 op (its ray.init / warm CUDA cost is paid once).
_FARM = None
_FARM_LOCK = None


def _farm():
    global _FARM, _FARM_LOCK
    import threading

    if _FARM_LOCK is None:
        _FARM_LOCK = threading.Lock()
    if _FARM is None:
        with _FARM_LOCK:
            if _FARM is None:
                from .sandbox_ray_farm import RayGpuFarm

                _FARM = RayGpuFarm.from_env().start()
    return _FARM


# ---- demo_cpu_terminalbench: warm-SHARED CPU path (one live session per task) ----------------
# The RELIABLE warm lever for the large Terminal-Bench image pool. A request carrying a sub_session
# on a CPU image (SANDBOX_SHARED_MODE=cpu) is an EPISODE slot: many episodes of one task all invoke
# under ONE shared runtimeSessionId, so they land on the SAME warm microVM (session affinity — no
# per-episode image pull / cold boot). Per-episode ISOLATION comes from the sub_session: op=open
# seeds a pristine copy of the task's baked workdir template into /sessions/<sub>; every exec/
# write_file/read_file for that sub resolves INSIDE it; op=close removes it. So episodes share the
# warm microVM but never see each other's workdir state. (Global OS state is still shared — see the
# WarmSharedRuntimeSandbox docstring; grade only on workdir artifacts, or recycle the session.)
SESSIONS_ROOT = os.environ.get("SANDBOX_SESSIONS_ROOT", "/sessions")
# Pristine copy of the task's initial workdir, baked into the image by deploy_terminalbench.py
# (`cp -a /app /opt/acs/task_template`). Reset target for each episode. Falls back to a bare mkdir
# when unset (e.g. a task whose Dockerfile seeds nothing).
TASK_TEMPLATE = os.environ.get("SANDBOX_TASK_TEMPLATE")


def _sub_dir(sub: str) -> str:
    # sub is a server-minted opaque id (ep-<hex> / anchor-*), but sanitize defensively so it can
    # never escape SESSIONS_ROOT via path separators.
    safe = "".join(c for c in sub if c.isalnum() or c in "-_")
    return os.path.join(SESSIONS_ROOT, safe or "ep")


def _resolve_in(base: str, path: str) -> str:
    """Resolve a (usually relative) client path inside this episode's sub-dir."""
    return path if os.path.isabs(path) else os.path.join(base, path)


def _handle_cpu_shared(op: str, request: dict) -> dict:
    sub = request["sub_session"]
    d = _sub_dir(sub)
    if op == "open":
        os.makedirs(d, exist_ok=True)
        # THE per-episode reset: a fresh, pristine copy of the task's baked workdir. Cheap for
        # Terminal-Bench task dirs (a handful of small seed files). `cp -a .../.` copies contents
        # (incl. dotfiles) into d without nesting a subdir. Best-effort: a missing template just
        # yields an empty workdir (the harness then writes PROBLEM.md into it).
        if TASK_TEMPLATE and os.path.isdir(TASK_TEMPLATE):
            subprocess.run(["cp", "-a", TASK_TEMPLATE + "/.", d + "/"], check=False)
        return {"ok": True, "workdir": d}
    if op == "close":
        subprocess.run(["rm", "-rf", d], check=False)
        return {"ok": True}
    if op == "ping":
        # Cheap liveness / heartbeat op: resets AgentCore's idle clock for the shared session
        # without touching the workdir (used by the SessionKeeper anchor pinger).
        return {"ok": True}
    if op == "exec":
        req = {**request, "workdir": d}          # run inside this episode's isolated workdir
        return _do_exec(req)
    if op == "write_file":
        req = {**request, "path": _resolve_in(d, request["path"])}
        return _do_write(req)
    if op == "read_file":
        req = {**request, "path": _resolve_in(d, request["path"])}
        return _do_read(req)
    return {"error": f"unknown cpu-shared op: {op!r}"}


def _handle_farm(op: str, request: dict) -> dict:
    sub = request["sub_session"]
    farm = _farm()
    if op == "open":
        return farm.open(sub)
    if op == "close":
        return farm.close(sub)
    if op == "ping":
        return {"ok": True, "farm": farm.describe()}
    if op == "exec":
        return farm.exec(sub, request.get("command", ""), request.get("env") or {},
                         int(request.get("timeout", 120)))
    if op == "write_file":
        return farm.write_file(sub, request["path"], request.get("text", ""))
    if op == "read_file":
        return farm.read_file(sub, request["path"])
    return {"error": f"unknown farm op: {op!r}"}


def handle(request: dict) -> dict:
    """Pure dispatch — unit-testable without AgentCore (no BedrockAgentCoreApp needed)."""
    # GPU (B3) one-time volume build: {"mode":"prepare"} installs the torch stack onto the volume.
    if (request or {}).get("mode") == "prepare":
        return _gpu_prepare(request)
    op = (request or {}).get("op", "ping")
    # A sub_session means a SHARED-session slot (many episodes on one warm microVM). Which kind:
    #   SANDBOX_SHARED_MODE=cpu  -> demo_cpu_terminalbench warm-shared workdir isolation (this image);
    #   otherwise                -> B4 GPU-shared Ray farm (many episodes on this one g6 GPU).
    if (request or {}).get("sub_session"):
        if os.environ.get("SANDBOX_SHARED_MODE", "").lower() == "cpu":
            return _handle_cpu_shared(op, request)
        return _handle_farm(op, request)
    if op == "ping":
        _ensure_workdir(request.get("workdir") or DEFAULT_WORKDIR)
        return {"ok": True}
    if op == "exec":
        return _do_exec(request)
    if op == "write_file":
        return _do_write(request)
    if op == "read_file":
        return _do_read(request)
    return {"error": f"unknown op: {op!r}"}


if BedrockAgentCoreApp is not None:
    app = BedrockAgentCoreApp()

    @app.entrypoint
    def entrypoint(request):  # noqa: ANN001
        return handle(request)

    if __name__ == "__main__":
        app.run()
