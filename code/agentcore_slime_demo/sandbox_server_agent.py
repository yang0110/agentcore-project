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

try:
    from bedrock_agentcore.runtime import BedrockAgentCoreApp
except Exception:  # pragma: no cover - only importable inside the deployed container
    BedrockAgentCoreApp = None  # type: ignore


DEFAULT_WORKDIR = os.environ.get("SANDBOX_WORKDIR", "/workspace")


def _ensure_workdir(workdir: str) -> None:
    os.makedirs(workdir or DEFAULT_WORKDIR, exist_ok=True)


def _do_exec(req: dict) -> dict:
    workdir = req.get("workdir") or DEFAULT_WORKDIR
    _ensure_workdir(workdir)
    cmd = req.get("command", "")
    user = req.get("user", "root")
    timeout = int(req.get("timeout", 120))
    env = {**os.environ, **(req.get("env") or {})}

    exports = "".join(f"export {k}={shlex.quote(str(v))}; " for k, v in (req.get("env") or {}).items())
    inner = f"cd {shlex.quote(workdir)}; {exports}bash -lc {shlex.quote(cmd)}"
    full = inner if user in ("", "root") else f"su {shlex.quote(user)} -c {shlex.quote(inner)}"
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
    return {"ok": True}


def _do_read(req: dict) -> dict:
    try:
        with open(req["path"]) as f:
            return {"text": f.read()}
    except FileNotFoundError:
        return {"text": "", "error": "not found"}


def handle(request: dict) -> dict:
    """Pure dispatch — unit-testable without AgentCore (no BedrockAgentCoreApp needed)."""
    op = (request or {}).get("op", "ping")
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
