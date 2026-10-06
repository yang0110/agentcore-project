"""Stdlib-only AgentCore HTTP-protocol server for the Terminal-Bench task images.

Runs on the demo's OWN standalone Python (/opt/acs/py, baked by deploy_terminalbench.py), never the
task's: Terminal-Bench images ship anything from python3.8 to no Python at all, and pip-installing
bedrock-agentcore into the task's interpreter would both fail on old Pythons and mutate the very
environment the task is graded in. So this speaks the AgentCore Runtime contract directly:

    GET  /ping         -> {"status": "Healthy"}
    POST /invocations  -> JSON op payload in, JSON result out (dispatched to sandbox_server_agent.handle)

on 0.0.0.0:8080 (threaded: a long exec must not block the platform's /ping health checks).

It also records BOOT TIMESTAMPS so each cold start can be decomposed on the client side:
    vm_boot_epoch      = now - /proc/uptime      (microVM kernel up)
    server_start_epoch = this process's start     (image pulled + container started + python up)
    first_invoke_epoch = first /invocations seen  (platform health-checked us and routed the call)
The client stamps its own send/receive epochs around the first ping; both clocks are EC2 time-sync'd.
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import tarfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SERVER_START_EPOCH = time.time()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# The task's working directory is the image's own WORKDIR (= our startup cwd; the wrapper never
# changes it). Tasks differ (/app, /app/astropy, /home/alice, ...), so report it instead of guessing.
IMAGE_WORKDIR = os.getcwd()
os.environ.setdefault("SANDBOX_WORKDIR", IMAGE_WORKDIR)

import sandbox_server_agent  # noqa: E402  (stdlib-only; its bedrock_agentcore import is optional)

_first_lock = threading.Lock()
_first_invoke_epoch: float | None = None


def _vm_boot_epoch() -> float | None:
    try:
        with open("/proc/uptime") as f:
            return time.time() - float(f.read().split()[0])
    except Exception:  # noqa: BLE001
        return None


VM_BOOT_EPOCH = _vm_boot_epoch()


def boot_info() -> dict:
    return {
        "vm_boot_epoch": VM_BOOT_EPOCH,
        "server_start_epoch": SERVER_START_EPOCH,
        "first_invoke_epoch": _first_invoke_epoch,
        "now_epoch": time.time(),
        "workdir": IMAGE_WORKDIR,
    }


def put_tar(req: dict) -> dict:
    """Extract a base64 tar.gz into `dest` (the grader injects the hidden tests this way, AFTER the
    agent's turns, so they are never in the image). Large bundles arrive in chunks: every chunk but
    the last sets `more`; they are staged in a file and extracted on the final one."""
    dest = req["dest"]
    stage = f"/tmp/.acs_put_{req.get('key', 'tar')}"
    with open(stage, "ab") as f:
        f.write(base64.b64decode(req.get("b64", "")))
    if req.get("more"):
        return {"ok": True, "staged": os.path.getsize(stage)}
    os.makedirs(dest, exist_ok=True)
    with open(stage, "rb") as f, tarfile.open(fileobj=io.BytesIO(f.read()), mode="r:gz") as tf:
        tf.extractall(dest)
    os.remove(stage)
    return {"ok": True}


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # quiet: every exec would otherwise log a line
        pass

    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/ping"):
            self._send(200, {"status": "Healthy", "time_of_last_update": int(time.time())})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        global _first_invoke_epoch
        if not self.path.startswith("/invocations"):
            self._send(404, {"error": "not found"})
            return
        with _first_lock:
            if _first_invoke_epoch is None:
                _first_invoke_epoch = time.time()
        n = int(self.headers.get("Content-Length") or 0)
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
            if (req or {}).get("op") == "put_tar":
                out = put_tar(req)
            else:
                out = sandbox_server_agent.handle(req)
            if (req or {}).get("op", "ping") == "ping" and isinstance(out, dict):
                out["boot"] = boot_info()
            self._send(200, out)
        except Exception as e:  # noqa: BLE001 — report, never crash the server
            self._send(500, {"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    ThreadingHTTPServer.daemon_threads = True
    ThreadingHTTPServer(("0.0.0.0", 8080), _Handler).serve_forever()
