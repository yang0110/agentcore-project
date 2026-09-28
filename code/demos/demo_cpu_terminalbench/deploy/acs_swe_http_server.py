"""BENCH=swebench entrypoint: deploy/acs_ctx/acs_http_server.py + chunked request bodies.

On capacity-provider Instances the AgentCore data plane forwards /invocations with
`Transfer-Encoding: chunked` (no Content-Length). The shared server reads Content-Length only, so the
body arrives empty and EVERY op is answered as a ping (`{"ok": true, "boot": ...}`; exec "fails" with
rc=1 and no output). The Terminal-Bench microVM path sends Content-Length, which is why it never hit
this. Kept as a separate shim so the shared server (and every Terminal-Bench image's recipe tag) is
untouched; the fix is the same two lines if it is ever folded into acs_http_server.py.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from http.server import ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import acs_http_server as base  # noqa: E402

_first_lock = threading.Lock()


def _read_body(h) -> bytes:
    if "chunked" in (h.headers.get("Transfer-Encoding") or "").lower():
        buf = bytearray()
        while True:
            size = int(h.rfile.readline().split(b";", 1)[0].strip() or b"0", 16)
            if size == 0:
                while h.rfile.readline() not in (b"\r\n", b"\n", b""):   # trailers
                    pass
                return bytes(buf)
            buf += h.rfile.read(size)
            h.rfile.readline()                                             # CRLF after each chunk
    return h.rfile.read(int(h.headers.get("Content-Length") or 0))


class _Handler(base._Handler):
    def do_POST(self):  # noqa: N802
        if not self.path.startswith("/invocations"):
            self._send(404, {"error": "not found"})
            return
        with _first_lock:
            if base._first_invoke_epoch is None:
                base._first_invoke_epoch = time.time()
        try:
            req = json.loads(_read_body(self) or b"{}")
            out = base.put_tar(req) if (req or {}).get("op") == "put_tar" else base.sandbox_server_agent.handle(req)
            if (req or {}).get("op", "ping") == "ping" and isinstance(out, dict):
                out["boot"] = base.boot_info()
            self._send(200, out)
        except Exception as e:  # noqa: BLE001 — report, never crash the server
            self._send(500, {"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    ThreadingHTTPServer.daemon_threads = True
    ThreadingHTTPServer(("0.0.0.0", 8080), _Handler).serve_forever()
