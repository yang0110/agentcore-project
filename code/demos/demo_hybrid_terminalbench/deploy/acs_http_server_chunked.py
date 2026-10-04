"""acs_http_server for AgentCore *Instance* (capacity-provider) Runtimes: also accepts chunked bodies.

On an Instance Runtime the platform forwards /invocations with `Transfer-Encoding: chunked` and no
Content-Length (a microVM sends Content-Length), so the stock server read an EMPTY body and answered every
op as a ping (seen 2026-10-04 on the g6 KernelBench Runtime). This wrapper reuses acs_http_server
unchanged — its file is part of the microVM image recipe hash, so it stays byte-identical — and only
swaps how the request body is read.
"""
from __future__ import annotations

import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import acs_http_server as base  # noqa: E402


def _read_chunked(rfile) -> bytes:
    out = io.BytesIO()
    while True:
        size = int(rfile.readline().split(b";", 1)[0].strip() or b"0", 16)
        if size == 0:
            while rfile.readline() not in (b"\r\n", b"\n", b""):   # trailers
                pass
            return out.getvalue()
        out.write(rfile.read(size))
        rfile.readline()                                            # CRLF after each chunk


class _Handler(base._Handler):
    def do_POST(self):  # noqa: N802
        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
            body = _read_chunked(self.rfile)
            self.headers.replace_header("Transfer-Encoding", "identity")
            self.headers["Content-Length"] = str(len(body))
            self.rfile = io.BufferedReader(io.BytesIO(body))
        return super().do_POST()


if __name__ == "__main__":
    base.ThreadingHTTPServer.daemon_threads = True
    base.ThreadingHTTPServer(("0.0.0.0", 8080), _Handler).serve_forever()
