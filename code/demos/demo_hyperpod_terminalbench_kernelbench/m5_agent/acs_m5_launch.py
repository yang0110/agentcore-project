"""m5 compose client entry point: the image's own start-up first, then our server, as upstream runs it.

The wrapper layer clears ENTRYPOINT and makes our server the CMD (the microVM contract). On a microVM that
is the only way; on the m5 it dropped what the image itself starts (LocalStack in create-bucket, munge +
slurmd in slurm-simple-node-monitoring) and took :8080 that tasks' own services need (nginx, pypi-server,
mlflow). compose_agent._rewrite therefore runs this file instead, with:

    ACS_ORIG_START   JSON argv = the image's ENTRYPOINT + (the compose command, else the image's CMD),
                     started in the background as ACS_EXEC_USER, output in /tmp/.acs_orig.log
    ACS_SERVER_PORT  where our server listens (18080; compose_agent publishes that port)

Stdlib only; runs on the wrapper's standalone Python (/opt/acs/py), never the task's.
"""
from __future__ import annotations

import json
import os
import pwd
import subprocess
import sys
from http.server import ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _start_orig() -> None:
    argv = json.loads(os.environ.get("ACS_ORIG_START") or "[]")
    if not argv:
        return
    kw, env = {}, dict(os.environ)
    user = os.environ.get("ACS_EXEC_USER", "")
    if user:
        pw = pwd.getpwnam(user)
        kw = {"user": pw.pw_uid, "group": pw.pw_gid, "extra_groups": os.getgrouplist(user, pw.pw_gid)}
        env.update(HOME=pw.pw_dir, USER=user, LOGNAME=user)
    for k in ("ACS_ORIG_START", "ACS_SERVER_PORT"):
        env.pop(k, None)
    with open("/tmp/.acs_orig.log", "ab") as log:
        try:
            subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env,
                             start_new_session=True, **kw)
        except OSError as e:          # a missing entrypoint binary must not take the server down
            log.write(f"acs_m5_launch: could not start {argv}: {e}\n".encode())


if __name__ == "__main__":
    _start_orig()
    import acs_http_server  # noqa: E402  (records its start time + the image WORKDIR = our cwd)

    ThreadingHTTPServer.daemon_threads = True
    ThreadingHTTPServer(("0.0.0.0", int(os.environ.get("ACS_SERVER_PORT", "18080"))),
                        acs_http_server._Handler).serve_forever()
