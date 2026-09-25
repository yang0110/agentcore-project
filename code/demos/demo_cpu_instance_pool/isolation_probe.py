#!/usr/bin/env python3
"""Cross-episode isolation probe for the CPU farm: two sub_sessions A and B on ONE instance.

A plants a secret file, a /tmp file and a detached background process; B then tries to reach all of
them. Every check prints PASS/FAIL, and the script exits non-zero on any FAIL.

    sudo python3 isolation_probe.py                 # offline: CpuFarm in-process (needs root for uid switching)
    AGENTCORE_SANDBOX_RUNTIME_ARN=<arn> python3 isolation_probe.py --live
                                                    # live: both sub_sessions under ONE runtimeSessionId
                                                    # => the same warm m5 (boot_id printed to prove it)

Checks:
  * A and B run as different, non-root uids,
  * B cannot list the farm root, nor list / read A's dir (Permission denied),
  * B cannot read or write A's files through the server's read_file / write_file ("../<A>/…"),
  * B cannot see A's /tmp file, nor signal A's process,
  * the server's AWS_* env is not inherited, and the NPROC / FSIZE caps are in force,
  * A's sub_session id (the key to A's farm ops) is not visible in any /proc/*/cmdline,
  * after A closes, A's detached process and A's /tmp file are gone.
It also prints (INFO, not checked) what B can reach over the SHARED network namespace.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SECRET = "secret-" + uuid.uuid4().hex[:8]
RESULTS: list[tuple[bool, str]] = []


def check(ok: bool, what: str, detail: str = "") -> None:
    RESULTS.append((ok, what))
    print(f"  {'PASS' if ok else 'FAIL'}  {what}" + (f"   [{detail.strip()[:160]}]" if detail else ""))


class FarmEp:
    """Offline: one sub_session of an in-process CpuFarm (the exact server code path)."""

    def __init__(self, farm, sub: str) -> None:
        self.farm, self.sub = farm, sub

    async def open(self) -> None:
        self.farm.open(self.sub)

    async def close(self) -> None:
        self.farm.close(self.sub)

    async def sh(self, cmd: str) -> tuple[int, str]:
        r = self.farm.exec(self.sub, cmd, {}, 30)
        return r["exit_code"], (r["stdout"] + r["stderr"])

    async def write(self, path: str, text: str) -> str:
        r = self.farm.write_file(self.sub, path, text)
        return "ok" if r.get("ok") else r.get("error", "")

    async def read(self, path: str) -> str:
        return self.farm.read_file(self.sub, path).get("text", "")


class LiveEp:
    """Live: one sub_session over invoke_agent_runtime (RayFarmRuntimeSandbox), shared runtimeSessionId."""

    def __init__(self, shared: str, sub: str) -> None:
        from agentcore_slime_demo.agentcore_runtime_sandbox import RayFarmRuntimeSandbox

        self.sb, self.sub = RayFarmRuntimeSandbox(shared_session_id=shared, sub_session=sub), sub

    async def open(self) -> None:
        await self.sb.__aenter__()

    async def close(self) -> None:
        await self.sb.__aexit__(None, None, None)

    async def sh(self, cmd: str) -> tuple[int, str]:
        code, out, err = await self.sb.exec(cmd, timeout=30)
        return code, out + err

    async def write(self, path: str, text: str) -> str:
        await self.sb.write_file(path, text)
        return "sent"

    async def read(self, path: str) -> str:
        return await self.sb.read_file(path)


async def probe(a, b, root: str) -> None:
    await a.open()
    await b.open()
    adir = f"{root}/{a.sub}"
    _, boot_a = await a.sh("cat /proc/sys/kernel/random/boot_id")
    _, boot_b = await b.sh("cat /proc/sys/kernel/random/boot_id")
    print(f"boot_id A={boot_a.strip()[:8]} B={boot_b.strip()[:8]}  root={root}")
    check(boot_a.strip() == boot_b.strip(), "A and B are on the SAME instance (same boot_id)")

    await a.write("marker.txt", SECRET)
    _, pid = await a.sh("setsid nohup sleep 1000 >/dev/null 2>&1 < /dev/null & echo $!")
    pid = pid.strip().splitlines()[-1] if pid.strip() else "0"
    await a.sh(f"echo {SECRET} > /tmp/leak_{a.sub}")

    _, ua = await a.sh("id -u")
    _, ub = await b.sh("id -u")
    ua, ub = ua.strip(), ub.strip()
    check(ua != ub and "0" not in (ua, ub), f"distinct non-root uids (A={ua}, B={ub})")

    code, out = await b.sh(f"ls {root}")
    check(code != 0 and "ermission denied" in out, "B cannot list the farm root", out)
    code, out = await b.sh(f"ls {adir}")
    check(code != 0 and "ermission denied" in out, "B cannot list A's dir", out)
    code, out = await b.sh(f"cat {adir}/marker.txt")
    check(SECRET not in out and "ermission denied" in out, "B cannot read A's file by explicit path", out)
    code, out = await b.sh(f"cat ../{a.sub}/marker.txt ../*/marker.txt")
    check(SECRET not in out, "B cannot read A's file via ../", out)

    got = await b.read(f"../{a.sub}/marker.txt")
    check(SECRET not in got, "server read_file refuses ../<A>/marker.txt", got)
    await b.write(f"../{a.sub}/marker.txt", "overwritten-by-B")
    check(SECRET in await a.read("marker.txt"), "server write_file refuses ../<A>/marker.txt")
    got = await b.read(f"{adir}/marker.txt")
    check(SECRET not in got, "server read_file refuses A's absolute path", got)

    code, out = await b.sh(f"cat /tmp/leak_{a.sub}")
    check(SECRET not in out, "B cannot read A's /tmp file (umask 077)", out)
    code, out = await b.sh(f"kill -0 {pid}")
    check(code != 0 and "ermitted" in out, f"B cannot signal A's process {pid}", out)
    code, out = await b.sh("env | grep -c '^AWS_' || true")
    check(out.strip().splitlines()[-1] == "0", "server AWS_* env not inherited", out)
    code, out = await b.sh("echo nproc=$(ulimit -u) fsize=$(ulimit -f)")
    check("unlimited" not in out.split("fsize")[0] and "fsize=unlimited" not in out,
          "NPROC and FSIZE caps in force", out)

    pat = f"{a.sub[:-1]}[{a.sub[-1]}]"   # the [x] trick: grep's own cmdline must not match
    code, out = await b.sh(f"grep -l '{pat}' /proc/[0-9]*/cmdline 2>/dev/null | wc -l")
    check(out.strip().splitlines()[-1] == "0", "A's sub_session id is not in any visible cmdline", out)
    # Informational only (shared network namespace): what an episode can reach on the instance.
    for what, url in (("server port 127.0.0.1:8080/ping", "http://127.0.0.1:8080/ping"),
                      ("instance metadata 169.254.169.254", "http://169.254.169.254/latest/meta-data/")):
        _, out = await b.sh(f"curl -s -m 3 -o /dev/null -w '%{{http_code}}' {url} || echo unreachable")
        print(f"  INFO  B -> {what}: http {out.strip()}")

    await a.close()
    code, out = await b.sh(f"kill -0 {pid} 2>&1; echo rc=$?")
    check("No such process" in out, f"A's detached process {pid} is gone after close", out)
    code, out = await b.sh(f"ls /tmp/leak_{a.sub} 2>&1")
    check("No such file" in out, "A's /tmp file is gone after close", out)
    await b.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="run against the deployed Runtime")
    args = ap.parse_args()
    a_sub, b_sub = f"ep-{uuid.uuid4().hex[:16]}", f"ep-{uuid.uuid4().hex[:16]}"
    if args.live:
        shared = f"isoprobe-{uuid.uuid4().hex}"
        a, b = LiveEp(shared, a_sub), LiveEp(shared, b_sub)
        root = os.environ.get("SANDBOX_FARM_ROOT", "/workspace/_farm")
    else:
        if os.geteuid() != 0:
            print("offline probe needs root (uid switching); run it inside the agentcore_p4 container")
            return 2
        from agentcore_slime_demo.sandbox_cpu_farm import CpuFarm

        root = os.path.join(tempfile.mkdtemp(prefix="isoprobe-"), "_farm")
        os.chmod(os.path.dirname(root), 0o711)
        farm = CpuFarm(root)
        a, b = FarmEp(farm, a_sub), FarmEp(farm, b_sub)
    asyncio.run(probe(a, b, root))
    bad = [w for ok, w in RESULTS if not ok]
    print(f"{len(RESULTS) - len(bad)}/{len(RESULTS)} checks passed" + (f"; FAILED: {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
