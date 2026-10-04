"""`DockerComposeSandbox` — one Terminal-Bench episode as a fresh `docker compose` project on the m5 host.

The second sandbox substrate of demo_hybrid_terminalbench, for the tasks an AgentCore microVM can't
run (multi-container compose, x86-only, images over 2 GB). A route looks like

    docker://<m5 private ip>:<agent port>/<task>

and the shared secret comes from TB_DOCKER_TOKEN. Open = the m5 agent's `up` (copy the task dir, rewrite
its compose file for this episode, start client + sidecars, wait for the client's server); every op
after that goes STRAIGHT to the episode's client container, which runs the same acs_http_server.py as
the microVM images (`POST /invocations`, same JSON ops, published on a random m5 port); close = `down`.

So this is `AgentCoreRuntimeSandbox` with a different transport: exec / write_file / read_file /
put_tar, the [sbx-op] timing records, the open epochs + boot reply that `_cold_start` decomposes, and
the in-session agent loop are all unchanged. Retries follow `_retry_invoke` (5xx / connection errors).
An `up` the m5 refuses with 429 (ACS_MAX_PROJECTS reached) waits and retries up to TB_DOCKER_OPEN_WAIT_S.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from .agentcore_runtime_sandbox import AgentCoreRuntimeSandbox, _timed_invoke

DOCKER_SCHEME = "docker://"


def is_docker_route(route: str) -> bool:
    return isinstance(route, str) and route.startswith(DOCKER_SCHEME)


def parse_route(route: str) -> tuple[str, int, str]:
    """docker://host:port/task -> (host, port, task)."""
    u = urllib.parse.urlparse(route)
    task = u.path.strip("/")
    if u.scheme != "docker" or not u.hostname or not u.port or not task:
        raise ValueError(f"bad docker route {route!r} (want docker://<host>:<port>/<task>)")
    return u.hostname, u.port, task


def _state_token() -> str:
    """The m5 agent secret from this demo's deploy/m5_host.json (host-side tools: validate, probe)."""
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "deploy", "m5_host.json")
    try:
        with open(p) as f:
            return json.load(f).get("token", "")
    except (OSError, ValueError):
        return ""


def agent_call(host: str, port: int, payload: dict, *, timeout: float = 900.0, token: str | None = None) -> dict:
    """One authenticated op on the m5 compose agent. 429 returns the body (busy); other errors raise.
    `token` overrides the shared host's secret (a per-episode instance has its own)."""
    token = token or os.environ.get("TB_DOCKER_TOKEN", "") or _state_token()
    if not token:
        raise RuntimeError("TB_DOCKER_TOKEN unset and no deploy/m5_host.json — run deploy/m5_host.py up")
    req = urllib.request.Request(
        f"http://{host}:{port}/op", data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json", "X-ACS-Token": token})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        if e.code == 429:
            return json.loads(body or "{}") | {"busy": True}
        raise RuntimeError(f"m5 agent {payload.get('op')} HTTP {e.code}: {body[-600:]}") from None


class DockerComposeSandbox(AgentCoreRuntimeSandbox):
    """A fresh compose project on the m5 per episode; ops go to its client container's server."""

    def __init__(self, route: str, *, workdir: str = "", session_id: str | None = None,
                 read_timeout: int | None = None) -> None:
        super().__init__(runtime_arn=route, workdir=workdir, session_id=session_id, read_timeout=read_timeout)
        self._agent_host, self._agent_port, self._task = parse_route(route)
        self._episode = self._session_id.lower()
        self._client_url: str | None = None
        self._token: str | None = None          # None = the shared host's secret (TB_DOCKER_TOKEN)
        self.sandbox_id = f"dc-{self._episode}"
        self.up_reply: dict = {}

    def _label(self) -> str:
        return f"docker:{self._task}"

    async def __aenter__(self) -> "DockerComposeSandbox":
        wait_s = float(os.environ.get("TB_DOCKER_OPEN_WAIT_S", "1800"))
        end = time.monotonic() + wait_s
        self.open_sent_epoch = time.time()
        t0 = getattr(self, "_open_t0", None) or time.perf_counter()   # a per-episode VM's open starts at launch
        while True:
            rep = await asyncio.to_thread(lambda: agent_call(self._agent_host, self._agent_port,
                                                             {"op": "up", "task": self._task, "episode": self._episode},
                                                             token=self._token))
            if not rep.get("busy"):
                break
            if time.monotonic() > end:
                raise RuntimeError(f"m5 host stayed full ({rep.get('live')}/{rep.get('max_projects')}) for {wait_s:.0f}s")
            await asyncio.sleep(5)
        self.up_reply = rep
        self._client_url = f"http://{self._agent_host}:{rep['port']}"
        # The first ping is part of the open: ONE [sbx-op] op=open record covers compose up + ping, the
        # same shape as a microVM open (metrics.py buckets it under perf/sandbox_open_*).
        from .agentcore_runtime_sandbox import _http_invoke_sync, _retry_invoke

        self.open_reply = await asyncio.to_thread(_retry_invoke, lambda: _http_invoke_sync(
            self._client_url, self._session_id, {"op": "ping", "workdir": self._workdir}, self._read_timeout),
            what="docker sandbox first ping")
        self.open_recv_epoch = time.time()
        sys.stdout.write(f"[sbx-op] ts={time.time():.3f} op=open ms={(time.perf_counter() - t0) * 1000.0:.1f} "
                         f"arn={self._label()} session={self._session_id} status=ok\n")
        sys.stdout.flush()
        boot = self.open_reply.setdefault("boot", {})
        boot.update(compose_pull_ms=rep.get("pull_ms"), compose_up_ms=rep.get("up_ms"),
                    compose_ready_ms=rep.get("ready_ms"))
        return self

    async def __aexit__(self, *exc) -> None:
        t0 = time.perf_counter()
        status = "ok"
        try:
            rep = await asyncio.to_thread(agent_call, self._agent_host, self._agent_port,
                                          {"op": "down", "episode": self._episode}, timeout=330)
            status = "ok" if rep.get("ok") else "down_error"
        except Exception as e:  # noqa: BLE001 — teardown: the pool's gc on next start is the fallback
            status = type(e).__name__
        sys.stdout.write(f"[sbx-op] ts={time.time():.3f} op=stop_session ms={(time.perf_counter() - t0) * 1000.0:.1f} "
                         f"arn={self._label()} session={self._session_id} status={status}\n")
        sys.stdout.flush()
        self._client_url = None

    def _invoke_sync(self, payload: dict) -> dict:
        if not self._client_url:
            raise RuntimeError(f"{self.sandbox_id}: not open")
        from .agentcore_runtime_sandbox import _http_invoke_sync

        # No 15-min AgentCore invoke cap here: an exec may legitimately run longer than the default read
        # timeout (a reference solution of 3600 s). The read must outlive the command, or the client times
        # out and the retry RE-RUNS the command.
        read_timeout = max(self._read_timeout, int(payload.get("timeout") or 0) + 120)
        return _timed_invoke(
            lambda: _http_invoke_sync(self._client_url, self._session_id, payload, read_timeout),
            op=str(payload.get("op")), arn=self._label(), session=self._session_id,
            what=f"docker sandbox op={payload.get('op')}")


def gc_hosts(routes, keep: list[str] | None = None) -> None:
    """Tear down every ep_* project on each m5 host in `routes` (crash recovery at pool start)."""
    hosts = {parse_route(r)[:2] for r in routes if is_docker_route(r)}
    for host, port in hosts:
        try:
            rep = agent_call(host, port, {"op": "gc", "keep": keep or []}, timeout=600)
            if rep.get("removed"):
                sys.stdout.write(f"[docker] gc {host}: removed {len(rep['removed'])} stale project(s)\n")
        except Exception as e:  # noqa: BLE001
            sys.stdout.write(f"[docker] gc {host} failed: {e}\n")
