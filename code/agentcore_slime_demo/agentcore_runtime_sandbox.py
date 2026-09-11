"""`AgentCoreRuntimeSandbox` — the sandbox as a *separate* AgentCore Runtime.

STATUS: untested scaffolding for the two-Runtime target (docs/harness-sandbox-decoupling.md
§6–§7). This is NOT the co-located Code-Interpreter path (`AgentCoreSandbox` in sandbox.py) —
it is the disaggregated one, where the sandbox is its **own** AgentCore Runtime that the harness
(or the slime rollout) drives over `invoke_agent_runtime`. Nothing here has been run against AWS;
a GPU Runtime instance costs money to launch, so treat every line as a sketch to smoke-test.

Contract: this is a drop-in `Sandbox` (sandbox.py:35-54). The difference from `AgentCoreSandbox`
is purely the transport — instead of the Code Interpreter client, each `exec`/`write_file`/
`read_file` becomes an `invoke_agent_runtime` call whose JSON payload names the op. The op is
served by `sandbox_server_agent.py` (a `BedrockAgentCoreApp` deployed on the sandbox Runtime).

Wiring (matches agentcore-usecases/common/gpu_deploy.py `invoke`):
    boto3.client("bedrock-agentcore").invoke_agent_runtime(
        agentRuntimeArn=<sandbox runtime ARN>,
        runtimeSessionId=<session id — one per episode = one isolated sandbox>,
        payload=json.dumps({"op": "exec", ...}).encode())
    -> resp["response"].read() -> json

The runtime ARN comes from AGENTCORE_SANDBOX_RUNTIME_ARN (set after `runtime-on-cp` deploys the
sandbox server). One `runtimeSessionId` per episode gives session isolation (the remote analogue
of one Code Interpreter session per rollout).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import uuid
from pathlib import Path

from .sandbox import ExecResult, FileContent

# Transient AgentCore InvokeAgentRuntime failures worth retrying: server-side 5xx (incl. the 502 a
# microVM returns while briefly unavailable) and throttling. botocore surfaces these either as an
# error Code in this set or as a message containing "(50x)". 4xx (bad ARN/payload) is NOT retried.
_RETRYABLE_CODES = {
    "RuntimeClientError", "ServiceException", "InternalServerException",
    "ThrottlingException", "TooManyRequestsException", "ServiceUnavailableException",
    "InternalFailure", "ServiceUnavailable",
}


def _is_transient(exc: Exception) -> bool:
    from botocore.exceptions import ClientError, ConnectionError as BotoConnError, EndpointConnectionError
    if isinstance(exc, (BotoConnError, EndpointConnectionError)):
        return True
    if isinstance(exc, ClientError):
        code = exc.response.get("Error", {}).get("Code", "")
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
        msg = str(exc)
        if code in _RETRYABLE_CODES or (isinstance(status, int) and 500 <= status < 600):
            return True
        if any(f"({s})" in msg for s in ("500", "502", "503", "504")):
            return True
        return _is_capacity(exc)
    return False


def _is_capacity(exc: Exception) -> bool:
    """A supply-side 'Insufficient EC2 capacity' error for the GPU sandbox's capacity-provider
    instance (e.g. g6.xlarge). AgentCore surfaces it as a RuntimeClientError whose message is
    'Insufficient EC2 capacity ... in the Availability Zone you requested (us-west-2X)'. This is
    qualitatively different from a 502 microVM blip: it is a regional GPU shortage that can persist
    for MANY MINUTES across every AZ, so it gets a far larger retry budget (see _retry_invoke)."""
    msg = str(exc).lower()
    return "insufficient" in msg and "capacity" in msg


def _retry_invoke(call, *, what: str = "invoke"):
    """Call `call()` (a boto3 invoke), retrying transient failures with exponential backoff.

    TWO budgets, by failure class:
      * transient 5xx/throttle (a brief microVM blip): INVOKE_MAX_RETRIES (default 8) extra attempts,
        INVOKE_RETRY_BASE_SEC (2.0s) backoff capped at INVOKE_RETRY_MAX_SEC (30s) — ~150s total.
      * 'Insufficient EC2 capacity' for the GPU sandbox instance (_is_capacity): a MUCH larger budget,
        CAPACITY_MAX_RETRIES (default 40) attempts, CAPACITY_RETRY_BASE_SEC (5.0s) capped at
        CAPACITY_RETRY_MAX_SEC (45s) — ~28min of patience. A regional g6.xlarge shortage is supply-side
        and can last minutes across all AZs; the tighter ~150s transient budget was too short and B3
        died 2026-09-09 when ONE episode's sandbox seed-write couldn't get an instance in time, aborting
        a whole multi-hour run. Pair this with a lower --sglang-server-concurrency to cut the number of
        simultaneous instances the run demands (less pressure on the shortage).
    Non-transient errors (bad ARN, 4xx) raise on the first try; retryable ones raise only once the
    matching budget is spent."""
    import logging
    import time

    max_retries = int(os.environ.get("INVOKE_MAX_RETRIES", "8"))
    base = float(os.environ.get("INVOKE_RETRY_BASE_SEC", "2.0"))
    cap = float(os.environ.get("INVOKE_RETRY_MAX_SEC", "30.0"))
    cap_max = int(os.environ.get("CAPACITY_MAX_RETRIES", "40"))
    cap_base = float(os.environ.get("CAPACITY_RETRY_BASE_SEC", "5.0"))
    cap_cap = float(os.environ.get("CAPACITY_RETRY_MAX_SEC", "45.0"))

    attempt = 0
    last: Exception | None = None
    while True:
        try:
            return call()
        except Exception as exc:  # noqa: BLE001 — re-raised below if not transient / budget spent
            if not _is_transient(exc):
                raise
            capacity = _is_capacity(exc)
            budget = cap_max if capacity else max_retries
            if attempt >= budget:
                raise
            last = exc
            b, c = (cap_base, cap_cap) if capacity else (base, cap)
            delay = min(b * (2 ** min(attempt, 6)), c)
            logging.getLogger(__name__).warning(
                "[agentcore] transient %s failure (attempt %d/%d%s): %s — retrying in %.1fs",
                what, attempt + 1, budget + 1, " CAPACITY" if capacity else "", exc, delay,
            )
            time.sleep(delay)
            attempt += 1
    raise last  # unreachable, keeps type-checkers happy


class AgentCoreRuntimeSandbox:
    """Sandbox backed by a remote AgentCore Runtime (the disaggregated target).

    Every op is one `invoke_agent_runtime` round-trip. The boto3 client is synchronous, so
    calls are offloaded with ``asyncio.to_thread`` (same pattern as ``AgentCoreSandbox``).
    """

    def __init__(
        self,
        runtime_arn: str | None = None,
        region: str | None = None,
        workdir: str = "/workspace",
        session_id: str | None = None,
        read_timeout: int | None = None,
    ) -> None:
        self.sandbox_id = f"acrt-{uuid.uuid4().hex}"
        self._arn = runtime_arn or os.environ.get("AGENTCORE_SANDBOX_RUNTIME_ARN")
        self._region = region or os.environ.get("AGENTCORE_REGION", "us-west-2")
        self._workdir = workdir
        # One session id == one isolated sandbox for the whole episode.
        self._session_id = session_id or f"sbx-{uuid.uuid4().hex}"
        self._read_timeout = read_timeout or int(os.environ.get("INVOKE_READ_TIMEOUT", "900"))
        self._client = None

    async def __aenter__(self) -> "AgentCoreRuntimeSandbox":
        if not self._arn:
            raise RuntimeError(
                "AGENTCORE_SANDBOX_RUNTIME_ARN unset — deploy sandbox_server_agent.py as an "
                "AgentCore Runtime first (see gpu_deploy.py runtime-on-cp)."
            )
        import boto3
        from botocore.config import Config

        # Wide read timeout: the first invoke on a GPU capacity provider can cold-launch the
        # instance (minutes), exactly like gpu_deploy.py `invoke`.
        cfg = Config(read_timeout=self._read_timeout, connect_timeout=60, retries={"max_attempts": 0})
        self._client = await asyncio.to_thread(
            boto3.client, "bedrock-agentcore", region_name=self._region, config=cfg
        )
        # Best-effort workspace init + liveness check (also warms a cold GPU instance).
        await self._invoke({"op": "ping", "workdir": self._workdir})
        return self

    async def __aexit__(self, *exc) -> None:
        # Session state is reclaimed by AgentCore when the session goes idle; nothing to stop
        # explicitly. (Instance teardown is a fleet concern — gpu_deploy.py cleanup.)
        self._client = None

    def _invoke_sync(self, payload: dict) -> dict:
        # AgentCore Runtimes intermittently return transient 5xx / throttling (e.g. a 502 while a
        # microVM is briefly unavailable). With retries disabled at the client, ONE such blip on ONE
        # episode's invoke used to bubble out of generate() and kill the whole ray job mid-training
        # (a2 died at step 8 after 7 good steps this way). Retry transient failures with backoff so a
        # single bad round-trip is absorbed instead of crashing a multi-hour run. Non-transient errors
        # (bad ARN, 4xx) still raise immediately.
        resp = _retry_invoke(
            lambda: self._client.invoke_agent_runtime(
                agentRuntimeArn=self._arn,
                runtimeSessionId=self._session_id,
                payload=json.dumps(payload).encode(),
            ),
            what=f"sandbox invoke op={payload.get('op')}",
        )
        raw = resp["response"].read()
        if not raw:
            return {}
        return json.loads(raw.decode())

    async def _invoke(self, payload: dict) -> dict:
        return await asyncio.to_thread(self._invoke_sync, payload)

    def _abs(self, path: str) -> str:
        if os.path.isabs(path) or not self._workdir:
            return path
        return f"{self._workdir}/{path}"

    async def exec(
        self, cmd: str, *, user: str = "root", env: dict[str, str] | None = None,
        timeout: int = 120, check: bool = False, idempotent: bool = True,
    ) -> ExecResult:
        res = await self._invoke({
            "op": "exec", "command": cmd, "user": user, "env": env or {},
            "timeout": int(timeout), "workdir": self._workdir,
        })
        code = int(res.get("exit_code", 1))
        stdout, stderr = res.get("stdout", ""), res.get("stderr", "")
        if check and code != 0:
            raise RuntimeError(f"exec failed ({code}): {stderr}")
        return (code, stdout, stderr)

    async def write_file(self, sandbox_path: str, content: FileContent, *, user: str = "root") -> None:
        if isinstance(content, Path):
            text = content.read_text()
        elif isinstance(content, bytes):
            text = content.decode(errors="replace")
        else:
            text = content
        await self._invoke({"op": "write_file", "path": self._abs(sandbox_path),
                            "text": text, "user": user})

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str:
        res = await self._invoke({"op": "read_file", "path": self._abs(sandbox_path)})
        return res.get("text", "")


class RayFarmRuntimeSandbox:
    """Demo B4 client: several episodes SHARE one g6 instance, whose single GPU is split by a Ray
    farm running inside it (``sandbox_ray_farm.RayGpuFarm`` on the server).

    The difference from ``AgentCoreRuntimeSandbox`` is the session model. AgentCore maps
    1 ``runtimeSessionId`` -> 1 instance, so to make N concurrent episodes share ONE g6 they all
    invoke under the SAME ``runtimeSessionId`` (``shared_session_id`` — one per rollout process) and
    carry a per-episode ``sub_session`` in every op payload. The g6 server leases one fractional-GPU
    actor per ``sub_session``, so the episodes run side by side on the single GPU. ``__aenter__``
    sends ``open`` (lease an actor, blocking on the farm's gate); ``__aexit__`` sends ``close``
    (clear this episode's dir + return the actor). Paths stay RELATIVE: the server resolves them
    inside the leased actor's per-``sub_session`` dir (where sitecustomize/PROBLEM.md/solution.py
    live), so ``workdir`` is advisory here.
    """

    def __init__(
        self,
        *,
        runtime_arn: str | None = None,
        region: str | None = None,
        workdir: str = "/workspace",
        shared_session_id: str | None = None,
        sub_session: str | None = None,
        read_timeout: int | None = None,
    ) -> None:
        self._arn = runtime_arn or os.environ.get("AGENTCORE_SANDBOX_RUNTIME_ARN")
        self._region = region or os.environ.get("AGENTCORE_REGION", "us-west-2")
        self._workdir = workdir
        # One SHARED runtimeSessionId => all episodes land on the SAME g6 instance (>=33 chars).
        self._session_id = (
            shared_session_id
            or os.environ.get("B4_SHARED_RUNTIME_SESSION")
            or f"b4farm-{uuid.uuid4().hex}"
        )
        # One sub-session per EPISODE => its own fractional-GPU actor + isolated workdir on the g6.
        self._sub = sub_session or f"ep-{uuid.uuid4().hex[:16]}"
        self.sandbox_id = f"rayfarm-{self._sub}"
        self._read_timeout = read_timeout or int(os.environ.get("INVOKE_READ_TIMEOUT", "900"))
        self._client = None

    async def __aenter__(self) -> "RayFarmRuntimeSandbox":
        if not self._arn:
            raise RuntimeError(
                "AGENTCORE_SANDBOX_RUNTIME_ARN unset — deploy the B4 GPU-shared sandbox_server_agent "
                "(ray farm baked in) as an AgentCore Runtime first."
            )
        import boto3
        from botocore.config import Config

        cfg = Config(read_timeout=self._read_timeout, connect_timeout=60, retries={"max_attempts": 0})
        self._client = await asyncio.to_thread(
            boto3.client, "bedrock-agentcore", region_name=self._region, config=cfg
        )
        # Lease a fractional-GPU actor for this episode (blocks on the g6 farm's gate if full).
        await self._invoke({"op": "open", "sub_session": self._sub, "workdir": self._workdir})
        return self

    async def __aexit__(self, *exc) -> None:
        if self._client is not None:
            with contextlib.suppress(Exception):
                await self._invoke({"op": "close", "sub_session": self._sub})
        self._client = None

    def _invoke_sync(self, payload: dict) -> dict:
        resp = _retry_invoke(
            lambda: self._client.invoke_agent_runtime(
                agentRuntimeArn=self._arn,
                runtimeSessionId=self._session_id,   # SHARED across episodes (one g6 instance)
                payload=json.dumps(payload).encode()),
            what=f"b4-farm invoke op={payload.get('op')} sub={self._sub}",
        )
        raw = resp["response"].read()
        return json.loads(raw.decode()) if raw else {}

    async def _invoke(self, payload: dict) -> dict:
        return await asyncio.to_thread(self._invoke_sync, payload)

    async def exec(
        self, cmd: str, *, user: str = "root", env: dict[str, str] | None = None,
        timeout: int = 120, check: bool = False, idempotent: bool = True,
    ) -> ExecResult:
        res = await self._invoke({
            "op": "exec", "sub_session": self._sub, "command": cmd,
            "env": env or {}, "timeout": int(timeout),
        })
        code = int(res.get("exit_code", 1))
        stdout, stderr = res.get("stdout", ""), res.get("stderr", "")
        if check and code != 0:
            raise RuntimeError(f"exec failed ({code}): {stderr}")
        return (code, stdout, stderr)

    async def write_file(self, sandbox_path: str, content: FileContent, *, user: str = "root") -> None:
        if isinstance(content, Path):
            text = content.read_text()
        elif isinstance(content, bytes):
            text = content.decode(errors="replace")
        else:
            text = content
        # Relative path on purpose: the server resolves it inside the leased actor's sub-session dir.
        await self._invoke({"op": "write_file", "sub_session": self._sub,
                            "path": sandbox_path, "text": text})

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str:
        res = await self._invoke({"op": "read_file", "sub_session": self._sub, "path": sandbox_path})
        return res.get("text", "")
