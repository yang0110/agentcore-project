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
import json
import os
import uuid
from pathlib import Path

from .sandbox import ExecResult, FileContent


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
        resp = self._client.invoke_agent_runtime(
            agentRuntimeArn=self._arn,
            runtimeSessionId=self._session_id,
            payload=json.dumps(payload).encode(),
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
        await self._invoke({"op": "write_file", "path": self._abs(sandbox_path), "text": text})

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str:
        res = await self._invoke({"op": "read_file", "path": self._abs(sandbox_path)})
        return res.get("text", "")
