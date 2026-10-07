"""`HarnessCliRuntimePool` — a small WARM pool of reused sessions on the decoupled CLI harness Runtime.

The harness Runtime (harness_runtime_agent.py) runs the real `claude -p`. For RL we want the harness
WARM, not a cold Node+CLI start per episode (the A3/B3 removal lesson), while the *sandbox* stays
fresh per episode. So we open a few long-lived `runtimeSessionId`s on the single harness Runtime ARN
and reuse them: borrow one per `claude_cli` episode, return it after. The weak-reset requirement is
met by harness_runtime_agent._run_claude using a per-episode cwd + a fresh sandbox session each
invoke, so no cross-episode state leaks on a reused harness session.

The harness entrypoint speaks `harness_kind` payloads (NOT the sandbox `op` protocol), so this pool
invokes `invoke_agent_runtime` directly rather than going through `AgentCoreRuntimeSandbox`. It
reuses the timing/retry helpers from agentcore_runtime_sandbox.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import secrets

logger = logging.getLogger(__name__)


class HarnessCliRuntimePool:
    def __init__(self, *, arn: str, region: str, n_sessions: int, keeper, idle_timeout: float,
                 read_timeout: int) -> None:
        self._arn = arn
        self._region = region
        self._n = max(1, int(n_sessions))
        self._keeper = keeper
        self._idle_timeout = idle_timeout
        self._read_timeout = read_timeout
        self._client = None
        self._free: asyncio.Queue[str] = asyncio.Queue()
        self._sessions: list[str] = []
        self._started = False

    def _ensure_client(self):
        if self._client is None:
            import boto3
            from botocore.config import Config

            cfg = Config(read_timeout=self._read_timeout, connect_timeout=60,
                         retries={"max_attempts": 0})
            self._client = boto3.client("bedrock-agentcore", region_name=self._region, config=cfg)
        return self._client

    async def invoke(self, session_id: str, payload: dict) -> dict:
        """One `invoke_agent_runtime` round-trip to the harness Runtime (timed + transient-retried)."""
        from .agentcore_runtime_sandbox import _timed_invoke

        client = self._ensure_client()
        resp = await asyncio.to_thread(
            _timed_invoke,
            lambda: client.invoke_agent_runtime(
                agentRuntimeArn=self._arn,
                runtimeSessionId=session_id,
                payload=json.dumps(payload).encode(),
            ),
            op=f"harness:{payload.get('harness_kind')}", arn=self._arn, session=session_id,
            what=f"harness invoke kind={payload.get('harness_kind')}",
        )
        raw = resp["response"].read()
        return json.loads(raw.decode()) if raw else {}

    async def _open_one(self) -> str:
        sid = f"harn-{secrets.token_hex(16)}"
        await self.invoke(sid, {"harness_kind": "noop"})      # warm the session (boot the microVM)
        if self._keeper is not None:
            async def _ping(s=sid):
                await self.invoke(s, {"harness_kind": "noop"})
            self._keeper.register(sid, _ping, self._idle_timeout)
        self._sessions.append(sid)
        return sid

    async def start(self) -> "HarnessCliRuntimePool":
        if self._started:
            return self
        for _ in range(self._n):
            sid = await self._open_one()
            self._free.put_nowait(sid)
        self._started = True
        logger.info("[harness_cli_pool] %d warm harness session(s) on %s", self._n, self._arn[-16:])
        return self

    @contextlib.asynccontextmanager
    async def borrow(self):
        """Borrow a warm harness session for one episode; return it after (replace a dead one)."""
        sid = await self._free.get()
        ok = True
        try:
            yield sid
        except BaseException:
            ok = False
            raise
        finally:
            if ok:
                self._free.put_nowait(sid)
            else:
                # The session may be wedged after a failed invoke: drop it and warm a replacement so
                # the pool keeps its size (best-effort; never block episode teardown on it).
                with contextlib.suppress(ValueError):
                    self._sessions.remove(sid)
                asyncio.get_running_loop().create_task(self._replace())

    async def _replace(self) -> None:
        with contextlib.suppress(Exception):
            sid = await self._open_one()
            self._free.put_nowait(sid)
