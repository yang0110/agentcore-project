"""Offline AgentCore idle-timeout simulator — the piece that makes this demo runnable with NO AWS.

Real AgentCore reclaims a session purely on *absence of invokes* past its idle timeout (~15 min
Code-Interpreter / 600 s GPU Runtime; recovery.py:18, deploy_two_runtime.py:402,430), and there is
NO snapshot — a reclaimed GPU instance comes back blank (see docs/gpu-snapshot.md). To show that
behavior in seconds instead of minutes, and without paying for a g6, we model it locally:

  * `ReclaimableLocalSandbox` is a `LocalSandbox` that tracks its own last-activity clock (updated
    by EVERY op — real or heartbeat) and raises "session terminated (idle timeout)" once reclaimed.
  * `AgentCoreIdleSim` is a background loop that watches the registered sandboxes and reclaims any
    that have been idle longer than `idle_timeout` — exactly what AgentCore does, just faster. It
    can also `force_recycle()` a session on demand, modelling a GPU instance recycle that kills even
    a heartbeated session (the case recovery.py must still handle).

Against real AgentCore you delete this file: the idle timer is AWS's, and the heartbeat is a real
`invoke_agent_runtime`/`executeCommand`. Here it is a local stand-in so the coordination pattern
(SessionKeeper + KeptSandbox) can be exercised end-to-end offline.
"""

from __future__ import annotations

import asyncio
import logging
import time

from agentcore_slime_demo.recovery import TransientSandboxError
from agentcore_slime_demo.sandbox import LocalSandbox

logger = logging.getLogger(__name__)


class ReclaimableLocalSandbox(LocalSandbox):
    """A LocalSandbox that can be 'reclaimed' by the idle simulator, like a real AgentCore session.

    Every op (real or heartbeat) refreshes `last_activity`; once `reclaimed` is set, any further op
    raises a message recovery.is_recoverable() classifies as recoverable ('terminated') — so the
    driver's recovery path re-seeds on a fresh session, exactly as with a real reclamation."""

    def __init__(self, *, clock=time.monotonic, **kw) -> None:
        super().__init__(**kw)
        self._clock = clock
        self.last_activity = clock()
        self.reclaimed = False
        self.reclaim_reason = ""

    async def __aenter__(self) -> "ReclaimableLocalSandbox":
        await super().__aenter__()
        self.last_activity = self._clock()
        return self

    def _touch(self) -> None:
        self.last_activity = self._clock()

    def _guard(self) -> None:
        if self.reclaimed:
            # Wording matches recovery._RECOVERABLE_MARKERS ('terminated') so it self-heals.
            raise TransientSandboxError(
                f"AgentCore session {self.sandbox_id} terminated ({self.reclaim_reason})"
            )

    async def exec(self, cmd: str, **kw):
        self._guard()
        self._touch()
        return await super().exec(cmd, **kw)

    async def write_file(self, *a, **kw):
        self._guard()
        self._touch()
        return await super().write_file(*a, **kw)

    async def read_file(self, *a, **kw):
        self._guard()
        self._touch()
        return await super().read_file(*a, **kw)

    def idle_seconds(self) -> float:
        return self._clock() - self.last_activity


class AgentCoreIdleSim:
    """Background reclaimer: marks a registered sandbox dead once it is idle past `idle_timeout`.

    This stands in for AgentCore's own idle reclamation so the demo needs no AWS. `tick` is the
    poll granularity; keep it small relative to `idle_timeout`."""

    def __init__(self, *, idle_timeout: float, tick: float = 0.25, clock=time.monotonic) -> None:
        self._idle_timeout = idle_timeout
        self._tick = tick
        self._clock = clock
        self._watched: list[ReclaimableLocalSandbox] = []
        self._task: asyncio.Task | None = None
        self.reclaimed_count = 0

    def watch(self, sb: ReclaimableLocalSandbox) -> None:
        self._watched.append(sb)

    def force_recycle(self, sb: ReclaimableLocalSandbox, reason: str = "instance recycle") -> None:
        """Kill a session regardless of activity — models a GPU instance recycle that defeats even
        a heartbeat (paused GPU state is not reattached; docs/gpu-snapshot.md)."""
        if not sb.reclaimed:
            sb.reclaimed = True
            sb.reclaim_reason = reason
            self.reclaimed_count += 1
            logger.info("[idle-sim] force-recycled %s (%s)", sb.sandbox_id, reason)

    async def start(self) -> "AgentCoreIdleSim":
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="agentcore-idle-sim")
        return self

    async def aclose(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def __aenter__(self) -> "AgentCoreIdleSim":
        return await self.start()

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    async def _run(self) -> None:
        try:
            while True:
                for sb in self._watched:
                    if not sb.reclaimed and sb.idle_seconds() >= self._idle_timeout:
                        sb.reclaimed = True
                        sb.reclaim_reason = f"idle > {self._idle_timeout:.0f}s"
                        self.reclaimed_count += 1
                        logger.info(
                            "[idle-sim] reclaimed %s after %.1fs idle (>= %.0fs timeout)",
                            sb.sandbox_id, sb.idle_seconds(), self._idle_timeout,
                        )
                await asyncio.sleep(self._tick)
        except asyncio.CancelledError:
            raise
