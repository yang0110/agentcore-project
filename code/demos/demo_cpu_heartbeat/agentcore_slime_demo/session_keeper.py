"""`SessionKeeper` — keep AgentCore sandbox sessions warm across the weight-sync barrier.

WHY THIS EXISTS (see docs/demos-program.md "quiesce / resume the sandbox"):
During slime's trainer->inference **weight sync**, generation stalls, so every open sandbox
session sits idle. AgentCore has no concept of an "episode" or a "pause"; it reclaims a session
purely on **absence of invokes** (idle timeout: ~15 min Code-Interpreter / 600 s GPU Runtime;
`recovery.py:18`, `deploy_two_runtime.py:402,430`). So a quiesce that goes silent longer than the
idle timeout silently loses the session — and on GPU the instance is recycled with the paused
state gone (there is NO snapshot; you re-seed, see recovery.py).

The fix is a **heartbeat that lives OUTSIDE the harness.** We do not want every harness to grow
its own keepalive; instead ONE process-wide `SessionKeeper`:

  * holds a registry of live sandbox sessions (each with its own cheap `pinger` coroutine),
  * exposes a **gate** the sandbox awaits before each real invoke, and
  * is driven by the trainer's weight-sync barrier via `sync_begin()` / `sync_end()`.

During the sync window (`sync_begin` .. `sync_end`) the keeper (a) holds new sandbox invokes at
the gate — the "lightweight quiesce" semantics chosen in demos-program.md — and (b) a single
background loop heartbeats each session that is going stale, at an interval well under its idle
timeout, so AgentCore's idle clock never trips. Real episode traffic and heartbeats both refresh
the same per-session clock, so outside the sync window the keeper fires **zero** pings.

`KeptSandbox` is the composition point: wrap any `Sandbox` (LocalSandbox, AgentCoreSandbox, the
Runtime sandbox) in a `KeptSandbox` and the gate/touch/register/heartbeat wiring is added without
the harness or the sandbox implementation knowing anything about weight sync. The heartbeat is a
raw no-op op (`exec "true"`, i.e. one `executeCommand` / `invoke_agent_runtime` round-trip) issued
BELOW the gate, because a heartbeat during quiesce must not block on the very gate it defeats.

Heartbeat is necessary but NOT sufficient: an instance recycle (GPU) or a lost session still
happens, so it composes with `recovery.py` — if a session dies anyway, resume by restarting on a
fresh session and re-seeding the workspace, don't crash the episode.

This module has NO slime and NO AWS dependency, so it unit-tests and runs offline.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)

Pinger = Callable[[], Awaitable[None]]


@dataclass
class _Session:
    session_id: str
    pinger: Pinger
    idle_timeout: float   # seconds AgentCore tolerates with no invoke before reclaiming
    last: float           # monotonic clock of the last invoke (real op OR heartbeat)
    pings: int = 0


class SessionKeeper:
    """Process-wide registry + heartbeat loop that keeps sandbox sessions warm during weight sync.

    Lifecycle: `async with SessionKeeper(...) as keeper:` (or `await keeper.start()`), then the
    trainer's weight-sync barrier calls `keeper.sync_begin()` before the sync and
    `keeper.sync_end()` after. Sandboxes call `register` / `unregister` / `touch` / `gate`
    (handled for you by `KeptSandbox`).

    Args:
      ping_fraction: heartbeat a session once it has been idle for `idle_timeout * ping_fraction`
                     seconds. 0.5 => ping at half the timeout, i.e. >=2 chances to reset before
                     expiry, with margin for a slow round-trip.
      tick:          loop granularity (seconds). Set well below the smallest idle_timeout.
    """

    def __init__(
        self,
        *,
        ping_fraction: float = 0.5,
        tick: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._sessions: dict[str, _Session] = {}
        self._ping_fraction = ping_fraction
        self._tick = tick
        self._clock = clock
        # Gate is OPEN (set) normally; sync_begin clears it so gate() blocks new invokes.
        self._gate = asyncio.Event()
        self._gate.set()
        self._quiesced = False
        # Continuous keep-warm (used by the REAL trainer, demo_cpu_heartbeat/run.sh): heartbeat any
        # session that goes idle past ping_fraction*idle_timeout REGARDLESS of an explicit sync
        # window, WITHOUT closing the gate. A decoupled slime run gives the harness no callback at
        # its internal weight-sync barrier, but generation naturally goes silent during that barrier
        # (and rollout tails) — so an always-on idle heartbeat is what actually holds sessions warm
        # across the live sync stall. `sync_begin/sync_end` (the gate) stay available and unchanged
        # for the offline heartbeat_demo.py which drives the barrier explicitly.
        self._keep_warm = False
        self._task: asyncio.Task | None = None
        # Observability for the demo / metrics.
        self.total_pings = 0
        self.quiesce_count = 0

    # --- lifecycle ---------------------------------------------------------------------------
    async def start(self) -> "SessionKeeper":
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="session-keeper")
        return self

    async def aclose(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def __aenter__(self) -> "SessionKeeper":
        return await self.start()

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    # --- registry (called by KeptSandbox) ----------------------------------------------------
    def register(self, session_id: str, pinger: Pinger, idle_timeout: float) -> None:
        self._sessions[session_id] = _Session(session_id, pinger, float(idle_timeout), self._clock())

    def unregister(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def touch(self, session_id: str) -> None:
        """Record real-op activity — resets this session's idle clock (like an invoke does)."""
        s = self._sessions.get(session_id)
        if s is not None:
            s.last = self._clock()

    def idle_seconds(self, session_id: str) -> float | None:
        s = self._sessions.get(session_id)
        return None if s is None else self._clock() - s.last

    @property
    def live_sessions(self) -> int:
        return len(self._sessions)

    # --- gate (the "lightweight quiesce") ----------------------------------------------------
    async def gate(self) -> None:
        """Await before each REAL sandbox invoke. Returns immediately unless quiesced, in which
        case it blocks until `sync_end()` re-opens the gate. Heartbeats deliberately bypass this."""
        if self._quiesced:
            await self._gate.wait()

    # --- weight-sync barrier (called by the trainer/driver, NOT the harness) -----------------
    def sync_begin(self) -> None:
        """Enter the weight-sync window: hold new invokes and start heartbeating stale sessions."""
        if not self._quiesced:
            self.quiesce_count += 1
        self._quiesced = True
        self._gate.clear()
        logger.info("[keeper] sync_begin — gate closed, %d live session(s) held warm", self.live_sessions)

    def sync_end(self) -> None:
        """Leave the weight-sync window: re-open the gate so held invokes resume."""
        self._quiesced = False
        self._gate.set()
        logger.info("[keeper] sync_end — gate open (fired %d heartbeat(s) this run)", self.total_pings)

    # --- continuous keep-warm (the REAL trainer path) ----------------------------------------
    def keep_warm_begin(self) -> None:
        """Turn on always-on idle heartbeating WITHOUT closing the gate. Real ops flow normally;
        any session idle past ping_fraction*idle_timeout (e.g. during the weight-sync stall) is
        heartbeated so AgentCore's idle clock never trips. Idempotent."""
        self._keep_warm = True

    def keep_warm_end(self) -> None:
        self._keep_warm = False

    # --- the heartbeat loop ------------------------------------------------------------------
    async def _run(self) -> None:
        try:
            while True:
                if self._quiesced or self._keep_warm:
                    now = self._clock()
                    for s in list(self._sessions.values()):
                        if now - s.last >= s.idle_timeout * self._ping_fraction:
                            try:
                                await s.pinger()          # a raw no-op invoke; resets the idle clock
                            except Exception as exc:      # noqa: BLE001 — a failed ping != a crash
                                logger.warning("[keeper] heartbeat failed for %s: %r", s.session_id, exc)
                            s.last = self._clock()
                            s.pings += 1
                            self.total_pings += 1
                await asyncio.sleep(self._tick)
        except asyncio.CancelledError:
            raise


class KeptSandbox:
    """Wrap any `Sandbox` so it participates in a `SessionKeeper` — the integration pattern.

    This is the ONLY code that knows the sandbox and the keeper exist together; the harness and the
    sandbox implementation stay oblivious. Every real op awaits the gate (held during weight sync)
    and touches the idle clock; the heartbeat issues a raw no-op op BELOW the gate so it keeps the
    session warm even while the gate is closed.

    The heartbeat is `exec(heartbeat_cmd)` — one `executeCommand` (Code Interpreter) or
    `invoke_agent_runtime` (Runtime) round-trip, which is exactly what resets AgentCore's idle
    timer. (The Runtime path also has a lighter `{"op":"ping"}`; a no-op exec is backend-uniform.)
    """

    def __init__(self, inner, keeper: SessionKeeper, *, idle_timeout: float = 600.0,
                 heartbeat_cmd: str = "true") -> None:
        self._inner = inner
        self._keeper = keeper
        self._idle_timeout = idle_timeout
        self._heartbeat_cmd = heartbeat_cmd
        self.sandbox_id = getattr(inner, "sandbox_id", "kept")

    async def __aenter__(self) -> "KeptSandbox":
        await self._inner.__aenter__()
        self.sandbox_id = self._inner.sandbox_id
        self._keeper.register(self.sandbox_id, self._heartbeat, self._idle_timeout)
        return self

    async def __aexit__(self, *exc) -> None:
        self._keeper.unregister(self.sandbox_id)
        await self._inner.__aexit__(*exc)

    async def _heartbeat(self) -> None:
        # RAW op: does NOT await the gate (a heartbeat during quiesce must defeat the gate, not
        # wait on it). One round-trip that resets AgentCore's idle timer.
        await self._inner.exec(self._heartbeat_cmd, timeout=30)

    async def _op(self) -> None:
        await self._keeper.gate()             # held shut during weight sync
        self._keeper.touch(self.sandbox_id)   # real traffic resets the idle clock too

    async def exec(self, cmd: str, **kw):
        await self._op()
        return await self._inner.exec(cmd, **kw)

    async def write_file(self, *a, **kw):
        await self._op()
        return await self._inner.write_file(*a, **kw)

    async def read_file(self, *a, **kw):
        await self._op()
        return await self._inner.read_file(*a, **kw)
