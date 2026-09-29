"""PREOPEN=N: keep N fresh sandbox sessions booted AHEAD of demand (library default 0 = off;
demo_cpu_tasks/run.sh defaults it to RB x NS, one step ahead).

The cold start (microVM boot + image pull + server ready) moves off the episode's critical path;
nothing else changes. Every pre-opened session is still brand-new, serves exactly ONE episode and is
stopped (StopRuntimeSession) when that episode ends — the same policy as without pre-open. While a
session waits for its episode it is registered with the heartbeat keeper (HEARTBEAT=1, default on
own Runtimes), so AgentCore does not idle-reclaim it.

    sandbox = await preopen.take(key, factory)   # factory() -> an un-entered sandbox
    async with sandbox as sb: ...                # already open; exit closes it as usual

Pools are per (key, event loop): key = what makes sessions interchangeable (backend + Runtime ARN /
image tag). Not for the warm-shared farms (agentcore-ray-farm): a sub_session opens in milliseconds
on the already-warm instance, and holding one would lease a farm slot for nothing — ignored there.
Terminal-Bench keeps its own look-ahead pre-open (TB_PREOPEN_BATCHES), which predicts WHICH task
Runtime is needed next; this one only keeps a count warm per key.

At interpreter exit any still-unclaimed session is stopped synchronously (best effort).
"""

from __future__ import annotations

import asyncio
import atexit
import collections
import contextlib
import os
import sys
import time

_POOLS: dict[tuple, collections.deque] = {}
_UNCLAIMED: set = set()          # opened sandboxes not yet handed to an episode (for atexit)
_WARNED = False


def preopen_n(kind: str | None = None) -> int:
    global _WARNED
    n = int(os.environ.get("PREOPEN", "0") or 0)
    if n > 0 and kind in ("agentcore-ray-farm", "ray-farm", "b4"):
        if not _WARNED:
            _WARNED = True
            print("[demo] preopen_skipped=warm_shared_farm (sub_sessions open on a warm instance)", flush=True)
        return 0
    return max(0, n)


class _Opened:
    """An already-entered sandbox in the `async with` shape generate code expects."""

    def __init__(self, sb, preopened: bool, wait_ms: float) -> None:
        self._sb, self.preopened, self.wait_ms = sb, preopened, wait_ms

    async def __aenter__(self):
        return self._sb

    async def __aexit__(self, *exc):
        return await self._sb.__aexit__(*exc)


async def _open(factory):
    sb = factory()
    if asyncio.iscoroutine(sb):
        sb = await sb
    sb = await sb.__aenter__() or sb
    _UNCLAIMED.add(sb)
    return sb


async def take(key, factory, kind: str | None = None):
    """Return an open sandbox for one episode: a pre-opened one if any, else open one now.
    Then top the pool for ``key`` back up to PREOPEN."""
    n = preopen_n(kind)
    if n <= 0:
        sb = factory()
        return await sb if asyncio.iscoroutine(sb) else sb
    pool = _POOLS.setdefault((key, id(asyncio.get_running_loop())), collections.deque())
    t0 = time.perf_counter()
    pending = pool.popleft() if pool else None
    sb = await (pending if pending is not None else _open(factory))
    _UNCLAIMED.discard(sb)
    wait_ms = (time.perf_counter() - t0) * 1000.0
    while len(pool) < n:
        pool.append(asyncio.ensure_future(_open(factory)))
    print(f"[demo] preopened={int(pending is not None)} preopen_wait_ms={wait_ms:.1f} "
          f"preopen_pool={len(pool)}", flush=True)
    return _Opened(sb, pending is not None, wait_ms)


@atexit.register
def _stop_unclaimed() -> None:
    for sb in list(_UNCLAIMED):
        client, arn, sid = (getattr(sb, a, None) for a in ("_client", "_arn", "_session_id"))
        if client is None or not arn or not sid:
            continue
        with contextlib.suppress(Exception):
            client.stop_runtime_session(runtimeSessionId=sid, agentRuntimeArn=arn)
            sys.stdout.write(f"[sbx-op] op=stop_session session={sid} status=ok (preopen, unclaimed)\n")
