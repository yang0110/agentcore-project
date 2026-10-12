"""Training rollout entrypoint: slime's fully-async rollout, plus a clean stop after the LAST step.

Why this exists
---------------
slime's fully-async worker (``slime.rollout.fully_async_rollout``) is a background thread that keeps
drawing groups and running episodes AHEAD of the trainer; Terminal-Bench's look-ahead pre-opens
their sandboxes even further ahead. Nothing tells it the run is over: after the last rollout it kept
opening and running episodes for steps that never come — through the whole final checkpoint save
(7.5 min in train48 run 2) — until the process exited under them. Those episodes then died with
``cannot schedule new futures after shutdown`` (25 at exit in train48 run 2, logged as infra errors)
and their sessions were never stopped (left to the idle timeout).

So once the last rollout's batch is collected, this stops the look-ahead WHILE the worker loop and
its executor are still alive:
  1. ``running = False``: the worker draws no new groups;
  2. on the worker's loop: drop the pool's demand, cancel pre-opens still booting and stop the
     booted ones (``TerminalBenchPool.aclose``), then cancel the in-flight episodes — each one closes
     its own session on the way out (StopRuntimeSession / compose down + reset);
  3. join the worker thread (bounded).
Anything this misses is still covered by the backstops: a late stop by any call that lands after its
sandbox closed, the exit hooks (stop every live AgentCore session; terminate every launched m5),
idle reclaim (RUNTIME_IDLE_TIMEOUT, 900 s), the m5 dead-man switch and ``m5_host.py gc-episodes``.

    --rollout-function-path agentcore_slime_demo.last_rollout.generate_rollout
"""

from __future__ import annotations

import asyncio
import logging
import time

logger = logging.getLogger(__name__)

# Bounds on the stop (a cancelled m5 episode still runs compose down + reset, a few tens of s).
_CLOSE_TIMEOUT_S = 180.0
_JOIN_TIMEOUT_S = 60.0


def generate_rollout(args, rollout_id, data_buffer, evaluation: bool = False):
    """slime ``--rollout-function-path`` entrypoint (training only; eval goes via mode_b_eval)."""
    from slime.rollout.fully_async_rollout import generate_rollout_fully_async

    out = generate_rollout_fully_async(args, rollout_id, data_buffer, evaluation=evaluation)
    if rollout_id >= int(args.num_rollout) - 1:
        stop_lookahead()
    return out


async def _close_on_worker_loop(pool) -> int:
    await pool.aclose()                 # demand cleared, pre-opens cancelled / stopped, keeper off
    me = asyncio.current_task()
    eps = [t for t in pool._workers if t is not me and not t.done()]
    for t in eps:                       # cancelling a dispatch worker cancels the episode it awaits;
        t.cancel()                      # the episode closes its sandbox before the worker unwinds
    await asyncio.gather(*eps, return_exceptions=True)
    # Then slime's group tasks on this loop: each awaits a pool job no worker will serve now, and the
    # worker's _loop (left alone: running=False ends it) waits up to 30 s for them before its
    # asyncio.run returns. That run's executor still waits for blocking calls in flight (a g6 open can
    # take minutes); each of those stops its own session when it lands (AgentCoreRuntimeSandbox late
    # stop), so the bounded join below may time out without leaking anything.
    rest = [t for t in asyncio.all_tasks() if t is not me and not t.done()
            and getattr(t.get_coro(), "__qualname__", "") != "AsyncRolloutWorker._loop"]
    for t in rest:
        t.cancel()
    await asyncio.wait(rest, timeout=30) if rest else None
    return len(eps)


def stop_lookahead() -> None:
    """Stop the fully-async worker and the Terminal-Bench look-ahead after the last rollout."""
    from slime.rollout import fully_async_rollout as fa

    from . import terminalbench_pool as tbp

    t0 = time.monotonic()
    worker = getattr(fa, "_global_worker", None)
    if worker is not None:
        worker.running = False
    pool = tbp._TB_POOL
    n = 0
    if pool is not None and pool.loop.is_running():
        try:
            n = asyncio.run_coroutine_threadsafe(_close_on_worker_loop(pool), pool.loop).result(_CLOSE_TIMEOUT_S)
        except Exception as exc:  # noqa: BLE001 — teardown: the backstops above still apply
            logger.warning("[last_rollout] look-ahead close incomplete: %r", exc)
    thread = getattr(worker, "worker_thread", None)
    if thread is not None:
        thread.join(timeout=_JOIN_TIMEOUT_S)
    with fa._worker_lock:
        if fa._global_worker is worker:
            fa._global_worker = None
    print(f"[demo] lookahead_stopped=1 episodes_cancelled={n} "
          f"worker_alive={int(bool(thread and thread.is_alive()))} ms={(time.monotonic() - t0) * 1000.0:.0f}",
          flush=True)
