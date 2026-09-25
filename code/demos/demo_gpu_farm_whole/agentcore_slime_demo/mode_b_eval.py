"""Held-out eval for demo_cpu_tasks's FULLY-ASYNC (Mode B) training.

Why this exists
---------------
Demo A's Mode B trains with slime's fully-async rollout
(``slime.rollout.fully_async_rollout.generate_rollout_fully_async``), which runs a **persistent
background worker thread with its own asyncio event loop**. slime's ``GenerateState`` is a
process-global singleton, and its ``asyncio.Semaphore`` binds to the FIRST event loop that
``acquire()``s it — the worker's loop. slime's periodic held-out eval, however, runs generation on
the MAIN loop (via ``slime.utils.async_utils.run``); it reuses that same singleton semaphore and
dies with ``RuntimeError: <Semaphore ...> is bound to a different event loop``.

We keep held-out eval (its solve-rate is a key metric) by wiring eval through THIS function
instead of the training rollout function:

    --rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async  # train
    --eval-function-path    agentcore_slime_demo.mode_b_eval.generate_rollout               # eval

On each eval it:
  1. Quiesces the fully-async worker and **waits for its event loop to fully tear down** (the
     worker's drain can run for up to ~30s after ``running=False``), so it cannot race eval to
     recreate the ``GenerateState`` singleton on its own loop.
  2. Drops the ``GenerateState`` singleton so slime's stock eval rebuilds the semaphore on THIS
     (main) loop, then runs the stock eval rollout unchanged (one agentic episode per eval prompt
     via our ``--custom-generate-function-path``, ``evaluation=True``).
  3. Drops the singleton again so the NEXT training rollout's worker rebuilds the semaphore on the
     fresh worker loop (otherwise the worker would inherit this main-loop-bound semaphore and the
     mirror-image crash would hit training).

Cost / caveat
-------------
The worker's warm output queue (completed-but-unconsumed groups) is discarded when it stops, so the
deep buffer **re-warms over the next few rollouts** after every eval. Keep ``EVAL_INTERVAL`` well
above the re-warm time so this perturbation stays a small fraction of the run and the staleness
curve is measured mostly at steady state (see docs/sandbox-concurrency-ablation.md).

This reaches into ``fully_async_rollout`` internals (``_global_worker``, ``_worker_lock``) and the
``GenerateState`` singleton on purpose; it is coupled to that slime module by design.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)

# How long to wait for the fully-async worker to die cleanly before orphaning it (see
# _quiesce_fully_async_worker). The worker drains in-flight tasks for up to ~30s, then tears its
# event loop down; a clean stop finishes shortly after that, so ~45s cleanly separates "stopping"
# from "wedged" without over-waiting.
_WORKER_STOP_GRACE_SECONDS = 45.0


def _drop_generate_state_singleton() -> None:
    """Remove the GenerateState singleton so its semaphore is rebuilt on the next acquire's loop."""
    from slime.rollout.sglang_rollout import GenerateState
    from slime.utils.misc import SingletonMeta

    # _instances is the shared metaclass registry; pop ONLY GenerateState (leave other singletons).
    SingletonMeta._instances.pop(GenerateState, None)


def _quiesce_fully_async_worker() -> None:
    """Stop the fully-async worker so eval can own the GenerateState semaphore.

    Sets ``running=False`` (which stops the worker submitting new tasks and makes it drain +
    tear down its event loop) and waits a BOUNDED grace window for its thread to die.

    Why bounded, not "join forever": the worker's ``asyncio.run`` teardown force-cancels any
    in-flight tasks left after the 30s drain, and one such cancellation can WEDGE inside
    ``_cancel_all_tasks`` (e.g. an aiohttp/sandbox cleanup that never completes). The deep buffer
    always keeps ~concurrency in-flight multi-turn episodes, so there are essentially always
    leftovers to force-cancel — an unbounded join would then hang the trainer forever. If the
    thread doesn't die within the grace window we ORPHAN it and proceed: it is a daemon thread
    (won't block process exit) holding no GPU (sglang goes idle once the worker stops submitting and
    its request was cancelled), and we null ``_global_worker`` + rebuild a fresh GenerateState
    singleton for eval, so the wedged thread's stale loop/semaphore can't interfere. Cost: one
    leaked daemon thread per wedged stop — acceptable for periodic eval; keep EVAL_INTERVAL large.
    """
    from slime.rollout import fully_async_rollout as fa

    worker = getattr(fa, "_global_worker", None)
    if worker is None:
        return
    # Signal the loop to stop; it exits its while-loop then drains in-flight tasks (up to ~30s).
    worker.running = False
    thread = getattr(worker, "worker_thread", None)
    if thread is not None:
        deadline = time.monotonic() + _WORKER_STOP_GRACE_SECONDS
        while thread.is_alive() and time.monotonic() < deadline:
            thread.join(timeout=1)
        if thread.is_alive():
            logger.warning(
                "fully-async worker did not stop within %.0fs (likely wedged cancelling an "
                "in-flight episode); orphaning it (daemon thread, no GPU) and proceeding to eval",
                _WORKER_STOP_GRACE_SECONDS,
            )
    with fa._worker_lock:
        # Only clear if it is still the worker we stopped (defensive against re-entrancy).
        if fa._global_worker is worker:
            fa._global_worker = None


def generate_rollout(args, rollout_id, data_source, evaluation: bool = False):
    """slime ``--eval-function-path`` entrypoint. Only meant for eval under Mode B."""
    from slime.rollout.sglang_rollout import generate_rollout as stock_generate_rollout

    if not evaluation:
        # We are the eval function; training uses the fully-async path. If ever called for a
        # training rollout, fall back to the stock rollout so behavior stays correct.
        return stock_generate_rollout(args, rollout_id, data_source, evaluation=evaluation)

    _quiesce_fully_async_worker()
    _drop_generate_state_singleton()
    try:
        return stock_generate_rollout(args, rollout_id, data_source, evaluation=True)
    finally:
        # Rebind for training: next rollout recreates the worker + a fresh singleton on its loop.
        _drop_generate_state_singleton()
