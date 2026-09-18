"""Custom Megatron init hook — enable the SPAWN-based persistent async checkpoint worker.

Wired via slime's ``--custom-megatron-init-path`` (see initialize.py: slime calls this at the
end of its ``init(args)``). It exists to close a gap between slime and Megatron's checkpoint
plumbing that otherwise makes the end-of-training checkpoint save intermittently DEADLOCK.

## The bug we are fixing

Megatron's async-save machinery has two backends (see
``megatron/core/dist_checkpointing/strategies/async_utils.py``):

* ``TemporalAsyncCaller`` — the module-level DEFAULT. Each save forks a writer process off the
  MAIN training process (``mp.get_context('fork').Process(...)``), then ``self.process.join()``.
* ``PersistentAsyncCaller`` — a SPAWNED (fresh-interpreter) worker created once and reused; the
  main process hands it host tensors over a queue and never forks to write.

Megatron only switches to the persistent backend inside ``megatron.training.initialize`` when
``args.async_save and args.use_persistent_ckpt_worker`` — it calls ``init_persistent_async_worker()``
there (initialize.py). **slime never calls that.** slime runs its own ``init()``
(``slime/backends/megatron_utils/initialize.py``) which bypasses ``initialize_megatron``, so even
though slime sets ``args.use_persistent_ckpt_worker = True`` (arguments.py), the async queue stays
the default fork-based ``TemporalAsyncCaller``.

Forking a process that holds CUDA/NCCL/inductor-compile threads is unsafe: if a background thread
holds an internal lock at the instant of the fork, the child inherits the locked mutex with no
thread to release it and hangs — so the writer's ``count_queue.join()`` (filesystem_async.py) never
returns, *after* the ``.distcp`` files are already fully written. It is intermittent (depends on the
lock state at the fork instant), which is exactly what we observed: some runs finalize fine, others
wedge forever in ``save_model`` with GPUs at 0%.

## The fix

Do the one thing slime omits: when async-save + persistent worker are both requested, call
``init_persistent_async_worker()`` ourselves. That re-creates Megatron's async queue as the
persistent, SPAWN-based caller. The spawned worker is a clean interpreter (no inherited CUDA/NCCL
threads or locked mutexes), so the write — and any inner writer fork it does — is safe. slime still
finalizes it synchronously at the final rollout (``maybe_finalize_async_save(blocking=True)``), so
the save completes before the process exits, unchanged in every respect except that it no longer
deadlocks.

This is a faithful replica of Megatron's own ``initialize.py`` lines (``if args.async_save and
args.use_persistent_ckpt_worker: init_persistent_async_worker()``), routed through the supported
slime hook rather than patching either dependency.
"""
from __future__ import annotations


def enable_persistent_async_ckpt(args) -> None:
    """Switch Megatron's async-save queue to the spawn-based persistent worker.

    No-op unless both ``--async-save`` and ``use_persistent_ckpt_worker`` are set (the latter is
    forced on by slime). Idempotent and safe to call once per actor process during init.
    """
    if not getattr(args, "async_save", False):
        return
    if not getattr(args, "use_persistent_ckpt_worker", False):
        return

    from megatron.training.async_utils import init_persistent_async_worker

    init_persistent_async_worker()
    # rank-0-ish breadcrumb so the fix is visible in the training log
    if getattr(args, "rank", 0) == 0:
        print(
            "[agentcore] enabled Megatron PersistentAsyncCaller (spawn) for async checkpoint save "
            "— avoids the fork-of-CUDA-process deadlock in the default TemporalAsyncCaller",
            flush=True,
        )
