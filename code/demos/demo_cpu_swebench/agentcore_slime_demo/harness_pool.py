"""The decoupled harness tier: every demo's default rollout topology (from demo_cpu_scale_lanes).

        throughput  =  min( inference , sandbox , harness )

Async RL scales inference (SGLang engines) and the sandbox (sessions); the HARNESS, the agent loop
that drives each episode, has to scale with them, or a few coroutines quietly become the ceiling.
It scales WITH THE SANDBOX, not the trainer: the mini_claude loop runs inside the episode's own
sandbox session (in_session_harness.py) and dials the engine's adapter back, so N sessions carry N
loops (DSec's "move the agent loop out of the trainer"). What stays in the trainer is a dispatch tier
of W workers and one AnthropicAdapter per engine.

The three tiers are sized independently and joined PER EPISODE (no worker owns an engine or a session):

  * harness    W = HARNESS_WORKERS dispatch workers (default RB x NS, one rollout step). A free
               worker PULLS the next episode off one shared queue; pulling balances by availability,
               which is robust to the wide, variable latency of multi-turn episodes.
  * sandbox    the demo's own sandbox: a fresh session per episode (PREOPEN / HEARTBEAT apply as
               before). SANDBOX_SESSIONS=M > 0 (opt-in, own-Runtime backend only) instead keeps M
               persistent sessions PER IMAGE, borrowed one episode at a time in a fresh workdir
               (/workspace/ep-<hex>, removed on return). Not for tasks that change the OS.
  * inference  the engine with the fewest in-flight episodes (ties rotate), sticky for all turns.

The sandbox is taken BEFORE the engine, so an episode waiting for a session is not engine load.
Episodes return ONE bare Sample, the shape slime's fully-async driver needs.

Which tier binds shows in each episode's `[demo]` record: qdepth > 0 = harness-bound,
sandbox_wait_ms > 0 = sandbox-bound, engine_inflight climbing = inference-bound.

Trainer-side ceiling: dispatch is IO, which asyncio overlaps well; the adapters' tokenisation and
trajectory flattening still share the GIL, which is the part to watch at large W
(docs/harness-scaling.md).
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import logging
import os
import secrets
import threading
import time

logger = logging.getLogger(__name__)

# ---- inference tier: one AnthropicAdapter per SGLang engine, process-wide -----------------------
_ADAPTERS: dict[str, tuple[object, str, str]] = {}   # engine url -> (adapter, trainer url, session url)
_ADAPTER_LOCK = threading.Lock()


def adapter_for(args, engine: str) -> tuple[object, str, str]:
    """Lazily start the AnthropicAdapter for ``engine`` (shared by every episode on it).

    Returns (adapter, url for the trainer, url the sandbox session dials back to). Sync and
    thread-locked, so it is safe from any event loop (the fully-async driver runs eval on a second loop)."""
    with _ADAPTER_LOCK:
        if engine not in _ADAPTERS:
            from slime.agent.adapters import AnthropicAdapter
            from slime.agent.aiohttp_threaded import run_app_in_thread
            from slime.utils.processing_utils import load_tokenizer

            from .in_session_harness import adapter_bind, adapter_urls

            adapter = AnthropicAdapter(
                tokenizer=load_tokenizer(args.hf_checkpoint, trust_remote_code=True),
                sglang_url=engine,
                tool_parser=getattr(args, "sglang_tool_call_parser", None) or None,
                reasoning_parser=getattr(args, "sglang_reasoning_parser", None) or None,
            )
            # 0.0.0.0:ADAPTER_PORT_BASE+i: the in-session loop dials it back (SG->SG rule).
            host, port = adapter_bind(len(_ADAPTERS))
            handle = run_app_in_thread(adapter.app, host=host, port=port,
                                       thread_name=f"anthropic-adapter-{len(_ADAPTERS)}")
            _ADAPTERS[engine] = (adapter, *adapter_urls(handle.port))
        return _ADAPTERS[engine]


def one_sample(samples, reward: float):
    """finish_session's list -> ONE bare Sample (the longest), so the fully-async driver keeps slime's non-fan-out
    shape. Returns (sample, fan_out)."""
    samples = samples if isinstance(samples, list) else [samples]
    out = max(samples, key=lambda s: len(getattr(s, "tokens", []) or []))
    out.reward = reward
    return out, len(samples)


# ---- the pool ------------------------------------------------------------------------------------
_POOLS: dict[int, "HarnessPool"] = {}
_POOL_LOCK = threading.Lock()


def get_pool(args) -> "HarnessPool":
    """One pool per event loop (the fully-async worker and its eval run on different loops).

    A pool whose loop has stopped is retired here: its pooled sessions are closed in the background,
    so a driver that rebuilds its loop (fully-async, per eval) does not pile up M more live sessions per
    image each time."""
    loop = asyncio.get_running_loop()
    with _POOL_LOCK:
        for k, old in list(_POOLS.items()):
            if old.loop is not loop and (old.loop.is_closed() or not old.loop.is_running()):
                del _POOLS[k]
                old.retire()
        key = id(loop)
        if key not in _POOLS:
            _POOLS[key] = HarnessPool(args)
        return _POOLS[key]


class _Job:
    """One queued episode: the caller's future plus the task a worker runs it in (for cancellation)."""

    __slots__ = ("episode", "fut", "task", "cancelled")

    def __init__(self, episode, fut) -> None:
        self.episode, self.fut, self.task, self.cancelled = episode, fut, None, False


class Worker:
    """What an episode gets from the harness tier: its worker id, the queue depth at pickup and the
    worker's warm HTTP session."""

    __slots__ = ("wid", "qdepth", "http")

    def __init__(self, wid: int, qdepth: int, http) -> None:
        self.wid, self.qdepth, self.http = wid, qdepth, http


class HarnessPool:
    """W dispatch workers, the per-engine in-flight counts and (opt-in) pooled sandbox sessions,
    sized independently and joined per episode (see the module docstring)."""

    def __init__(self, args) -> None:
        self.args = args
        self.loop = asyncio.get_running_loop()
        step = int(getattr(args, "rollout_batch_size", 0) or 0) * int(getattr(args, "n_samples_per_prompt", 0) or 0)
        self.num_workers = max(1, int(os.environ.get("HARNESS_WORKERS", "") or step or 32))
        self.region = os.environ.get("AGENTCORE_REGION", os.environ.get("AWS_REGION", "us-west-2"))
        self.sandbox_kind = os.environ.get("SANDBOX", "agentcore").lower()
        m = int(os.environ.get("SANDBOX_SESSIONS", "0") or 0)
        if m > 0 and self.sandbox_kind not in ("agentcore-runtime", "runtime"):
            logger.warning("[harness_pool] SANDBOX_SESSIONS=%d ignored: pooling is for SANDBOX=agentcore-runtime "
                           "(got %s)", m, self.sandbox_kind)
            m = 0
        self.num_sessions = m
        self.open_rate = float(os.environ.get("SANDBOX_OPEN_RATE", "20"))
        self._queue: asyncio.Queue | None = None
        self._workers: list[asyncio.Task] = []
        self._http = None                            # ONE warm aiohttp session shared by every worker
        self._engine_urls: list[str] = []
        self._inflight: list[int] = []               # episodes currently on each engine
        self._rr = -1
        self._free: dict[object, asyncio.Queue] = {}  # image key -> free pooled sessions
        self._pool_sbx: list = []                    # every opened pooled session
        self._slot_episodes: dict[int, int] = {}     # slot -> episodes it has served
        self._started = False
        self._start_lock = asyncio.Lock()

    # ---- lifecycle -----------------------------------------------------------
    async def _ensure_started(self) -> None:
        if self._started:
            return
        async with self._start_lock:
            if self._started:
                return
            import concurrent.futures

            import aiohttp

            from . import router_trace

            # Every sandbox op is a blocking boto3 call run via asyncio.to_thread, i.e. on this loop's
            # default executor, which holds min(32, cpus+4) threads. With W episodes in flight that
            # would quietly cap the sandbox tier, so size it to the pool (+96: slack for pre-opens,
            # e.g. Terminal-Bench's look-ahead keeps up to 64 sessions booting).
            asyncio.get_running_loop().set_default_executor(concurrent.futures.ThreadPoolExecutor(
                max_workers=max(self.num_workers, self.num_sessions) + 96, thread_name_prefix="sbx-op"))
            router_url = f"http://{self.args.sglang_router_ip}:{self.args.sglang_router_port}"
            # Falls back to [router_url] if discovery fails (unpinned, still recorded).
            self._engine_urls = list(router_trace.engine_urls(router_url))
            self._inflight = [0] * len(self._engine_urls)
            # One warm, unlimited connector for all workers (the default caps at 100 connections).
            self._http = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0))
            self._queue = asyncio.Queue()
            for wid in range(self.num_workers):
                self._workers.append(asyncio.create_task(self._worker_loop(wid)))
            if self.num_sessions > 0:
                atexit.register(self._close_sessions_sync)
            self._started = True
            print(f"[demo] harness_pool_workers={self.num_workers} engines={len(self._engine_urls)} "
                  f"pooled_sessions_per_image={self.num_sessions} sandbox={self.sandbox_kind}", flush=True)

    async def _worker_loop(self, wid: int) -> None:
        assert self._queue is not None
        while True:
            job = await self._queue.get()
            if job is None:  # shutdown sentinel
                self._queue.task_done()
                return
            fut = job.fut
            try:
                if job.cancelled:      # the caller gave up before a worker got to it
                    continue
                job.task = asyncio.ensure_future(job.episode(Worker(wid, self._queue.qsize(), self._http)))
                result = await job.task
                if not fut.done():
                    fut.set_result(result)
            except BaseException as exc:  # noqa: BLE001 — never let one episode kill the worker
                if not fut.done():
                    fut.set_exception(exc)
                if isinstance(exc, asyncio.CancelledError) and not job.cancelled:
                    raise          # the worker itself is being cancelled (shutdown)
                if not isinstance(exc, asyncio.CancelledError):
                    logger.warning("[harness_pool] worker %d episode failed: %s", wid, exc)
            finally:
                job.task = None
                self._queue.task_done()

    async def run(self, episode):
        """Queue one episode (``await episode(worker)``) and await its result; a free worker pulls it."""
        await self._ensure_started()
        assert self._queue is not None
        job = _Job(episode, asyncio.get_running_loop().create_future())
        await self._queue.put(job)
        try:
            return await job.fut
        except asyncio.CancelledError:
            # Caller cancelled (rollout abort / over-sampling): stop its episode too, so the worker
            # frees its sandbox, engine slot and farm-gate slot instead of finishing abandoned work.
            job.cancelled = True
            if job.task is not None:
                job.task.cancel()
            raise

    # ---- inference tier: least-loaded engine per episode ---------------------------
    def acquire_engine(self) -> tuple[int, str]:
        """(engine_idx, engine_url) of the engine with the fewest in-flight episodes; ties rotate so
        an idle start doesn't pile onto engine 0. Pair with release_engine(idx)."""
        n = len(self._inflight)
        self._rr = (self._rr + 1) % n
        i = min(range(n), key=lambda j: (self._inflight[j], (j - self._rr) % n))
        self._inflight[i] += 1
        return i, self._engine_urls[i]

    def release_engine(self, i: int) -> None:
        self._inflight[i] -= 1

    @contextlib.contextmanager
    def engine(self):
        """acquire_engine() for the body of a `with`."""
        i, url = self.acquire_engine()
        try:
            yield i, url
        finally:
            self.release_engine(i)

    def inflight(self, i: int) -> int:
        return self._inflight[i]

    # ---- sandbox tier ----------------------------------------------------------------
    @contextlib.asynccontextmanager
    async def sandbox(self, key, factory):
        """Yield (sb, wait_ms, slot) for one episode.

        Default: a fresh session from ``factory()`` (an un-entered sandbox, or a coroutine returning
        one), through PREOPEN; stopped when the episode ends. SANDBOX_SESSIONS=M: borrow any free
        pooled session for ``key`` (M are opened on the key's first episode), in a fresh workdir."""
        if self.num_sessions <= 0:
            try:
                from . import preopen
            except ImportError:    # demos whose sandbox tier has its own pre-open (terminalbench_pool)
                preopen = None
            t0 = time.perf_counter()
            if preopen is not None:
                cm = await preopen.take(key, factory, kind=self.sandbox_kind)
            else:
                cm = factory()
                cm = await cm if asyncio.iscoroutine(cm) else cm
            # Timed through __aenter__: without PREOPEN the open (cold boot, farm open gate) happens
            # there, and it is exactly the wait that makes a run sandbox-bound.
            async with cm as sb:
                yield sb, (time.perf_counter() - t0) * 1000.0, None
            return
        async with self._checkout(key, factory) as got:
            yield got

    def _pooled_free(self, key, factory) -> asyncio.Queue:
        q = self._free.get(key)
        if q is None:
            q = self._free[key] = asyncio.Queue()
            asyncio.get_running_loop().create_task(self._open_sessions(key, factory, q))
        return q

    async def _open_sessions(self, key, factory, free: asyncio.Queue) -> None:
        """Open M pooled sessions for ``key`` (the one cold start each), paced under the new-session
        rate (AgentCore allows 25/s). A slot that fails to open shrinks the pool, not the run."""
        from . import router_trace

        base = len(self._slot_episodes)
        opened = 0

        async def _open(i: int) -> None:
            nonlocal opened
            slot = base + i
            self._slot_episodes[slot] = 0
            await asyncio.sleep(i / max(self.open_rate, 0.1))
            t0 = time.perf_counter()
            try:
                sb = factory()
                sb = await sb if asyncio.iscoroutine(sb) else sb
                sb = await sb.__aenter__() or sb
            except Exception as exc:  # noqa: BLE001
                logger.warning("[harness_pool] sandbox slot %d failed to open: %s", slot, exc)
                return
            real = router_trace._unwrap(sb)
            real._pool_root, real._pool_slot = getattr(real, "_workdir", "") or "/workspace", slot
            self._pool_sbx.append(sb)
            opened += 1
            print(f"[demo] session_open=1 sandbox_slot={slot} sandbox_session={router_trace.session_of(sb)} "
                  f"runtime_arn={router_trace.arn_of(sb)} open_ms={(time.perf_counter() - t0) * 1000.0:.1f}",
                  flush=True)
            free.put_nowait(sb)

        await asyncio.gather(*(_open(i) for i in range(self.num_sessions)))
        print(f"[demo] session_pool_ready={opened} of={self.num_sessions} "
              f"key={str(key).replace(' ', '')}", flush=True)
        if not opened:
            free.put_nowait(None)   # wakes waiters; _checkout re-queues it and raises

    @contextlib.asynccontextmanager
    async def _checkout(self, key, factory):
        from . import router_trace

        free = self._pooled_free(key, factory)
        t0 = time.perf_counter()
        sb = await free.get()
        if sb is None:
            free.put_nowait(None)
            raise RuntimeError(f"no pooled sandbox session could be opened for {key!r} (see slot warnings)")
        wait_ms = (time.perf_counter() - t0) * 1000.0
        real = router_trace._unwrap(sb)
        root, slot = real._pool_root, real._pool_slot
        real._workdir = f"{root}/ep-{secrets.token_hex(8)}"   # the server mkdirs it on the first op
        self._slot_episodes[slot] += 1
        failed = False
        try:
            yield sb, wait_ms, slot
        except BaseException:
            failed = True
            raise
        finally:
            ep_dir, real._workdir = real._workdir, root
            alive = True
            try:
                await sb.exec(f"rm -rf {ep_dir}", timeout=60)
            except Exception:  # noqa: BLE001
                alive = not failed   # cleanup failing right after a failed episode = a dead session
            if alive:
                free.put_nowait(sb)
            else:
                asyncio.get_running_loop().create_task(self._replace_session(key, factory, free, sb, slot))

    async def _replace_session(self, key, factory, free: asyncio.Queue, dead, slot) -> None:
        """Retire a dead pooled session (best-effort stop) and open a fresh one into the same slot."""
        from . import router_trace

        with contextlib.suppress(Exception):
            self._pool_sbx.remove(dead)
        with contextlib.suppress(Exception):
            await dead.__aexit__(None, None, None)
        try:
            sb = factory()
            sb = await sb if asyncio.iscoroutine(sb) else sb
            sb = await sb.__aenter__() or sb
        except Exception as exc:  # noqa: BLE001
            logger.warning("[harness_pool] sandbox slot %d is dead and could not be reopened: %s", slot, exc)
            if not self._pool_sbx:
                free.put_nowait(None)   # no session left: wake waiters so they raise
            return
        real = router_trace._unwrap(sb)
        real._pool_root, real._pool_slot = getattr(real, "_workdir", "") or "/workspace", slot
        self._pool_sbx.append(sb)
        print(f"[demo] session_reopen=1 sandbox_slot={slot} sandbox_session={router_trace.session_of(sb)}",
              flush=True)
        free.put_nowait(sb)

    def retire(self) -> None:
        """This pool's loop has stopped: close its pooled sessions in the background (own thread/loop)
        and cancel its idle workers."""
        with contextlib.suppress(Exception):
            atexit.unregister(self._close_sessions_sync)
        if not self.loop.is_closed():
            for t in self._workers:
                with contextlib.suppress(Exception):
                    self.loop.call_soon_threadsafe(t.cancel)
        if self._pool_sbx:
            threading.Thread(target=self._close_sessions_sync, name="sbx-pool-retire", daemon=True).start()

    def slot_episodes(self, slot) -> int:
        return self._slot_episodes.get(slot, 0)

    def _close_sessions_sync(self) -> None:
        """Process exit: stop every pooled session so its microVM is reclaimed now, not at idle timeout."""
        sbxs, self._pool_sbx = list(self._pool_sbx), []
        if not sbxs:
            return

        async def _close_all() -> None:
            await asyncio.gather(*(sb.__aexit__(None, None, None) for sb in sbxs), return_exceptions=True)

        def _run() -> None:
            with contextlib.suppress(Exception):
                asyncio.run(_close_all())

        # Own thread + own loop: works at interpreter exit and while the rollout loop is still running.
        t = threading.Thread(target=_run, name="sbx-pool-close")
        t.start()
        t.join(timeout=120)

    def tier_fields(self, worker: Worker, e: int, wait_ms: float, slot) -> dict:
        """The per-episode tier fields for the [demo] record (which tier binds)."""
        f = {"worker": worker.wid, "qdepth": worker.qdepth, "engine": e,
             "engine_inflight": self._inflight[e], "sandbox_wait_ms": round(wait_ms, 1)}
        if slot is not None:
            f.update(sandbox_slot=slot, session_episode=self._slot_episodes.get(slot, 0))
        return f
