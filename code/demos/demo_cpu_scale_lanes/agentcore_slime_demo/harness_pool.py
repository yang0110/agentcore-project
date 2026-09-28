"""In-process WARM harness worker pool — the harness-scaling demo's core.

This module exists to demonstrate ONE thing: the async-RL rollout scaling law

        throughput  =  min( inference , sandbox , harness )

If you deepen the async rollout buffer and scale inference + sandbox but leave a SINGLE
process driving every episode, the *harness* becomes the ceiling (its concurrency, the
asyncio event loop, FDs, RAM for in-flight trajectories). So the harness has to scale too —
and because a harness is persistent/warm while a sandbox is ad-hoc per episode, "scale the
harness" does NOT mean "make it ad-hoc": it means run a fixed set of WARM workers and route
episodes to them.

DECOUPLED TIERS. The three tiers are sized independently and joined only PER EPISODE — no worker
owns an engine or a microVM. The default is 4 SGLang engines : 256 harness workers : 256 microVM
sessions. When an episode starts, it gets:

  * harness    (W = HARNESS_WORKERS warm asyncio workers, default 256) — whichever worker is free
        PULLS the episode off one shared asyncio.Queue. Pull/work-stealing balances by AVAILABILITY,
        which is robust to the wide, variable latency of multi-turn agentic episodes.
  * inference  (E SGLang engines, E = ROLLOUT_GPUS / GPUS_PER_ENGINE, default 4) — the engine with
        the fewest in-flight episodes, via that engine's AnthropicAdapter (one per engine). The
        episode stays on it for all its turns (prefix-cache locality), but the next episode that
        worker drives can land on any engine.
  * sandbox    (M = SANDBOX_SESSIONS persistent AgentCore sessions = microVMs, default 256) — any
        FREE session from a shared pool, spread over the K Runtime ARNs as ARN[slot % K]. The
        pool opens all M sessions at start (rate-limited to SANDBOX_OPEN_RATE/s, under AgentCore's
        25 new sessions/s), and each episode borrows one for its whole life in a FRESH workdir
        (/workspace/ep-<hex>, deleted when the episode returns the session). SANDBOX_SESSIONS=0 =
        a fresh session per episode, round-robined over the K ARNs (make_sandbox).

Because W and M are separate knobs, either can bind: W < M leaves microVMs idle (harness-bound);
M < W makes workers wait for a free session (`sandbox_wait_ms` > 0, sandbox-bound). Every episode
logs one `[demo] worker=… engine=… sandbox_slot=… sandbox_session=… sandbox_wait_ms=…` record and
one `[router]` record; metrics.py checks each episode stayed on one engine + one session + one
Runtime and reports how the workers and sessions were mixed (the decoupling).

Isolation: a pooled session is reused by episodes ONE AT A TIME (checked out, then returned), each in
its own fresh workdir. That is safe for this demo's DeepScaleR checker, which only reads the
episode's workdir; tasks that change the OS need SANDBOX_SESSIONS=0 (a fresh microVM each).

Placement is IN-PROCESS (asyncio workers in the trainer's rollout process — the demo's chosen
design). Honest ceiling note: in-process workers lift the IO-CONCURRENCY ceiling — episodes spend
almost all wall-clock AWAITING (LLM turns over HTTP to the adapter, tool exec over HTTP to the
sandbox Runtime), which asyncio overlaps well. They do NOT lift the CPU ceiling: the GIL still
serialises the CPU-bound bits (tokenisation / trajectory flattening in the adapter). Lifting THAT
needs separate processes or dedicated harness Runtimes (see docs/harness-scaling.md).

The harness itself is the SAME MiniClaudeHarness (the "mimic Claude Code" agent): it speaks the
Anthropic Messages API to slime's in-process AnthropicAdapter (which serves completions from the
policy under training and records the trajectory), and runs its tools in the remote sandbox
Runtime. So this demo also exercises the adapter / tool-parser path (needs --sglang-tool-call-parser
qwen25 + --sglang-reasoning-parser qwen3).
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

_POOL: "HarnessPool | None" = None


class _NoWait:
    """Adapts a per-episode sandbox to the pooled `(sandbox, wait_ms)` context shape."""

    def __init__(self, sb) -> None:
        self._sb = sb

    async def __aenter__(self):
        return await self._sb.__aenter__(), 0.0

    async def __aexit__(self, *exc):
        return await self._sb.__aexit__(*exc)
_POOL_LOCK = threading.Lock()


def get_pool(args) -> "HarnessPool":
    """Process-wide singleton. Built once from slime `args` on the first episode."""
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = HarnessPool(args)
    return _POOL




class HarnessPool:
    """W warm in-process harness workers, E per-engine adapters and M pooled sandbox sessions, all
    sized independently and joined per episode (see the module docstring)."""

    def __init__(self, args) -> None:
        self.args = args
        self.num_workers = int(os.environ.get("HARNESS_WORKERS", "256"))
        self.max_turns = int(os.environ.get("EPISODE_MAX_TURNS", "4"))
        self.max_tokens = int(os.environ.get("EPISODE_MAX_TOKENS", "1024"))
        self.region = os.environ.get("AGENTCORE_REGION", os.environ.get("AWS_REGION", "us-west-2"))
        self.sandbox_kind = os.environ.get("SANDBOX", "agentcore-runtime")
        # The sandbox tier: M persistent sessions shared by all workers (only for the Runtime backend).
        self.num_sessions = (
            int(os.environ.get("SANDBOX_SESSIONS", "0"))
            if self.sandbox_kind in ("agentcore-runtime", "runtime") else 0
        )
        self.open_rate = float(os.environ.get("SANDBOX_OPEN_RATE", "20"))
        self._queue: asyncio.Queue | None = None
        self._workers: list[asyncio.Task] = []
        self._http = None                  # ONE warm aiohttp session shared by every worker
        self._adapters: list = []          # AnthropicAdapter per ENGINE
        self._adapter_urls: list[str] = [] # its in-thread endpoint URL, per engine
        self._session_urls: list[str] = [] # the same adapter as the sandbox session dials it back
        self._engine_urls: list[str] = []  # engine base URL, per engine
        self._inflight: list[int] = []     # episodes currently on each engine
        self._free: asyncio.Queue | None = None     # free pooled sandbox sessions
        self._pool_sbx: list = []                   # every opened pooled session
        self._slot_episodes: dict[int, int] = {}    # slot -> episodes it has served
        self._opener: asyncio.Task | None = None
        self._harness = None
        self._started = False
        self._start_lock = asyncio.Lock()

    # ---- lifecycle -----------------------------------------------------------
    async def _ensure_started(self) -> None:
        """Stand up the adapters, the W workers and the session pool exactly once, on the calling loop."""
        if self._started:
            return
        async with self._start_lock:
            if self._started:
                return
            import concurrent.futures

            import aiohttp

            from slime.agent.adapters import AnthropicAdapter
            from slime.agent.aiohttp_threaded import run_app_in_thread
            from slime.utils.processing_utils import load_tokenizer

            from . import router_trace
            from .in_session_harness import adapter_bind, adapter_urls, make_harness

            # Every sandbox op is a blocking boto3 call run via asyncio.to_thread, i.e. on this loop's
            # default executor, which holds min(32, cpus+4) threads. With 256 episodes in flight that
            # would quietly cap the sandbox tier at ~32 concurrent ops, so size it to the pool.
            asyncio.get_running_loop().set_default_executor(
                concurrent.futures.ThreadPoolExecutor(
                    max_workers=max(self.num_workers, self.num_sessions) + 32,
                    thread_name_prefix="sbx-op",
                )
            )

            # ONE adapter per SGLang engine (not per worker): its sglang_url is that single engine,
            # so all turns of an episode opened on it hit that engine. Uses only the adapter's
            # PUBLIC sglang_url arg — no slime patch. Falls back to the router if discovery fails.
            router_url = f"http://{self.args.sglang_router_ip}:{self.args.sglang_router_port}"
            engines = router_trace.engine_urls(router_url)
            tokenizer = load_tokenizer(self.args.hf_checkpoint, trust_remote_code=True)
            for i, engine in enumerate(engines):
                adapter = AnthropicAdapter(
                    tokenizer=tokenizer,
                    sglang_url=engine,
                    tool_parser=getattr(self.args, "sglang_tool_call_parser", None) or None,
                    reasoning_parser=getattr(self.args, "sglang_reasoning_parser", None) or None,
                )
                host, port = adapter_bind(i)   # 0.0.0.0:ADAPTER_PORT_BASE+i when the loop runs in the session
                handle = run_app_in_thread(
                    adapter.app, host=host, port=port, thread_name=f"anthropic-adapter-e{i}",
                )
                self._adapters.append(adapter)
                local_url, session_url = adapter_urls(handle.port)
                self._adapter_urls.append(local_url)
                self._session_urls.append(session_url)
                self._engine_urls.append(engine)
            self._inflight = [0] * len(engines)

            if self.num_sessions > 0:
                self._free = asyncio.Queue()
                self._opener = asyncio.create_task(self._open_sessions())
                atexit.register(self._close_sessions_sync)

            # HARNESS_WHERE=session (default on own Runtimes): the loop runs INSIDE the sandbox session
            # and dials the engine's adapter back; the worker here only dispatches and polls.
            self._harness = make_harness(max_turns=self.max_turns)
            self._queue = asyncio.Queue()
            # One warm, unlimited connector for all workers (the default caps at 100 connections).
            self._http = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0))
            for wid in range(self.num_workers):
                self._workers.append(asyncio.create_task(self._worker_loop(wid)))
            self._started = True
            logger.info(
                "[harness_pool] started W=%d workers, E=%d engines, M=%d pooled sessions (decoupled), sandbox=%s",
                self.num_workers, len(engines), self.num_sessions, self.sandbox_kind,
            )

    async def _worker_loop(self, wid: int) -> None:
        assert self._queue is not None
        while True:
            job = await self._queue.get()
            if job is None:  # shutdown sentinel
                self._queue.task_done()
                return
            task, sampling_params, base_sample, fut = job
            try:
                result = await self._run_episode(wid, task, sampling_params, base_sample)
                if not fut.done():
                    fut.set_result(result)
            except Exception as exc:  # noqa: BLE001 — never let one episode kill the worker
                logger.warning("[harness_pool] worker %d episode failed: %s", wid, exc)
                if not fut.done():
                    fut.set_exception(exc)
            finally:
                self._queue.task_done()

    # ---- inference tier: least-loaded engine per episode ---------------------------
    @contextlib.contextmanager
    def _engine(self):
        # Least in-flight engine; ties rotate so an idle start doesn't pile onto engine 0.
        n = len(self._inflight)
        self._rr = (getattr(self, "_rr", -1) + 1) % n
        i = min(range(n), key=lambda j: (self._inflight[j], (j - self._rr) % n))
        self._inflight[i] += 1
        try:
            yield i
        finally:
            self._inflight[i] -= 1

    # ---- sandbox tier: M pooled microVM sessions, shared by all workers -----------
    async def _open_sessions(self) -> None:
        """Open the M pooled sessions (the one cold start each), paced under the new-session rate."""
        import boto3
        from botocore.config import Config

        from . import router_trace
        from .agentcore_runtime_sandbox import AgentCoreRuntimeSandbox
        from .sandbox import _runtime_arns
        from .session_keeper import get_keeper, heartbeat_enabled, heartbeat_idle

        assert self._free is not None
        arns = _runtime_arns() or [os.environ.get("AGENTCORE_SANDBOX_RUNTIME_ARN", "")]
        # ONE thread-safe boto3 client for the whole pool, with a connection pool as wide as the
        # number of ops that can be in flight (the botocore default of 10 would serialise them).
        client = boto3.client(
            "bedrock-agentcore", region_name=self.region,
            config=Config(read_timeout=int(os.environ.get("INVOKE_READ_TIMEOUT", "900")),
                          connect_timeout=60, retries={"max_attempts": 0},
                          max_pool_connections=max(self.num_workers, self.num_sessions) + 16),
        )

        async def _open(slot: int) -> None:
            await asyncio.sleep(slot / max(self.open_rate, 0.1))
            arn = arns[slot % len(arns)] or None
            # >=33 chars for AgentCore runtimeSessionId (token_hex(16)=32 + prefix clears it).
            sb = AgentCoreRuntimeSandbox(runtime_arn=arn, region=self.region, client=client,
                                         session_id=f"pool{slot}-{secrets.token_hex(16)}")
            t0 = time.perf_counter()
            try:
                await sb.__aenter__()
            except Exception as exc:  # noqa: BLE001 — a failed slot shrinks the pool, not the run
                logger.warning("[harness_pool] sandbox slot %d failed to open: %s", slot, exc)
                return
            sb._pool_root, sb._pool_slot = sb._workdir, slot
            self._pool_sbx.append(sb)
            self._slot_episodes[slot] = 0
            # HEARTBEAT=1: a pooled microVM lives for the whole run, so it is exactly what the
            # weight-sync stall can idle-reclaim. Register it once with the process-wide keeper.
            if heartbeat_enabled():
                async def _ping(s=sb) -> None:
                    await s.exec("true", timeout=30)
                (await get_keeper()).register(sb._session_id, _ping, heartbeat_idle())
            print(f"[demo] session_open=1 sandbox_slot={slot} sandbox_session={sb._session_id} "
                  f"runtime_arn={router_trace.arn_tail(arn)} "
                  f"open_ms={(time.perf_counter() - t0) * 1000.0:.1f}", flush=True)
            self._free.put_nowait(sb)

        await asyncio.gather(*(_open(i) for i in range(self.num_sessions)))
        print(f"[demo] session_pool_ready={len(self._pool_sbx)} of={self.num_sessions}", flush=True)
        if not self._pool_sbx:
            self._free.put_nowait(None)   # wakes waiters; _checkout re-queues it and raises

    @contextlib.asynccontextmanager
    async def _checkout(self):
        """Borrow ANY free pooled session for one episode, in a FRESH workdir removed on return."""
        assert self._free is not None
        t0 = time.perf_counter()
        sb = await self._free.get()
        if sb is None:
            self._free.put_nowait(None)
            raise RuntimeError("no sandbox session could be opened (see the slot warnings above)")
        wait_ms = (time.perf_counter() - t0) * 1000.0
        root = sb._pool_root
        sb._workdir = f"{root}/ep-{secrets.token_hex(6)}"   # server mkdirs it on the first op
        self._slot_episodes[sb._pool_slot] += 1
        try:
            yield sb, wait_ms
        finally:
            ep_dir, sb._workdir = sb._workdir, root
            with contextlib.suppress(Exception):
                await sb.exec(f"rm -rf {ep_dir}", timeout=60)
            self._free.put_nowait(sb)

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

    # ---- the episode --------------------------------------------------------
    def _problem_text(self, task) -> str:
        """PROBLEM.md the agent reads. We put the task-specific instructions HERE (not in the
        harness's fixed system prompt) so the coding-shape mini harness stays unchanged: the
        family's own system prompt (e.g. math's 'store the answer in `answer`') + the question."""
        return f"{task.system_prompt()}\n\n{task.initial_user_prompt()}\n"

    _AGENT_PROMPT = (
        "Read PROBLEM.md in this directory and follow it. Write your solution to solution.py using "
        "the write_file tool, then run `python3 run_checks.py` with the bash tool and read its "
        "output. If any check fails, revise solution.py and run the checks again. Repeat until the "
        "command exits 0."
    )

    async def _run_episode(self, wid: int, task, sampling_params, base_sample):
        from . import router_trace
        from .sandbox import make_sandbox

        # Adapter Bearer session id == the trajectory key; AgentCore runtimeSessionId needs len>=33.
        sid = getattr(base_sample, "session_id", None)
        if not sid or len(sid) < 33:
            sid = f"hs-{task.task_id}-{secrets.token_hex(16)}"
        base_sample.session_id = sid

        # Sandbox FIRST (any free pooled session, fresh workdir; or a fresh session per episode when
        # SANDBOX_SESSIONS=0), so an episode waiting for a session does not count as engine load.
        # Either way the sandbox is sticky for the episode's life (it holds the files).
        if self.num_sessions > 0:
            sandbox = self._checkout()
        else:
            from .session_keeper import maybe_keep_warm
            from . import preopen   # PREOPEN=N (opt-in): N fresh sessions booted ahead of demand
            sandbox = _NoWait(await preopen.take(self.sandbox_kind, lambda: maybe_keep_warm(
                make_sandbox(self.sandbox_kind, region=self.region)), kind=self.sandbox_kind))
        async with sandbox as (sb, wait_ms):
            with self._engine() as e:
                adapter = self._adapters[e]
                adapter_url = self._adapter_urls[e]
                engine = self._engine_urls[e]
                adapter.open_session(sid, sampling_defaults=sampling_params)
                try:
                    slot = getattr(sb, "_pool_slot", None)
                    # ONE [router] record per episode: the join that ties this episode's sandbox
                    # session (also stamped on every [sbx-op]) to the worker that drove it and the
                    # engine it ran on. metrics.py checks each episode is sticky to all three.
                    router_trace.emit(
                        task_id=task.task_id,
                        rollout_id=getattr(base_sample, "index", "?"),
                        harness_worker=wid,
                        engine=engine,
                        engine_pinned=int(engine != ""),
                        adapter_session=sid,
                        sandbox_id=getattr(sb, "sandbox_id", "?"),
                        sandbox_session=getattr(sb, "_session_id", getattr(sb, "_sub", "?")),
                        sandbox_slot=slot,
                        runtime_arn=router_trace.arn_tail(getattr(sb, "_arn", None)),
                    )
                    # DEMO-SPECIFIC metric ([demo] channel): which worker, engine and pooled session
                    # this episode drew, how long it waited for a free session (sandbox-bound when
                    # > 0) and the harness queue depth at pickup (harness-bound when > 0).
                    pooled = (f" sandbox_slot={slot} session_episode={self._slot_episodes[slot]}"
                              if slot is not None else "")
                    print(f"[demo] worker={wid} engine={e} engine_inflight={self._inflight[e]} "
                          f"sandbox_session={getattr(sb, '_session_id', '?')}{pooled} "
                          f"sandbox_wait_ms={wait_ms:.1f} "
                          f"qdepth={self._queue.qsize() if self._queue else 0}", flush=True)

                    # Seed the workspace exactly like run_episode: family files, then the checker,
                    # then PROBLEM.md — all RELATIVE paths (resolved against the sandbox workdir).
                    for name, content in getattr(task, "setup_files", lambda: {})().items():
                        await sb.write_file(name, content)
                    await sb.write_file("run_checks.py", task.runner_source())
                    await sb.write_file("PROBLEM.md", self._problem_text(task))

                    # The WARM harness drives the write->run->fix loop over the shared http session.
                    res = await self._harness.run(
                        sb,
                        session_id=sid,
                        adapter_url=(self._session_urls[e] if getattr(self._harness, "in_session", False)
                                     else adapter_url),
                        local_adapter_url=adapter_url,
                        prompt=self._AGENT_PROMPT,
                        max_tokens=self.max_tokens,
                        http=self._http,
                    )

                    # Score with a HIDDEN copy of the checker (RLVR — reward from verification, no LLM
                    # judge): setup files re-seeded, checker under a random name, nonce'd PASSED line,
                    # so rewriting the visible run_checks.py or sys.exit(0) in solution.py earns nothing.
                    from .mini_claude_episode import _hidden_grade
                    exit_code, _out = await _hidden_grade(sb, task)
                    reward = task.reward(int(exit_code))

                    # Drain the adapter-captured trajectory into loss-masked GRPO samples (a LIST).
                    return await adapter.finish_session(
                        sid, base_sample=base_sample, reward=float(reward),
                        extra_metadata={"task_id": task.task_id, "solved": reward == 1.0,
                                        "harness_worker": wid, "engine": e, "sandbox_slot": slot,
                                        "harness_where": res.where},
                    )
                finally:
                    await adapter.drop_session(sid, wait_timeout=30)

    # ---- public entry point -------------------------------------------------
    async def submit(self, task, sampling_params, base_sample):
        """Enqueue one episode and await the finished Sample list (blocks until a worker frees)."""
        await self._ensure_started()
        assert self._queue is not None
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        await self._queue.put((task, sampling_params, base_sample, fut))
        return await fut
