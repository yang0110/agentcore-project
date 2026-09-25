"""In-process WARM harness worker pool — the harness-scaling demo's core.

This module exists to demonstrate ONE thing: the async-RL rollout scaling law

        throughput  =  min( inference , sandbox , harness )

If you deepen the async rollout buffer and scale inference + sandbox but leave a SINGLE
process driving every episode, the *harness* becomes the ceiling (its concurrency, the
asyncio event loop, FDs, RAM for in-flight trajectories). So the harness has to scale too —
and because a harness is persistent/warm while a sandbox is ad-hoc per episode, "scale the
harness" does NOT mean "make it ad-hoc": it means run a fixed set of WARM workers and route
episodes to them.

The three replicated tiers scale AND load-balance DIFFERENTLY — this is the whole point of
the demo, and the asymmetry is deliberate:

  LANES. The default topology binds the three tiers into W fixed 1:1:1 LANES — worker w is pinned
  to SGLang engine w % E and to its own persistent microVM — so with W = E = 4 the rollout runs on
  4 harness workers : 4 engines : 4 microVMs. Each episode logs one `[demo] lane=...` record
  (worker -> engine -> sandbox session) that metrics.py checks, so the 4x4x4 is verifiable.


  * inference  (E SGLang engines)  — STATELESS per request. slime's SGLang ROUTER load-balances
        every `/generate` call across the E engines (cache-aware). A single episode's turns may
        land on different engines. LB granularity = per REQUEST.
  * sandbox    (W microVMs)        — STATEFUL per episode: a sandbox holds the episode's files
        (solution.py, run_checks.py). In LANE mode (the default, LANE_SANDBOX=1, SANDBOX=
        agentcore-runtime) each worker owns ONE persistent AgentCore session (= one microVM) on
        Runtime ARN[w % K], opened once and reused for every episode that worker drives; each
        episode gets its own fresh workdir under it (/workspace/ep-<hex>, deleted after). So W
        workers => exactly W concurrent microVMs, and an episode is sticky to its worker's microVM.
        LANE_SANDBOX=0 restores the old shape: a fresh session per episode, round-robined across
        the K ARNs (make_sandbox -> _next_runtime_arn).
  * harness    (W warm workers)    — STATELESS but WARM. A shared asyncio.Queue; whichever worker
        is free PULLS the next episode. Pull/work-stealing balances by AVAILABILITY, which is
        robust to the wide, variable per-episode latency of multi-turn agentic rollouts (a push /
        hash assignment would hot-spot a worker stuck on a slow episode). LB granularity = per
        EPISODE, by pull. A worker holds its episode start->finish (it owns the aiohttp session and
        the adapter session id), so an episode is sticky to its worker just as it is to its sandbox.

Placement is IN-PROCESS (asyncio workers in the trainer process — the demo's chosen design).
Honest ceiling note: in-process workers lift the IO-CONCURRENCY ceiling — episodes spend almost
all wall-clock AWAITING (LLM turns over HTTP to the adapter, tool exec over HTTP to the sandbox
Runtime), which asyncio overlaps well. They do NOT lift the CPU ceiling: the GIL still serialises
the CPU-bound bits (tokenisation / trajectory flattening in the adapter's finish_session). Lifting
THAT needs separate processes or dedicated harness Runtimes — a different placement, out of scope
for this demo (see docs/harness-scaling.md).

The harness itself is the SAME MiniClaudeHarness (the "mimic Claude Code" agent): it speaks the
Anthropic Messages API to slime's in-process AnthropicAdapter (which serves completions from the
policy under training and records the trajectory), and runs its tools in the remote sandbox
Runtime. So this demo also exercises the adapter / tool-parser path (needs --sglang-tool-call-parser
qwen25 + --sglang-reasoning-parser qwen3) that the in-process run_episode harness sidesteps.
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
_POOL_LOCK = threading.Lock()


def get_pool(args) -> "HarnessPool":
    """Process-wide singleton. Built once from slime `args` on the first episode."""
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = HarnessPool(args)
    return _POOL


class HarnessPool:
    """W warm in-process harness workers fed by one shared asyncio.Queue.

    One shared AnthropicAdapter (localhost, in-process) fronts SGLang for all workers; each
    worker owns a warm aiohttp session it reuses across episodes and a fresh remote sandbox
    session per episode.
    """

    def __init__(self, args) -> None:
        self.args = args
        # THE scaling knob. Default 4 to sit symmetric with the demo's 4 inference engines and
        # 4 sandbox Runtimes (throughput = min over the three tiers; all-4 is the balanced point).
        self.num_workers = int(os.environ.get("HARNESS_WORKERS", "4"))
        self.max_turns = int(os.environ.get("EPISODE_MAX_TURNS", "4"))
        self.max_tokens = int(os.environ.get("EPISODE_MAX_TOKENS", "1024"))
        self.region = os.environ.get("AGENTCORE_REGION", os.environ.get("AWS_REGION", "us-west-2"))
        self.sandbox_kind = os.environ.get("SANDBOX", "agentcore-runtime")
        self._queue: asyncio.Queue | None = None
        self._workers: list[asyncio.Task] = []
        self._sessions: list = []          # per-worker warm aiohttp sessions
        # PER-WORKER adapters, one each, so a worker's episodes are PINNED to one SGLang engine
        # (see _ensure_started). Was one shared adapter; per-worker pinning makes inference sticky
        # per episode (an episode is sticky to its worker => sticky to that worker's engine).
        self._adapters: list = []          # AnthropicAdapter per worker
        self._adapter_urls: list[str] = [] # its in-thread endpoint URL, per worker
        self._engine_urls: list[str] = []  # the SGLang engine each worker is pinned to, per worker
        # LANE mode: one persistent sandbox session (= one microVM) per worker, reused across that
        # worker's episodes with a fresh per-episode workdir. Only for the AgentCore Runtime backend.
        self.lane_sandbox = (
            os.environ.get("LANE_SANDBOX", "1") not in ("0", "false", "no")
            and self.sandbox_kind in ("agentcore-runtime", "runtime")
        )
        self._lane_sbx: dict[int, object] = {}      # wid -> its open AgentCoreRuntimeSandbox
        self._lane_episodes: dict[int, int] = {}    # wid -> episodes driven on that microVM
        self._harness = None
        self._started = False
        self._start_lock = asyncio.Lock()

    # ---- lifecycle -----------------------------------------------------------
    async def _ensure_started(self) -> None:
        """Stand up the adapter + the W warm workers exactly once, on the calling loop."""
        if self._started:
            return
        async with self._start_lock:
            if self._started:
                return
            import aiohttp

            from slime.agent.adapters import AnthropicAdapter
            from slime.agent.aiohttp_threaded import run_app_in_thread
            from slime.utils.processing_utils import load_tokenizer

            from . import router_trace
            from .mini_agent import MiniClaudeHarness

            router_url = f"http://{self.args.sglang_router_ip}:{self.args.sglang_router_port}"
            # Discover the E SGLang engines behind the router ONCE, then PIN worker w to engine
            # w % E. slime's router normally load-balances /generate per request across engines;
            # by giving each worker its OWN adapter whose sglang_url is a single engine, every turn
            # of every episode that worker drives hits that one engine — inference becomes sticky
            # per episode (episode is sticky to its worker). Falls back to the router if discovery
            # fails (unpinned, still recorded). Uses only the adapter's PUBLIC sglang_url arg — no
            # slime patch.
            engines = router_trace.engine_urls(router_url)
            tokenizer = load_tokenizer(self.args.hf_checkpoint, trust_remote_code=True)

            # PER-WORKER adapter, each on its own localhost thread/port. Workers dial 127.0.0.1, so
            # no security group / public port is needed (unlike the harness-Runtime variant).
            for wid in range(self.num_workers):
                engine = engines[wid % len(engines)]
                adapter = AnthropicAdapter(
                    tokenizer=tokenizer,
                    sglang_url=engine,
                    tool_parser=getattr(self.args, "sglang_tool_call_parser", None) or None,
                    reasoning_parser=getattr(self.args, "sglang_reasoning_parser", None) or None,
                )
                handle = run_app_in_thread(
                    adapter.app, host="127.0.0.1", port=0,
                    thread_name=f"anthropic-adapter-w{wid}",
                )
                self._adapters.append(adapter)
                self._adapter_urls.append(f"http://127.0.0.1:{handle.port}")
                self._engine_urls.append(engine)

            self._harness = MiniClaudeHarness(max_turns=self.max_turns)
            self._queue = asyncio.Queue()
            for wid in range(self.num_workers):
                sess = aiohttp.ClientSession()   # created on THIS loop; reused across episodes (warm)
                self._sessions.append(sess)
                self._workers.append(asyncio.create_task(self._worker_loop(wid, sess)))
            self._started = True
            logger.info(
                "[harness_pool] started W=%d warm workers over E=%d engines (pinned w->w%%E), sandbox=%s",
                self.num_workers, len(engines), self.sandbox_kind,
            )

    async def _worker_loop(self, wid: int, http) -> None:
        assert self._queue is not None
        while True:
            job = await self._queue.get()
            if job is None:  # shutdown sentinel
                self._queue.task_done()
                return
            task, sampling_params, base_sample, fut = job
            try:
                result = await self._run_episode(wid, http, task, sampling_params, base_sample)
                if not fut.done():
                    fut.set_result(result)
            except Exception as exc:  # noqa: BLE001 — never let one episode kill the worker
                logger.warning("[harness_pool] worker %d episode failed: %s", wid, exc)
                if not fut.done():
                    fut.set_exception(exc)
            finally:
                self._queue.task_done()

    # ---- lanes: one persistent microVM per worker ------------------------------
    async def _lane_sandbox(self, wid: int):
        """This worker's persistent sandbox session, opened on first use (the one cold start)."""
        sb = self._lane_sbx.get(wid)
        if sb is not None:
            return sb
        from . import router_trace
        from .agentcore_runtime_sandbox import AgentCoreRuntimeSandbox
        from .sandbox import _runtime_arns

        arns = _runtime_arns() or [os.environ.get("AGENTCORE_SANDBOX_RUNTIME_ARN", "")]
        arn = arns[wid % len(arns)] or None
        # >=33 chars for AgentCore runtimeSessionId (token_hex(16)=32 + prefix clears it).
        sb = AgentCoreRuntimeSandbox(runtime_arn=arn, region=self.region,
                                     session_id=f"lane{wid}-{secrets.token_hex(16)}")
        t0 = time.perf_counter()
        await sb.__aenter__()
        sb._lane_root = sb._workdir
        self._lane_sbx[wid] = sb
        self._lane_episodes[wid] = 0
        # HEARTBEAT=1: this microVM lives for the whole run, so it is exactly what the weight-sync
        # stall can idle-reclaim. Register it once with the process-wide keeper (raw no-op pinger);
        # real episode ops bypass the keeper's clock, so it simply pings every
        # HEARTBEAT_PING_FRACTION*HEARTBEAT_IDLE seconds (one cheap no-op every few minutes).
        from .session_keeper import get_keeper, heartbeat_enabled, heartbeat_idle
        if heartbeat_enabled():
            async def _ping(s=sb) -> None:
                await s.exec("true", timeout=30)
            (await get_keeper()).register(sb._session_id, _ping, heartbeat_idle())
        if len(self._lane_sbx) == 1:
            atexit.register(self._close_lanes_sync)
        print(f"[demo] lane_open=1 lane={wid} engine={self._engine_urls[wid]} "
              f"sandbox_session={sb._session_id} runtime_arn={router_trace.arn_tail(arn)} "
              f"open_ms={(time.perf_counter() - t0) * 1000.0:.1f}", flush=True)
        return sb

    @contextlib.asynccontextmanager
    async def _lane_episode(self, wid: int):
        """Borrow the worker's microVM for one episode, in a FRESH workdir that is removed after."""
        sb = await self._lane_sandbox(wid)
        root = sb._lane_root
        sb._workdir = f"{root}/ep-{secrets.token_hex(6)}"   # server mkdirs it on the first op
        self._lane_episodes[wid] += 1
        try:
            yield sb
        finally:
            ep_dir, sb._workdir = sb._workdir, root
            with contextlib.suppress(Exception):
                await sb.exec(f"rm -rf {ep_dir}", timeout=60)

    def _close_lanes_sync(self) -> None:
        """Process exit: stop every lane session so its microVM is reclaimed now, not at idle timeout."""
        for sb in list(self._lane_sbx.values()):
            with contextlib.suppress(Exception):
                asyncio.run(sb.__aexit__(None, None, None))
        self._lane_sbx.clear()

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

    async def _run_episode(self, wid: int, http, task, sampling_params, base_sample):
        from . import router_trace
        from .sandbox import make_sandbox

        adapter = self._adapters[wid]
        adapter_url = self._adapter_urls[wid]
        engine = self._engine_urls[wid]

        # Adapter Bearer session id == the trajectory key; AgentCore runtimeSessionId needs len>=33.
        sid = getattr(base_sample, "session_id", None)
        if not sid or len(sid) < 33:
            sid = f"hs-{task.task_id}-{secrets.token_hex(16)}"
        base_sample.session_id = sid
        adapter.open_session(sid, sampling_defaults=sampling_params)

        # LANE mode: this worker's own persistent microVM, fresh workdir per episode. Otherwise a
        # fresh sandbox session per episode, round-robined across the K ARNs by make_sandbox.
        # Either way the sandbox is sticky for the episode's life (it holds the episode's files).
        if self.lane_sandbox:
            sandbox = self._lane_episode(wid)
        else:
            from .session_keeper import maybe_keep_warm
            sandbox = await maybe_keep_warm(make_sandbox(self.sandbox_kind, region=self.region))
        try:
            async with sandbox as sb:
                # ONE [router] record per episode: the join that ties this episode's sandbox
                # session (also stamped on every [sbx-op]) to the worker that drove it and the
                # engine it was pinned to. metrics.py checks these are all sticky per episode.
                router_trace.emit(
                    task_id=task.task_id,
                    rollout_id=getattr(base_sample, "index", "?"),
                    harness_worker=wid,
                    engine=engine,
                    engine_pinned=int(len(self._engine_urls) > 0 and engine != ""),
                    adapter_session=sid,
                    sandbox_id=getattr(sb, "sandbox_id", "?"),
                    sandbox_session=getattr(sb, "_session_id", getattr(sb, "_sub", "?")),
                    runtime_arn=router_trace.arn_tail(getattr(sb, "_arn", None)),
                    lane=wid if self.lane_sandbox else None,
                )
                # DEMO-SPECIFIC metric ([demo] channel): which warm worker drove this episode +
                # the shared queue depth at pickup time. metrics.py value-counts `worker` (per-worker
                # episode counts = utilization: the pull-queue balances by availability, no hot-spot)
                # and aggregates `qdepth` (backlog waiting on harness capacity). This episode does not
                # go through harness.run_episode, so this is its only [demo] record.
                # In LANE mode the same record also carries the lane mapping (worker -> engine ->
                # microVM session) and how many episodes that microVM has served so far.
                lane = (f" lane={wid} engine={engine} sandbox_session={sb._session_id} "
                        f"lane_episode={self._lane_episodes[wid]}" if self.lane_sandbox else "")
                print(f"[demo] worker={wid} qdepth={self._queue.qsize() if self._queue else 0}{lane}",
                      flush=True)

                # Seed the workspace exactly like run_episode: family files, then the checker,
                # then PROBLEM.md — all RELATIVE paths (resolved against the sandbox workdir).
                for name, content in getattr(task, "setup_files", lambda: {})().items():
                    await sb.write_file(name, content)
                await sb.write_file("run_checks.py", task.runner_source())
                await sb.write_file("PROBLEM.md", self._problem_text(task))

                # The WARM harness drives the write->run->fix loop, reusing this worker's session.
                await self._harness.run(
                    sb,
                    session_id=sid,
                    adapter_url=adapter_url,
                    prompt=self._AGENT_PROMPT,
                    max_tokens=self.max_tokens,
                    http=http,
                )

                # Score with the SAME checker (RLVR — reward from verification, no LLM judge).
                exit_code, _out, _err = await sb.exec(
                    getattr(task, "check_command", "python3 run_checks.py"), timeout=120,
                )
                reward = task.reward(int(exit_code))

            # Drain the adapter-captured trajectory into loss-masked GRPO samples (a LIST).
            return await adapter.finish_session(
                sid, base_sample=base_sample, reward=float(reward),
                extra_metadata={"task_id": task.task_id, "solved": reward == 1.0, "harness_worker": wid},
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
