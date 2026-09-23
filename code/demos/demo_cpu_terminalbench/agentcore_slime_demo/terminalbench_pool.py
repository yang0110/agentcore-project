"""Isolated harness pool for demo_cpu_terminalbench — ONE FRESH microVM PER EPISODE, pre-opened.

The demo's whole point is a LARGE image pool (one container image / one AgentCore Runtime per
Terminal-Bench task). Every episode must be ISOLATED from every other: Terminal-Bench tasks act on
the whole OS (apt install, /etc, background services, /tmp), so a shared microVM with a reset
workdir would leak state between episodes. The isolation boundary AgentCore actually guarantees is
the SESSION: one runtimeSessionId = one dedicated microVM, terminated + memory-sanitized when the
session is stopped. So this pool gives every episode its OWN brand-new session and never reuses one:

  1. Strict per-task ROUTING: the episode's task.image (the task name) is looked up in
     AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE. A missing task is an error — never a fallback to some
     other task's Runtime (that would run the episode against the wrong image).
  2. Fresh session per episode, opened AHEAD of time: the cold start (allocate microVM + pull the
     task's image + boot) starts the moment the episode is QUEUED (submit), not when a harness
     worker picks it up. While the episode waits for a free worker, its microVM boots in the
     background, so with a queue backlog (CONC > HARNESS_WORKERS) the cold start is off the critical
     path. run.sh admits HARNESS_WORKERS + TB_PREOPEN_MAX episodes so that backlog exists; at most
     TB_PREOPEN_MAX sessions are open ahead (default HARNESS_WORKERS), bounding the idle microVMs you
     pay for — beyond that an episode opens its session when a worker picks it up.
  3. Stop on exit: StopRuntimeSession when the episode ends (AgentCoreRuntimeSandbox.__aexit__),
     so the microVM is torn down immediately and never serves another episode.
  4. A process-wide SessionKeeper HEARTBEAT holds every open session (pre-opened or running) alive
     across slime's trainer->inference weight-sync stall, when generation goes silent and AgentCore
     would otherwise idle-reclaim the microVM mid-episode.

This subclasses the harness-scaling demo's HarnessPool (W warm in-process workers + one adapter per
worker, so an episode is sticky to its worker => sticky to one SGLang engine) and swaps only the
sandbox tier. The agentic loop is the SAME MiniClaudeHarness (Anthropic Messages API ->
AnthropicAdapter -> policy under training; tools run in the sandbox), so this exercises the faithful
tool-calling path (needs --sglang-tool-call-parser qwen25 + --sglang-reasoning-parser qwen3).

Wire with:
    --custom-generate-function-path agentcore_slime_demo.terminalbench_pool_generate.generate
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import secrets
import threading
import time

from .harness_pool import HarnessPool

logger = logging.getLogger(__name__)

_TB_POOL: "TerminalBenchPool | None" = None
_TB_LOCK = threading.Lock()


def get_tb_pool(args) -> "TerminalBenchPool":
    """Process-wide singleton, built once from slime `args` on the first episode."""
    global _TB_POOL
    with _TB_LOCK:
        if _TB_POOL is None:
            _TB_POOL = TerminalBenchPool(args)
    return _TB_POOL


class TerminalBenchPool(HarnessPool):
    """HarnessPool whose sandbox tier is a fresh, pre-opened session per episode + a heartbeat."""

    # Terminal-Bench tasks act through the shell, not by writing a solution.py that a checker runs.
    # Override the coding-shaped prompt so the agent uses its tools against the working directory.
    _AGENT_PROMPT = (
        "Read PROBLEM.md in the current directory and complete the task it describes. Use the bash "
        "tool to inspect the environment and make changes, and write_file to create or edit files. "
        "Verify your work with shell commands as you go. When the task is complete, stop."
    )

    def __init__(self, args) -> None:
        super().__init__(args)
        self.sandbox_kind = "agentcore-runtime"
        self.workdir = os.environ.get("TB_WORKDIR", "/app")   # the task's own baked workdir
        # Serverless microVM idle timeout is ~15 min; heartbeat at half that with margin.
        self.idle_timeout = float(os.environ.get("WARM_IDLE_TIMEOUT", "900"))
        self.preopen_max = int(os.environ.get("TB_PREOPEN_MAX", str(self.num_workers)))
        # Strict task -> Runtime ARN map (the image pool), parsed once.
        self._arn_by_image: dict[str, str] = json.loads(
            os.environ.get("AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE", "").strip() or "{}"
        )
        self._keeper = None
        # id(base_sample) -> in-flight open of that episode's own session (started at submit).
        self._preopen: dict[int, asyncio.Task] = {}
        # Queued episodes still waiting for a pre-open slot, in submit order: id(base_sample) -> arn.
        self._waiting: dict[int, str] = {}

    # ---- lifecycle: also stand up the heartbeat keeper -----------------------
    async def _ensure_started(self) -> None:
        first = not self._started
        await super()._ensure_started()
        if first:
            from .session_keeper import SessionKeeper

            self._keeper = await SessionKeeper(
                ping_fraction=float(os.environ.get("WARM_PING_FRACTION", "0.5")),
                tick=float(os.environ.get("WARM_KEEPER_TICK", "30.0")),
            ).start()
            # Always-on idle heartbeat: hold any open session that goes idle (e.g. across the
            # weight-sync stall) without gating real ops.
            self._keeper.keep_warm_begin()
            logger.info("[terminalbench_pool] SessionKeeper started (idle_timeout=%.0fs, preopen_max=%d)",
                        self.idle_timeout, self.preopen_max)

    def _arn_for(self, task) -> str:
        image = getattr(task, "image", task.task_id)
        arn = self._arn_by_image.get(image)
        if not arn:
            raise RuntimeError(
                f"no Runtime ARN for task {task.task_id!r} (image {image!r}) in "
                "AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE — deploy it (deploy_terminalbench.py) or "
                "drop it from the dataset. Refusing to run it on another task's Runtime."
            )
        return arn

    async def _open_session(self, arn: str):
        """Open one BRAND-NEW session (= its own microVM) on the task's Runtime; heartbeat it."""
        from .agentcore_runtime_sandbox import AgentCoreRuntimeSandbox

        # >=33 chars for AgentCore runtimeSessionId (token_hex(16)=32 + prefix clears it).
        sb = AgentCoreRuntimeSandbox(
            runtime_arn=arn, region=self.region, workdir=self.workdir,
            session_id=f"tbep-{secrets.token_hex(16)}",
        )
        await sb.__aenter__()          # the cold start: allocate + pull + boot + ping

        async def _ping(s=sb) -> None:
            # A raw no-op invoke on the session resets AgentCore's idle clock.
            await s.exec("true", timeout=30)

        self._keeper.register(sb._session_id, _ping, self.idle_timeout)
        return sb

    def _refill(self) -> None:
        """Start pre-opening the oldest queued episodes until TB_PREOPEN_MAX are booting ahead."""
        while self._waiting and len(self._preopen) < self.preopen_max:
            key = next(iter(self._waiting))
            self._preopen[key] = asyncio.create_task(self._open_session(self._waiting.pop(key)))

    async def _close_session(self, sb) -> None:
        self._keeper.unregister(sb._session_id)
        await sb.__aexit__(None, None, None)   # StopRuntimeSession: the microVM is torn down

    # ---- the episode: its own fresh session on THIS task's Runtime -----------
    async def _run_episode(self, wid: int, http, task, sampling_params, base_sample):
        from . import router_trace
        from .agentcore_runtime_sandbox import _arn_tail

        adapter = self._adapters[wid]
        adapter_url = self._adapter_urls[wid]
        engine = self._engine_urls[wid]

        # Adapter Bearer session id == trajectory key.
        sid = getattr(base_sample, "session_id", None)
        if not sid or len(sid) < 33:
            sid = f"tb-{task.task_id}-{secrets.token_hex(16)}"
        base_sample.session_id = sid

        arn = self._arn_for(task)
        pending = self._preopen.pop(id(base_sample), None)
        self._waiting.pop(id(base_sample), None)
        self._refill()                      # this slot is free: start the next queued episode's boot
        t0 = time.perf_counter()
        sb = await (pending if pending is not None else self._open_session(arn))
        wait_ms = (time.perf_counter() - t0) * 1000.0

        adapter.open_session(sid, sampling_defaults=sampling_params)
        try:
            router_trace.emit(
                task_id=task.task_id,
                rollout_id=getattr(base_sample, "index", "?"),
                harness_worker=wid,
                engine=engine,
                engine_pinned=int(len(self._engine_urls) > 0 and engine != ""),
                adapter_session=sid,
                sandbox_id=getattr(sb, "sandbox_id", "?"),
                sandbox_session=sb._session_id,
                runtime_arn=_arn_tail(arn),
            )
            # DEMO-SPECIFIC metric ([demo] channel): every episode is on its OWN fresh session;
            # `preopened` says whether its cold start was started at submit, and `open_wait_ms` is
            # how much of that cold start the worker still had to wait for (the critical-path cost).
            print(f"[demo] worker={wid} qdepth={self._queue.qsize() if self._queue else 0} "
                  f"fresh_session=1 preopened={int(pending is not None)} "
                  f"open_wait_ms={wait_ms:.1f} arn={_arn_tail(arn)}", flush=True)

            # Seed only the instruction; the task's environment is already baked into the image.
            # No run_checks.py (the oracle is the baked grade.sh).
            await sb.write_file("PROBLEM.md", self._problem_text(task))

            await self._harness.run(
                sb,
                session_id=sid,
                adapter_url=adapter_url,
                prompt=self._AGENT_PROMPT,
                max_tokens=self.max_tokens,
                http=http,
            )

            # RLVR reward: the task's OWN hidden pytest oracle, injected + run by grade.sh.
            exit_code, _out, _err = await sb.exec(
                getattr(task, "check_command", "bash /opt/acs/oracle/grade.sh"), timeout=600,
            )
            reward = task.reward(int(exit_code))

            return await adapter.finish_session(
                sid, base_sample=base_sample, reward=float(reward),
                extra_metadata={"task_id": task.task_id, "solved": reward == 1.0,
                                "harness_worker": wid, "sandbox_session": sb._session_id},
            )
        finally:
            await adapter.drop_session(sid, wait_timeout=30)
            await self._close_session(sb)

    # ---- public entry point: start the episode's cold start at enqueue time --
    async def submit(self, task, sampling_params, base_sample):
        await self._ensure_started()
        arn = self._arn_for(task)          # fail fast on an unmapped task, before queueing
        self._waiting[id(base_sample)] = arn
        self._refill()
        try:
            return await super().submit(task, sampling_params, base_sample)
        finally:
            # Only left here if the episode failed before a worker claimed it: don't leak it.
            self._waiting.pop(id(base_sample), None)
            leftover = self._preopen.pop(id(base_sample), None)
            if leftover is not None:
                with contextlib.suppress(Exception):
                    await self._close_session(await leftover)

    # ---- shutdown: stop any still-pre-opened sessions + the keeper -----------
    async def aclose(self) -> None:
        for pending in list(self._preopen.values()):
            with contextlib.suppress(Exception):
                await self._close_session(await pending)
        self._preopen.clear()
        self._waiting.clear()
        if self._keeper is not None:
            await self._keeper.aclose()
