"""Isolated harness pool for Terminal-Bench rows — ONE FRESH microVM PER EPISODE, pre-opened.

Terminal-Bench is a LARGE image pool (one container image / one AgentCore Runtime per task; each
row names its task's image). Every episode must be ISOLATED from every other: Terminal-Bench tasks act on
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
     background. Better still, the look-ahead data source (tb_lookahead_source.py) tells the pool
     which episodes the NEXT TB_PREOPEN_BATCHES rollouts will hold before slime even draws them, so
     their sessions boot while the current batch runs. Sessions are pooled per task (an episode
     takes any ready session on its task's Runtime); at most TB_PREOPEN_BATCHES x RB x NS unclaimed
     sessions are open ahead (default 2 x 4 x 8 = 64), bounding the idle microVMs you pay for —
     beyond that an episode opens its session when a worker picks it up.
  3. Stop on exit: StopRuntimeSession when the episode ends (AgentCoreRuntimeSandbox.__aexit__),
     so the microVM is torn down immediately and never serves another episode.
  4. A process-wide SessionKeeper HEARTBEAT holds every open session (pre-opened or running) alive
     across slime's trainer->inference weight-sync stall, when generation goes silent and AgentCore
     would otherwise idle-reclaim the microVM mid-episode.

This subclasses the shared decoupled HarnessPool (harness_pool.py, every demo's default: W dispatch
workers pulling from one queue, each episode on the least-loaded SGLang engine, sticky for its turns)
and swaps only the sandbox tier. The agentic loop is the SAME mini_claude loop, run INSIDE the
episode's sandbox session (Anthropic Messages API -> AnthropicAdapter -> policy under training; tools
run in the same session), so the harness scales with the sandbox tier, not the trainer. Needs
--sglang-tool-call-parser qwen25 + --sglang-reasoning-parser qwen3.

generate.generate sends every row of family "terminalbench" here: every row of this demo.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import json
import logging
import os
import secrets
import threading
import time

from .harness_pool import HarnessPool, adapter_for, one_sample

logger = logging.getLogger(__name__)

_TB_POOL: "TerminalBenchPool | None" = None
_TB_LOCK = threading.Lock()


def get_tb_pool(args) -> "TerminalBenchPool":
    """One pool per event loop, like harness_pool.get_pool: its queue, workers and keeper live on the
    loop that built it. The fully-async worker and the eval run on different loops; a process-wide
    singleton made eval queue onto the (orphaned) worker loop, whose workers ran the episodes and then
    resolved eval's futures from the wrong thread, so eval never woke up (hang after the last episode).

    A new loop supersedes the previous pool: its pre-opened sessions and keeper are closed on its own
    loop (in-flight episodes there are left alone)."""
    global _TB_POOL
    loop = asyncio.get_running_loop()
    with _TB_LOCK:
        old = _TB_POOL
        if old is None or old.loop is not loop:
            _TB_POOL = TerminalBenchPool(args)
            if old is not None and not old.loop.is_closed():
                with contextlib.suppress(Exception):
                    asyncio.run_coroutine_threadsafe(old.aclose(), old.loop)
    return _TB_POOL


def oracle_bundle(tasks_dir: str, task_id: str) -> bytes:
    """tar.gz of a task's hidden oracle laid out as Terminal-Bench puts it in the container:
    /tests/run-tests.sh + the CONTENTS of tests/ (so $TEST_DIR/test_outputs.py resolves)."""
    import io
    import tarfile
    from pathlib import Path

    tdir = Path(tasks_dir) / task_id
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for cand in ("run-tests.sh", "run_tests.sh"):
            if (tdir / cand).is_file():
                tf.add(tdir / cand, arcname="run-tests.sh")
                break
        if (tdir / "tests").is_dir():
            for child in sorted((tdir / "tests").iterdir()):
                tf.add(child, arcname=child.name)
    return buf.getvalue()


async def put_tar(sb, data: bytes, dest: str) -> None:
    """Upload a tar.gz into the sandbox (op=put_tar on acs_http_server), chunked under the payload cap."""
    import base64

    b64 = base64.b64encode(data).decode()
    chunk, key = 3 * 2**20, secrets.token_hex(4)
    for i in range(0, max(1, len(b64)), chunk):
        r = await sb._invoke({"op": "put_tar", "dest": dest, "key": key,
                              "b64": b64[i:i + chunk], "more": i + chunk < len(b64)})
        if r.get("error"):
            raise RuntimeError(f"put_tar failed: {r['error']}")


async def grade_episode(sb, task_id: str, tasks_dir: str, test_timeout_s: float = 180.0) -> dict:
    """Inject the hidden tests AFTER the agent's turns, run them, parse pytest's -rA summary.

    Mirrors Terminal-Bench: tests are copied to /tests, `bash /tests/run-tests.sh` runs in the
    task's workdir with TEST_DIR=/tests, and the task is solved iff pytest reports >=1 PASSED and
    no FAILED/ERROR (falls back to the script's exit code when no summary is printed).
    """
    import re

    await put_tar(sb, oracle_bundle(tasks_dir, task_id), "/tests")
    code, out, err = await sb.exec("bash /tests/run-tests.sh", timeout=int(test_timeout_s) + 60,
                                   env={"TEST_DIR": "/tests"})
    status = re.findall(r"(?m)^(PASSED|FAILED|ERROR|XFAIL|XPASS|SKIPPED)\s+\S", f"{out}\n{err}")
    passed = status.count("PASSED") + status.count("XPASS")
    failed = status.count("FAILED") + status.count("ERROR")
    solved = (passed > 0 and failed == 0) if status else code == 0
    return {"solved": solved, "grade_exit": int(code), "tests_passed": passed,
            "tests_total": passed + failed, "grade_timeout": int(code == 124),
            "grade_tail": (out + err)[-600:]}


class TerminalBenchPool(HarnessPool):
    """HarnessPool whose sandbox tier is a fresh, pre-opened session per episode + a heartbeat."""

    # Terminal-Bench tasks act through the shell, not by writing a solution.py that a checker runs.
    # The task instruction IS the user prompt (the persona is the harness system prompt, set in
    # _ensure_started); nothing is written into the task's workdir.
    def _problem_text(self, task) -> str:
        return task.initial_user_prompt()

    def __init__(self, args) -> None:
        super().__init__(args)
        self.sandbox_kind = "agentcore-runtime"
        self.max_turns = int(os.environ.get("EPISODE_MAX_TURNS", "4"))
        self.max_tokens = int(os.environ.get("EPISODE_MAX_TOKENS", "1024"))
        self._harness = None
        # "" = use the image's own WORKDIR (the server reports it); TB_WORKDIR forces one for all tasks.
        self.workdir = os.environ.get("TB_WORKDIR", "")
        # The hidden oracles live on the TRAINER (fetched tasks/ dir), never in the images.
        self.tasks_dir = os.environ.get(
            "TB_TASKS_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                         "terminalbench", "tasks"))
        self.open_attempts = int(os.environ.get("TB_OPEN_ATTEMPTS", "3"))
        # Serverless microVM idle timeout is ~15 min; heartbeat at half that with margin.
        self.idle_timeout = float(os.environ.get("WARM_IDLE_TIMEOUT", "900"))
        # The ONE pre-open knob: how many upcoming batches to keep booted. The session cap follows
        # from slime's own batch shape, so the two can never disagree. 0 = no look-ahead (each episode
        # still starts booting at submit, bounded by the admission gate).
        batches = int(os.environ.get("TB_PREOPEN_BATCHES", "2"))
        self.preopen_max = (batches * args.rollout_batch_size * args.n_samples_per_prompt
                            if batches > 0 else self.num_workers)
        # Strict task -> Runtime ARN map (the image pool), parsed once.
        self._arn_by_image: dict[str, str] = json.loads(
            os.environ.get("AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE", "").strip() or "{}"
        )
        self._keeper = None
        # Upcoming episodes, in the order they will run, as their task's Runtime ARN:
        #   _demand — drawn by slime (or submitted without a look-ahead reservation), not yet claimed;
        #   _future — predicted by the look-ahead data source for the next prompts (replaced per draw).
        self._demand: list[str] = []
        self._future: list[str] = []
        # ARN -> unclaimed sessions opened (or opening) on that task's Runtime, oldest first. Any of
        # them serves any episode of that task: each is brand-new and serves exactly one episode.
        self._ready: dict[str, collections.deque[asyncio.Task]] = collections.defaultdict(collections.deque)

    # ---- lifecycle: also stand up the heartbeat keeper -----------------------
    async def _ensure_started(self) -> None:
        first = not self._started
        await super()._ensure_started()
        if first:
            from .in_session_harness import make_harness
            from .session_keeper import SessionKeeper

            # The loop runs INSIDE the episode's sandbox session (always; no in-trainer harness).
            self._harness = make_harness(max_turns=self.max_turns)

            self._keeper = await SessionKeeper(
                ping_fraction=float(os.environ.get("WARM_PING_FRACTION", "0.5")),
                tick=float(os.environ.get("WARM_KEEPER_TICK", "30.0")),
            ).start()
            # Always-on idle heartbeat: hold any open session that goes idle (e.g. across the
            # weight-sync stall) without gating real ops.
            self._keeper.keep_warm_begin()
            # Terminal-Bench persona (not the solution.py/run_checks.py coding loop) + a longer
            # per-command timeout (apt/pip/builds are routine here).
            from .task_families import TerminalBenchTask

            self._harness.system_prompt = TerminalBenchTask.SYSTEM
            self._harness.exec_timeout_sec = int(os.environ.get("TB_EXEC_TIMEOUT", "180"))
            logger.info("[terminalbench_pool] SessionKeeper started (idle_timeout=%.0fs, preopen_max=%d)",
                        self.idle_timeout, self.preopen_max)

    def arn_for_sample(self, sample) -> str | None:
        """The Runtime ARN a Terminal-Bench row routes to; None for any other row (a mixed batch),
        which the default harness tier opens itself."""
        from .generate import _task_from_sample

        task = _task_from_sample(sample)
        return self._arn_for(task) if getattr(task, "family", "") == "terminalbench" else None

    def _arn_for(self, task) -> str:
        image = getattr(task, "image", task.task_id)
        arn = self._arn_by_image.get(image)
        if not arn:
            raise RuntimeError(
                f"no Runtime ARN for task {task.task_id!r} (image {image!r}) in "
                "AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE — deploy it (steps 2-4) or "
                "drop it from the dataset. Refusing to run it on another task's Runtime."
            )
        return arn

    async def _open_session(self, arn: str):
        """Open one BRAND-NEW session (= its own microVM) on the task's Runtime; heartbeat it.

        Retries a failed open on a NEW session id (a transient cold-start failure must not kill the
        episode); the workdir is the image's own WORKDIR, reported by the server's first ping.
        """
        from .agentcore_runtime_sandbox import AgentCoreRuntimeSandbox

        last = None
        for attempt in range(self.open_attempts):
            # >=33 chars for AgentCore runtimeSessionId (token_hex(16)=32 + prefix clears it).
            sb = AgentCoreRuntimeSandbox(
                runtime_arn=arn, region=self.region, workdir=self.workdir,
                session_id=f"tbep-{secrets.token_hex(16)}",
            )
            try:
                await sb.__aenter__()      # the cold start: allocate + pull + boot + ping
                break
            except Exception as e:  # noqa: BLE001
                last = e
                logger.warning("[terminalbench_pool] open attempt %d on %s failed: %s", attempt + 1, arn, e)
                with contextlib.suppress(Exception):
                    await sb.__aexit__(None, None, None)
        else:
            raise RuntimeError(f"could not open a session on {arn} after {self.open_attempts} attempts: {last}")
        sb._tb_opened_at = time.perf_counter()   # boot done: start of the pre-open idle gap
        sb._tb_open_attempts = attempt + 1
        boot = (getattr(sb, "open_reply", None) or {}).get("boot") or {}
        if not self.workdir and boot.get("workdir"):
            sb._workdir = boot["workdir"]           # the task's own WORKDIR (/app, /app/astropy, ...)

        async def _ping(s=sb) -> None:
            # A raw no-op invoke on the session resets AgentCore's idle clock.
            await s.exec("true", timeout=30)

        self._keeper.register(sb._session_id, _ping, self.idle_timeout)
        return sb

    @staticmethod
    def _cold_start(sb) -> dict:
        """Decompose this session's cold start from client epochs + the server's boot timestamps.

        open_ms       client: first ping sent -> reply (the whole cold start the episode paid)
        pull_boot_ms  sent -> server process started (allocate microVM + pull image + start container)
        ready_ms      server started -> first invoke delivered (platform health check + routing)
        warm_hit      1 if the server was already running BEFORE we asked (AgentCore pre-provisioned it)
        """
        boot = (getattr(sb, "open_reply", None) or {}).get("boot") or {}
        sent, recv = getattr(sb, "open_sent_epoch", None), getattr(sb, "open_recv_epoch", None)
        out = {"open_ms": round((recv - sent) * 1000.0, 1) if sent and recv else -1.0}
        srv, first = boot.get("server_start_epoch"), boot.get("first_invoke_epoch")
        if sent and srv:
            out["pull_boot_ms"] = round(max(0.0, srv - sent) * 1000.0, 1)
            out["warm_hit"] = int(srv < sent)
        if srv and first:
            out["ready_ms"] = round((first - max(srv, sent or srv)) * 1000.0, 1)
        return out

    # ---- pre-open window: keep sessions booted for the next TB_PREOPEN_BATCHES batches -----
    def reserve(self, drawn: list[str], upcoming: list[str]) -> None:
        """Called by the look-ahead data source on every draw (sync, from slime's rollout loop).

        `drawn` = the episodes slime just drew (they will be submitted now); `upcoming` = the episodes
        after them, predicted. The fully-async driver draws one group at a time, so `drawn` is APPENDED
        (earlier draws may still be waiting for a worker) and `upcoming` REPLACES the sliding window.
        """
        self._demand.extend(drawn)
        self._future = list(upcoming)
        try:
            asyncio.get_running_loop().create_task(self._start_and_refill())
        except RuntimeError:          # no loop yet: the first submit refills
            pass

    async def _start_and_refill(self) -> None:
        await self._ensure_started()
        self._refill()

    def _refill(self) -> None:
        """Open sessions for the first `preopen_max` upcoming episodes; stop surplus ones."""
        if self._keeper is None:
            return
        need = collections.Counter((self._demand + self._future)[: self.preopen_max])
        for arn in set(need) | set(self._ready):
            q = self._ready[arn]
            while len(q) > need[arn]:     # no longer upcoming: drop the newest (least booted) one
                asyncio.get_running_loop().create_task(self._discard(q.pop()))
            while len(q) < need[arn]:
                q.append(asyncio.get_running_loop().create_task(self._open_session(arn)))
            if not q:
                del self._ready[arn]

    def _claim(self, arn: str):
        """A worker picked up an episode of `arn`: take a ready session for it (or None)."""
        with contextlib.suppress(ValueError):
            self._demand.remove(arn)
        q = self._ready.get(arn)
        pending = q.popleft() if q else None
        self._refill()                # a slot moved from "ahead" to "running": boot the next one
        return pending

    async def _discard(self, pending: asyncio.Task) -> None:
        with contextlib.suppress(Exception):
            await self._close_session(await pending)

    async def _close_session(self, sb) -> None:
        self._keeper.unregister(sb._session_id)
        await sb.__aexit__(None, None, None)   # StopRuntimeSession: the microVM is torn down

    async def _grade(self, sb, task) -> dict:
        return await grade_episode(sb, task.task_id, self.tasks_dir,
                                   float(task.spec.get("max_test_timeout_sec") or 180))

    # ---- the episode: its own fresh session on THIS task's Runtime -----------
    async def _run_episode(self, worker, task, sampling_params, base_sample):
        from . import router_trace
        from .agentcore_runtime_sandbox import _arn_tail

        wid = worker.wid

        # Adapter Bearer session id == trajectory key.
        sid = getattr(base_sample, "session_id", None)
        if not sid or len(sid) < 33:
            sid = f"tb-{task.task_id}-{secrets.token_hex(16)}"
        base_sample.session_id = sid

        arn = self._arn_for(task)
        pending = self._claim(arn)
        t0 = time.perf_counter()
        sb = await (pending if pending is not None else self._open_session(arn))
        wait_ms = (time.perf_counter() - t0) * 1000.0
        # How long the booted microVM sat idle before this worker claimed it (0 if it was still
        # booting at pickup). This is the gap the SessionKeeper must bridge (< idle timeout).
        idle_ms = max(0.0, (t0 - sb._tb_opened_at) * 1000.0)
        cold = self._cold_start(sb)
        spec = task.spec or {}

        # Inference AFTER the sandbox: the least-loaded engine, sticky for all of this episode's turns.
        e, engine = self.acquire_engine()
        try:
            adapter, _local_url, session_url = adapter_for(self.args, engine)
            adapter.open_session(sid, sampling_defaults=sampling_params)
        except BaseException:
            self.release_engine(e)
            await self._close_session(sb)
            raise
        ep = {"task": task.task_id, "difficulty": spec.get("difficulty", "?"),
              "category": spec.get("category", "?"), "solved": 0, "infra_error": "none"}
        try:
            router_trace.emit(
                task_id=task.task_id,
                rollout_id=getattr(base_sample, "index", "?"),
                harness_worker=wid,
                engine=engine,
                engine_pinned=int(len(self._engine_urls) > 1),
                adapter_session=sid,
                sandbox_id=getattr(sb, "sandbox_id", "?"),
                sandbox_session=sb._session_id,
                runtime_arn=_arn_tail(arn),
                sandbox_image=task.image,
                family=task.family,
            )
            # DEMO-SPECIFIC metric ([demo] channel): every episode is on its OWN fresh session;
            # `preopened` says whether its cold start was started at submit, and `open_wait_ms` is
            # how much of that cold start the worker still had to wait for (the critical-path cost);
            # `preopen_idle_ms` is the reverse: how long the booted microVM waited for the worker.
            # open_ms / pull_boot_ms / ready_ms / warm_hit decompose the cold start itself.
            print(f"[demo] worker={wid} qdepth={worker.qdepth} engine={e} "
                  f"engine_inflight={self.inflight(e)} fresh_session=1 preopened={int(pending is not None)} "
                  f"open_wait_ms={wait_ms:.1f} preopen_idle_ms={idle_ms:.1f} "
                  f"open_attempts={sb._tb_open_attempts} "
                  + " ".join(f"{k}={v}" for k, v in cold.items())
                  + f" arn={_arn_tail(arn)}", flush=True)

            # The task's environment is baked into its image; the instruction goes straight into
            # the prompt (nothing is written into the task's workdir — some tests inspect it).
            ta = time.perf_counter()
            res = await self._harness.run(
                sb,
                session_id=sid,
                adapter_url=session_url,   # the URL the in-session loop dials back
                prompt=self._problem_text(task),
                max_tokens=self.max_tokens,
                http=worker.http,
            )
            ep.update(agent_ms=round((time.perf_counter() - ta) * 1000.0, 1), turns=res.turns,
                      tool_calls=res.tool_calls, stop=res.stop_reason,
                      harness_where=res.where)

            # RLVR reward: the task's OWN hidden pytest oracle, injected + run after the agent.
            tg = time.perf_counter()
            try:
                g = await self._grade(sb, task)
            except Exception as exc:  # noqa: BLE001 — a grading infra failure scores 0, flagged
                # NOT `as e`: `e` is the engine index released in `finally`, and Python deletes an
                # except-variable when its block ends (UnboundLocalError killed an eval, 2026-10-04).
                logger.warning("[terminalbench_pool] grade failed for %s: %s", task.task_id, exc)
                g = {"solved": False, "grade_exit": -1, "tests_passed": 0, "tests_total": 0,
                     "grade_timeout": 0}
                ep["infra_error"] = "grade"
            g.pop("grade_tail", None)
            ep.update(g, grade_ms=round((time.perf_counter() - tg) * 1000.0, 1))
            ep["solved"] = int(g["solved"])
            reward = 1.0 if g["solved"] else 0.0

            samples = await adapter.finish_session(
                sid, base_sample=base_sample, reward=float(reward),
                extra_metadata={"task_id": task.task_id, "solved": bool(g["solved"]),
                                "harness_worker": wid, "sandbox_session": sb._session_id},
            )
            out, ep["fan_out"] = one_sample(samples, float(reward))
            return out
        except Exception as exc:
            ep["infra_error"] = type(exc).__name__
            raise
        finally:
            ep["episode_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
            ep["session_s"] = round(time.perf_counter() - sb._tb_opened_at, 1)  # microVM-seconds paid
            print("[demo] episode_end " + " ".join(f"{k}={v}" for k, v in ep.items()), flush=True)
            await adapter.drop_session(sid, wait_timeout=30)
            self.release_engine(e)
            await self._close_session(sb)

    # ---- public entry point: start the episode's cold start at enqueue time --
    async def submit(self, task, sampling_params, base_sample):
        await self._ensure_started()
        arn = self._arn_for(task)          # fail fast on an unmapped task, before queueing
        if not (getattr(base_sample, "metadata", None) or {}).get("tb_reserved"):
            # Not announced by the look-ahead source (eval, or no source wired): start its boot now.
            self._demand.append(arn)
        self._refill()
        return await self.run(lambda worker: self._run_episode(worker, task, sampling_params, base_sample))

    # ---- shutdown: stop any still-pre-opened sessions + the keeper -----------
    async def aclose(self) -> None:
        self._demand, self._future = [], []
        for q in list(self._ready.values()):
            for pending in q:
                await self._discard(pending)
        self._ready.clear()
        if self._keeper is not None:
            await self._keeper.aclose()
