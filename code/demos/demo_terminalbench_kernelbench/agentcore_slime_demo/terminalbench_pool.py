"""Isolated harness pool for Terminal-Bench rows — ONE FRESH SANDBOX PER EPISODE, pre-opened, on TWO substrates.

demo_terminalbench_kernelbench routes every task to one of three substrates (the route string says which):
  * "arn:aws:bedrock-agentcore:..."     -> a fresh AgentCore microVM session (AgentCoreRuntimeSandbox)
  * "docker://<m5 ip>:<port>/<task>"    -> the m5 tier, for compose / x86-only / > 2 GB / root-needing tasks:
                                           TB_M5_MODE=per_episode (default) a fresh EC2 instance per episode
                                           (Ec2EpisodeSandbox); =shared a fresh compose project on the one
                                           shared m5 (DockerComposeSandbox)
  * "gpu:arn:aws:bedrock-agentcore:..." -> a fresh AgentCore session on a g6 capacity provider = its own
                                           GPU instance (AgentCoreRuntimeSandbox; empty unless TB_GPU_TASKS)
  * "gpuec2:<task>"                     -> OPT-IN (TB_KB_SUBSTRATE=ec2): one GPU slot of a warm plain-EC2 g6
                                           instance, a fresh container per episode (Ec2GpuSandbox)
All speak the same ops to the same in-image server, so everything below is substrate-agnostic except
three things: the heartbeat (docker projects have no idle reclaim, so they are not registered), an
open gate per bounded substrate (TB_DOCKER_MAX_PROJECTS m5 projects, TB_GPU_MAX_SESSIONS g6 instances,
held from open to close), and the `substrate=` field on the [demo] lines. The rest of this docstring
describes the microVM case.

Terminal-Bench is a LARGE image pool (one container image / one AgentCore Runtime per task; each
row names its task's image). Every episode must be ISOLATED from every other: Terminal-Bench tasks act on
the whole OS (apt install, /etc, background services, /tmp), so a shared microVM with a reset
workdir would leak state between episodes. The isolation boundary AgentCore actually guarantees is
the SESSION: one runtimeSessionId = one dedicated microVM, terminated + memory-sanitized when the
session is stopped. So this pool gives every episode its OWN brand-new session and never reuses one:

  1. Strict per-task ROUTING: the episode's task.image (the task name) is looked up in
     TB_ROUTE_BY_IMAGE. A missing task is an error — never a fallback to some
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
import re
import os
import secrets
import sys
import threading
import time
import zlib

from . import skills
from .docker_compose_sandbox import DockerComposeSandbox, gc_hosts, is_docker_route
from .harness_pool import HarnessPool, adapter_for, one_sample

logger = logging.getLogger(__name__)

_TB_POOL: "TerminalBenchPool | None" = None
_TB_LOCK = threading.Lock()
_DOCKER_GC_DONE = False        # m5 stale-project gc: once per process (a later pool must not kill live ones)
_G6_GC_DONE = False            # same for the plain-EC2 g6 instances


GPU_PREFIX = "gpu:"


def _cancel_reason(exc: BaseException) -> str | None:
    """Why an episode died WITHOUT a sandbox fault, or None if `exc` is a real infra error.
    "abort": the caller cancelled it (rollout abort / over-sampling). "shutdown": the process tore
    down under it — the loop's executor or the interpreter is already shut down (the look-ahead
    pre-opens episodes for batches that never run; train48 run 2 logged 25 of these at exit, all
    `cannot schedule new futures after shutdown` — last_rollout.stop_lookahead now prevents them). Such episodes are tagged cancelled=<reason> and
    keep infra_error=none, so they never count as infra errors."""
    if sys.is_finalizing():
        return "shutdown"
    with contextlib.suppress(RuntimeError):
        if getattr(asyncio.get_running_loop(), "_executor_shutdown_called", False):
            return "shutdown"
    seen, cur = set(), exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if "after shutdown" in str(cur) or "interpreter shutdown" in str(cur):
            return "shutdown"
        cur = cur.__cause__ or cur.__context__
    return "abort" if isinstance(exc, asyncio.CancelledError) else None


G6EC2_PREFIX = "gpuec2:"


def substrate_of(route: str) -> str:
    if is_docker_route(route):
        return "docker"
    if str(route).startswith(G6EC2_PREFIX):
        return "gpu_ec2"
    return "gpu" if str(route).startswith(GPU_PREFIX) else "microvm"


def m5_mode() -> str:
    mode = os.environ.get("TB_M5_MODE", "warm").strip().lower()
    if mode not in ("warm", "per_episode", "shared"):
        raise RuntimeError(f"TB_M5_MODE={mode!r}: warm | per_episode | shared")
    return mode


def new_sandbox(route: str, *, region: str, workdir: str, session_id: str | None = None):
    """An UNOPENED sandbox for `route` (a microVM Runtime ARN, a docker:// m5 route, gpu:<g6 Runtime ARN>
    or gpuec2:<task> on the plain-EC2 g6 pool)."""
    from .agentcore_runtime_sandbox import AgentCoreRuntimeSandbox

    if str(route).startswith(G6EC2_PREFIX):
        from .ec2_gpu_sandbox import Ec2GpuSandbox

        return Ec2GpuSandbox(route, region=region, workdir=workdir, session_id=session_id,
                             read_timeout=int(os.environ.get("TB_GPU_READ_TIMEOUT", "900")))

    if is_docker_route(route):
        if m5_mode() == "warm":
            from .ec2_episode_sandbox import Ec2WarmSandbox

            return Ec2WarmSandbox(route, region=region, workdir=workdir, session_id=session_id)
        if m5_mode() == "per_episode":
            from .ec2_episode_sandbox import Ec2EpisodeSandbox

            return Ec2EpisodeSandbox(route, region=region, workdir=workdir, session_id=session_id)
        return DockerComposeSandbox(route, workdir=workdir, session_id=session_id)
    if str(route).startswith(GPU_PREFIX):     # one session = one whole g6: the first ping cold-boots it
        return AgentCoreRuntimeSandbox(runtime_arn=route[len(GPU_PREFIX):], region=region, workdir=workdir,
                                       session_id=session_id,
                                       read_timeout=int(os.environ.get("TB_GPU_READ_TIMEOUT", "900")))
    return AgentCoreRuntimeSandbox(runtime_arn=route, region=region, workdir=workdir, session_id=session_id)


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
            if old is not None and m5_mode() == "warm":
                # The old loop may be wedged and never close its sessions: free the m5 cap they hold
                # now, or this loop's m5 opens wait for an instance forever.
                from .ec2_episode_sandbox import reclaim_warm
                with contextlib.suppress(Exception):
                    reclaim_warm(old.loop, old.region)
            if old is not None and any(substrate_of(r) == "gpu_ec2" for r in old._arn_by_image.values()):
                from .ec2_gpu_sandbox import reclaim_g6   # same, for the plain-EC2 g6 slots it holds
                with contextlib.suppress(Exception):
                    reclaim_g6(old.loop, old.region)
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


_SUMMARY_HDR = re.compile(r"^=+ short test summary info =+$")
_SUMMARY_LINE = re.compile(r"^(PASSED|FAILED|ERROR|XFAIL|XPASS|SKIPPED) (\S+)")


def pytest_summary(text: str) -> list[str]:
    """Statuses from pytest's `short test summary info` block(s) only, as upstream's PytestParser.

    Scanning ALL output (the old regex) counted lines the TESTS print — `ERROR 3: ...`, `FAILED (0/5)` —
    and log records (`ERROR traitlets:...`) as test results: jupyter-notebook-server, movie-helper and
    winning-avg-corewars failed with every real test passing. A summary line names a test file
    (`tests/test_outputs.py::test_x`, or the bare file for a collection ERROR); the block ends at the
    `=== N passed ... ===` line. No block (pytest without -rA, all passed) -> [] -> the exit code decides."""
    status, inside = [], False
    for ln in text.splitlines():
        ln = ln.rstrip()
        if _SUMMARY_HDR.match(ln):
            inside = True
        elif inside and ln.startswith("="):
            inside = False
        elif inside:
            m = _SUMMARY_LINE.match(ln)
            if m and ".py" in m.group(2):
                status.append(m.group(1))
    return status


async def grade_episode(sb, task_id: str, tasks_dir: str, test_timeout_s: float = 180.0) -> dict:
    """Inject the hidden tests AFTER the agent's turns, run them, parse pytest's -rA summary.

    Mirrors Terminal-Bench: tests are copied to /tests, `bash /tests/run-tests.sh` runs in the
    task's workdir with TEST_DIR=/tests, and the task is solved iff pytest reports >=1 PASSED and
    no FAILED/ERROR (falls back to the script's exit code when no summary is printed).
    """
    import re

    await put_tar(sb, oracle_bundle(tasks_dir, task_id), "/tests")
    code, out, err = await sb.exec("bash /tests/run-tests.sh", timeout=int(test_timeout_s) + 60,
                                   env={"TEST_DIR": "/tests", "ACS_GRADER": "1"})   # as root (m5 ACS_EXEC_USER)
    status = pytest_summary(f"{out}\n{err}")
    passed = status.count("PASSED") + status.count("XPASS")
    failed = status.count("FAILED") + status.count("ERROR")
    solved = (passed > 0 and failed == 0) if status else code == 0
    return {"solved": solved, "grade_exit": int(code), "tests_passed": passed,
            "tests_total": passed + failed, "grade_timeout": int(code == 124),
            # stdout (pytest's summary) first: tests install their deps with uv/pip, whose progress goes to
            # stderr, so a plain (out + err) tail was all install noise and hid why the tests failed.
            "grade_tail": out[-1500:] + (f"\n[stderr] {err[-300:]}" if err.strip() else "")}


def _parse_harness_set(spec: str) -> list[tuple[str, float]]:
    """Parse HARNESS_SET into [(kind, cumulative_percent)] for crc32 assignment.

    "mini_claude"                       -> [("mini_claude", 100.0)]  (default; current behavior)
    "mini_claude:0.5,claude_cli:0.5"    -> [("mini_claude", 50.0), ("claude_cli", 100.0)]
    "mini_claude,claude_cli"            -> equal split among kinds with no explicit fraction.
    """
    items: list[tuple[str, float | None]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            k, f = part.split(":", 1)
            items.append((k.strip(), float(f)))
        else:
            items.append((part, None))
    if not items:
        items = [("mini_claude", None)]
    rem = max(0.0, 1.0 - sum(f for _, f in items if f is not None))
    missing = [i for i, (_, f) in enumerate(items) if f is None]
    each = rem / len(missing) if missing else 0.0
    fracs = [each if f is None else f for _, f in items]
    tot = sum(fracs) or 1.0
    out, cum = [], 0.0
    for (k, _), fr in zip(items, fracs):
        cum += 100.0 * fr / tot
        out.append((k, cum))
    out[-1] = (out[-1][0], 100.0)      # guard float drift: last bucket catches everything
    return out


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
        # Multi-harness routing (default "mini_claude" => the current in-session path only, no change).
        # A mix like "mini_claude:0.5,claude_cli:0.5" routes a fraction of episodes to the DECOUPLED
        # real-Claude-Code-CLI harness Runtime (its own microVM) — see decoupled_cli_harness.py.
        self._harness_set = _parse_harness_set(os.environ.get("HARNESS_SET", "mini_claude"))
        self._harness_assign = os.environ.get("HARNESS_ASSIGN", "prompt").strip().lower()  # prompt|episode
        self._harness_cli_arn = os.environ.get("AGENTCORE_HARNESS_CLI_RUNTIME_ARN", "").strip()
        self._harnesses: dict[str, object] = {}
        self._harness_cli_pool = None
        self._decoupled = None
        self._init_done = False     # guards the harness/keeper/pool init (see _ensure_started)
        # "" = use the image's own WORKDIR (the server reports it); TB_WORKDIR forces one for all tasks.
        self.workdir = os.environ.get("TB_WORKDIR", "")
        # The hidden oracles live on the TRAINER (fetched tasks/ dir), never in the images.
        self.tasks_dir = os.environ.get(
            "TB_TASKS_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                         "terminalbench", "tasks"))
        self.open_attempts = int(os.environ.get("TB_OPEN_ATTEMPTS", "3"))
        self.partial_credit = float(os.environ.get("TB_PARTIAL_CREDIT", "0.5"))   # see _reward
        # Serverless microVM idle timeout is ~15 min; heartbeat at half that with margin.
        self.idle_timeout = float(os.environ.get("WARM_IDLE_TIMEOUT", "900"))
        # The ONE pre-open knob: how many upcoming batches to keep booted. The session cap follows
        # from slime's own batch shape, so the two can never disagree. 0 = no look-ahead (each episode
        # still starts booting at submit, bounded by the admission gate).
        batches = int(os.environ.get("TB_PREOPEN_BATCHES", "2"))
        self.preopen_max = (batches * args.rollout_batch_size * args.n_samples_per_prompt
                            if batches > 0 else self.num_workers)
        # Slow-to-boot substrates (a per-episode m5 VM ~35-60 s, a g6 Instance) look further ahead, so
        # their cold start is hidden too; the substrate gates still cap how many are open at once.
        slow = int(os.environ.get("TB_PREOPEN_BATCHES_SLOW", "5")) if batches > 0 else 0
        self.preopen_max_slow = max(self.preopen_max,
                                    slow * args.rollout_batch_size * args.n_samples_per_prompt)
        # Strict task -> route map (AgentCore Runtime ARN or docker:// m5 route), parsed once.
        self._arn_by_image: dict[str, str] = json.loads(os.environ.get("TB_ROUTE_BY_IMAGE", "").strip() or "{}")
        # Bounded substrates: at most this many sessions at once (pre-opened + running); an open past it
        # waits here. The m5's own ACS_MAX_PROJECTS is a backstop; a g6 session is a whole instance.
        m5_cap = (os.environ.get("TB_M5_MAX_INSTANCES", "24") if m5_mode() in ("warm", "per_episode")
                  else os.environ.get("TB_DOCKER_MAX_PROJECTS", "16"))
        self._gates = {"docker": asyncio.Semaphore(int(m5_cap)),
                       "gpu": asyncio.Semaphore(int(os.environ.get("TB_GPU_MAX_SESSIONS", "32"))),
                       # plain-EC2 g6: the slot pool itself waits for a free GPU (TB_G6_MAX_INSTANCES)
                       "gpu_ec2": asyncio.Semaphore(int(os.environ.get("TB_GPU_MAX_SESSIONS", "32")))}
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
        await super()._ensure_started()
        if self._init_done:
            return
        # Serialize the harness/keeper/pool init under the base's start lock (free once super()
        # returns): submit() awaits this fully before enqueuing, so a worker never picks up an
        # episode while self._decoupled / self._keeper is still half-built (the 'NoneType has no
        # attribute run' race, seen when the slower claude_cli pool widened the window).
        async with self._start_lock:
            if self._init_done:
                return
            global _DOCKER_GC_DONE
            if not _DOCKER_GC_DONE and any(is_docker_route(r) for r in self._arn_by_image.values()):
                _DOCKER_GC_DONE = True
                if m5_mode() in ("warm", "per_episode"):   # a crashed run's per-episode / warm instances
                    from .ec2_episode_sandbox import episode_config, terminate_tagged

                    n = await asyncio.to_thread(terminate_tagged, self.region, episode_config().get("tag", "acs-tbh-m5-ep"))
                    logger.info("[terminalbench_pool] terminated %d stale per-episode m5 instance(s)", n)
                else:
                    await asyncio.to_thread(gc_hosts, list(self._arn_by_image.values()))
            global _G6_GC_DONE
            if not _G6_GC_DONE and any(substrate_of(r) == "gpu_ec2" for r in self._arn_by_image.values()):
                _G6_GC_DONE = True
                from .ec2_episode_sandbox import terminate_tagged
                from .ec2_gpu_sandbox import g6_config

                n = await asyncio.to_thread(terminate_tagged, self.region, g6_config().get("tag", "acs-tbh-g6ec2"))
                logger.info("[terminalbench_pool] terminated %d stale plain-EC2 g6 instance(s)", n)
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
            self._harnesses["mini_claude"] = self._harness

            # Decoupled harnesses (their own harness Runtime microVM, driving the sandbox over the
            # bridge): claude_cli = the real Claude Code CLI; mini_claude_rt = the mini mimic hosted
            # in the harness Runtime instead of in-session. Both share ONE warm Runtime pool. Assign
            # the harness objects BEFORE awaiting pool.start(): a racing episode then borrows (and
            # waits on the pool's queue) instead of hitting a None harness.
            _DECOUPLED_KINDS = {"claude_cli": "claude", "mini_claude_rt": "mini"}
            decoupled = [k for k, _ in self._harness_set if k in _DECOUPLED_KINDS]
            if decoupled:
                if not self._harness_cli_arn:
                    raise RuntimeError(f"HARNESS_SET selects {decoupled} but AGENTCORE_HARNESS_CLI_RUNTIME_ARN "
                                       "is unset — deploy the harness Runtime (4_deploy_runtime.sh step 4h).")
                from .decoupled_cli_harness import DecoupledCliHarness
                from .harness_cli_pool import HarnessCliRuntimePool

                # Fraction of episodes going to the decoupled pool = sum of its kinds' per-kind
                # fractions (the set stores CUMULATIVE %, so diff consecutive buckets).
                prev = 0.0
                frac = 0.0
                for k, cum in self._harness_set:
                    if k in _DECOUPLED_KINDS:
                        frac += (cum - prev) / 100.0
                    prev = cum
                n_sess = int(os.environ.get("HARNESS_CLI_SESSIONS", "") or
                             max(1, round(self.num_workers * max(frac, 0.1))))
                read_timeout = max(int(os.environ.get("INVOKE_READ_TIMEOUT", "900")),
                                   int(os.environ.get("AGENT_TIME_BUDGET_SEC", "900")) + 60)
                self._harness_cli_pool = HarnessCliRuntimePool(
                    arn=self._harness_cli_arn, region=self.region, n_sessions=n_sess,
                    keeper=self._keeper, idle_timeout=self.idle_timeout, read_timeout=read_timeout)
                for k in decoupled:
                    h = DecoupledCliHarness(self._harness_cli_pool, kind=_DECOUPLED_KINDS[k],
                                            exec_timeout_sec=self._harness.exec_timeout_sec)
                    h.system_prompt = TerminalBenchTask.SYSTEM     # TB persona for the mini variant
                    self._harnesses[k] = h
                self._decoupled = self._harnesses.get("claude_cli")
                await self._harness_cli_pool.start()

            self._init_done = True
            logger.info("[terminalbench_pool] SessionKeeper started (idle_timeout=%.0fs, preopen_max=%d, "
                        "harness_set=%s)", self.idle_timeout, self.preopen_max,
                        ",".join(f"{k}:{c:.0f}" for k, c in self._harness_set))

    def routes(self, task) -> bool:
        """True if this pool runs `task`: every Terminal-Bench row, and any other row whose image has a
        route (KernelBench rows -> the g6 tier)."""
        return getattr(task, "family", "") == "terminalbench" or getattr(task, "image", "") in self._arn_by_image

    def arn_for_sample(self, sample) -> str | None:
        """The route a row this pool runs goes to; None for any other row (the default harness tier
        opens those itself)."""
        from .generate import _task_from_sample

        task = _task_from_sample(sample)
        return self._arn_for(task) if self.routes(task) else None

    def _arn_for(self, task) -> str:
        image = getattr(task, "image", task.task_id)
        arn = self._arn_by_image.get(image)
        if not arn:
            raise RuntimeError(
                f"no route for task {task.task_id!r} (image {image!r}) in "
                "TB_ROUTE_BY_IMAGE — deploy it (steps 2-4) or "
                "drop it from the dataset. Refusing to run it on another task's Runtime."
            )
        return arn

    def _pick_harness(self, task) -> str:
        """Which harness runs this episode. Default set => always "mini_claude" (no change).

        Assignment is PER-PROMPT by default (crc32 of task_id): all n_samples_per_prompt rollouts of a
        prompt share a harness, so GRPO's within-group advantage isn't conflated with harness
        difficulty. HARNESS_ASSIGN=episode routes each episode independently (maximal mixing)."""
        if len(self._harness_set) == 1:
            return self._harness_set[0][0]
        key = secrets.token_hex(8) if self._harness_assign == "episode" else str(task.task_id)
        r = (zlib.crc32(key.encode()) % 10000) / 100.0      # 0..100, stable across train+eval
        for kind, cum in self._harness_set:
            if r < cum:
                return kind
        return self._harness_set[-1][0]

    async def _open_session(self, arn: str, prio: dict | None = None):
        """Open one BRAND-NEW session (its own microVM, or its own compose project on the m5) on the
        task's route; heartbeat it (microVM only).

        Retries a failed open on a NEW session id (a transient cold-start failure must not kill the
        episode); the workdir is the image's own WORKDIR, reported by the server's first ping.
        """
        sub = substrate_of(arn)
        docker = sub in ("docker", "gpu_ec2")     # our own EC2: no AgentCore idle reclaim to heartbeat
        gate = self._gates.get(sub)
        if gate is not None:
            await gate.acquire()                   # released by _close_session (or below on failure)
        last = None
        for attempt in range(self.open_attempts):
            # >=33 chars for AgentCore runtimeSessionId (token_hex(16)=32 + prefix clears it).
            sb = new_sandbox(arn, region=self.region, workdir=self.workdir,
                             session_id=f"tbep-{secrets.token_hex(16)}")
            sb._tb_prio = prio if prio is not None else {"hi": True}   # a worker's own open: high
            try:
                await sb.__aenter__()      # the cold start: allocate + pull + boot + ping (or compose up)
                break
            except BaseException as e:  # noqa: BLE001
                last = e
                logger.warning("[terminalbench_pool] open attempt %d on %s failed: %s", attempt + 1, arn, e)
                with contextlib.suppress(Exception):
                    await sb.__aexit__(None, None, None)
                if not isinstance(e, Exception):     # cancelled (a discarded pre-open): stop retrying
                    if gate is not None:
                        gate.release()
                    raise
        else:
            if gate is not None:
                gate.release()
            raise RuntimeError(f"could not open a session on {arn} after {self.open_attempts} attempts: {last}")
        sb._tb_substrate = sub
        sb._tb_opened_at = time.perf_counter()   # boot done: start of the pre-open idle gap
        sb._tb_open_attempts = attempt + 1
        boot = (getattr(sb, "open_reply", None) or {}).get("boot") or {}
        if not self.workdir and boot.get("workdir"):
            sb._workdir = boot["workdir"]           # the task's own WORKDIR (/app, /app/astropy, ...)

        if docker:
            return sb                               # no idle reclaim on the m5 / own g6: nothing to heartbeat

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
        # warm m5 pool: was this instance reused, how many episodes it has served, pool size at open
        out.update({k: boot[k] for k in ("warm_reused", "instance_episodes", "warm_instances", "instance_type",
                                         "gpu_slot", "g6_instances", "g6_slots_busy") if k in boot})
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
        window = (self._demand + self._future)[: self.preopen_max_slow]
        need = collections.Counter(a for i, a in enumerate(window)
                                   if i < (self.preopen_max_slow if substrate_of(a) in ("docker", "gpu", "gpu_ec2")
                                           else self.preopen_max))
        for arn in set(need) | set(self._ready):
            q = self._ready[arn]
            while len(q) > need[arn]:     # no longer upcoming: drop the newest (least booted) one
                asyncio.get_running_loop().create_task(self._discard(q.pop()))
            while len(q) < need[arn]:
                prio = {"hi": False}      # a look-ahead open: yields capped slots to claimed ones
                t = asyncio.get_running_loop().create_task(self._open_session(arn, prio))
                t._tb_prio = prio
                q.append(t)
            if not q:
                del self._ready[arn]

    def _claim(self, arn: str):
        """A worker picked up an episode of `arn`: take a ready session for it (or None)."""
        with contextlib.suppress(ValueError):
            self._demand.remove(arn)
        q = self._ready.get(arn)
        pending = None
        if q:
            # An already-OPEN session first: waiting on the oldest still-pending open while a newer one
            # for the same route sits ready deadlocked the coverage124 run (16 EC2 GPU slots all held by
            # unclaimed pre-opens, 16 workers each waiting on a head that needed a 17th slot).
            pending = next((t for t in q if t.done() and not t.cancelled() and t.exception() is None), None)
            if pending is not None:
                q.remove(pending)
            else:
                pending = q.popleft()
                getattr(pending, "_tb_prio", {})["hi"] = True   # a worker now waits on it
        self._refill()                # a slot moved from "ahead" to "running": boot the next one
        return pending

    async def _discard(self, pending: asyncio.Task) -> None:
        try:
            sb = await pending
        except BaseException:  # noqa: BLE001
            if pending.cancelled() or pending.done():
                return                 # the open was cancelled or failed: nothing to close (and aclose
            raise                      # must go on to the next one, not stop at a CancelledError)
        with contextlib.suppress(Exception):
            await self._close_session(sb)

    async def _close_session(self, sb) -> None:
        self._keeper.unregister(sb._session_id)
        try:
            await sb.__aexit__(None, None, None)   # StopRuntimeSession / compose down: torn down
        finally:
            gate = self._gates.get(getattr(sb, "_tb_substrate", ""))
            if gate is not None:
                gate.release()

    def _reward(self, g: dict, evaluation: bool) -> float:
        """1 if every hidden test passes. Otherwise, in TRAINING only, partial credit
        TB_PARTIAL_CREDIT x (tests passed / tests total), default 0.5, so a solve still beats any
        partial run. All-or-nothing gave GRPO almost no signal: in train48 run 2, 32 of 48 steps had
        every group at one reward (grad_norm 0), while 873 of 1,940 unsolved episodes passed some
        tests. Eval stays 0/1, so eval/terminalbench remains the solve rate. TB_PARTIAL_CREDIT=0 =
        the old reward."""
        if g["solved"]:
            return 1.0
        total = int(g.get("tests_total") or 0)
        if evaluation or self.partial_credit <= 0 or total <= 0:
            return 0.0
        return self.partial_credit * min(1.0, int(g.get("tests_passed") or 0) / total)

    async def _grade(self, sb, task) -> dict:
        return await grade_episode(sb, task.task_id, self.tasks_dir,
                                   float(task.spec.get("max_test_timeout_sec") or 180))

    # ---- the episode: its own fresh session on THIS task's Runtime -----------
    async def _run_episode(self, worker, task, sampling_params, base_sample, evaluation: bool = False):
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
        try:
            sb = await (pending if pending is not None else self._open_session(arn))
        except Exception as exc:  # noqa: BLE001 — NOT BaseException: a cancellation must still propagate.
            # A sandbox session-open failure (classically a transient "Insufficient EC2 capacity"
            # for the g6/KernelBench tier) must NOT take down the whole job during EVAL — eval is a
            # metric, not a gradient. Skip this eval episode: score it 0, flag it, and let training
            # continue. (A training rollout still raises: the fully-async driver tolerates a dropped
            # training sample, but an eval that silently vanished would break slime's as_completed
            # gather in eval_rollout_single_dataset.) _open_session releases its substrate gate on
            # the way out, so nothing leaks here.
            if not evaluation:
                raise
            logger.warning("[terminalbench_pool] eval session-open failed for %s on %s; skipping this "
                           "eval episode (reward 0): %s", task.task_id, _arn_tail(arn), exc)
            print(f"[demo] episode_end task={task.task_id} family={getattr(task, 'family', '?')} "
                  f"solved=0 infra_error={type(exc).__name__} eval_skipped=open_failed reward=0.0",
                  flush=True)
            base_sample.reward = 0.0
            base_sample.status = base_sample.Status.COMPLETED   # not TRUNCATED: a clean skip, not a cutoff
            md = dict(getattr(base_sample, "metadata", None) or {})
            md["eval_skipped"] = "open_failed"
            base_sample.metadata = md
            return base_sample
        wait_ms = (time.perf_counter() - t0) * 1000.0
        # How long the booted microVM sat idle before this worker claimed it (0 if it was still
        # booting at pickup). This is the gap the SessionKeeper must bridge (< idle timeout).
        idle_ms = max(0.0, (t0 - sb._tb_opened_at) * 1000.0)
        cold = self._cold_start(sb)
        if getattr(task, "family", "") != "terminalbench":    # KernelBench (g6 tier): the coding episode
            return await self._run_coding_episode(worker, task, sampling_params, base_sample, sb, arn,
                                                  pending is not None, wait_ms, idle_ms, cold, evaluation)
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
        harness_kind = self._pick_harness(task)
        ep = {"task": task.task_id, "substrate": sb._tb_substrate, "difficulty": spec.get("difficulty", "?"),
              "category": spec.get("category", "?"), "harness": harness_kind, "solved": 0,
              "infra_error": "none"}
        if evaluation:
            ep["eval"] = 1          # lets metrics.py split solve-rate per harness at eval time
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
                harness=harness_kind,
            )
            # DEMO-SPECIFIC metric ([demo] channel): every episode is on its OWN fresh session;
            # `preopened` says whether its cold start was started at submit, and `open_wait_ms` is
            # how much of that cold start the worker still had to wait for (the critical-path cost);
            # `preopen_idle_ms` is the reverse: how long the booted microVM waited for the worker.
            # open_ms / pull_boot_ms / ready_ms / warm_hit decompose the cold start itself.
            print(f"[demo] worker={wid} qdepth={worker.qdepth} engine={e} "
                  f"engine_inflight={self.inflight(e)} fresh_session=1 substrate={sb._tb_substrate} "
                  f"preopened={int(pending is not None)} "
                  f"open_wait_ms={wait_ms:.1f} preopen_idle_ms={idle_ms:.1f} "
                  f"open_attempts={sb._tb_open_attempts} "
                  + " ".join(f"{k}={v}" for k, v in cold.items())
                  + f" arn={_arn_tail(arn)}", flush=True)

            # The task's environment is baked into its image; the instruction goes straight into
            # the prompt (nothing is written into the task's workdir — some tests inspect it). The
            # mini_claude loop runs INSIDE sb; the decoupled claude_cli harness runs in its own
            # Runtime and drives sb over the network. Both capture the trajectory proxy-side and both
            # return a MiniAgentResult, so grading + sample finish below are shared.
            ta = time.perf_counter()
            harness = self._harnesses.get(harness_kind, self._harness)
            skills_inst = None
            if getattr(harness, "in_session", True):
                # SKILLS=listed: install the family's skills outside the workdir and list them in this
                # episode's system prompt (in-session mini_claude only; removed again before _grade).
                skills_inst = await skills.install(sb, "terminalbench", evaluation=evaluation)
                res = await harness.run(
                    sb,
                    session_id=sid,
                    adapter_url=session_url,       # the URL the in-session loop dials back
                    prompt=self._problem_text(task),
                    max_tokens=self.max_tokens,
                    http=worker.http,
                    system_prompt=(harness.system_prompt + skills_inst.listing) if skills_inst else None,
                )
                await skills.uninstall(sb, skills_inst)
            else:                                  # decoupled: runs in its own harness Runtime, drives sb
                res = await harness.run(
                    sb,
                    session_id=sid,
                    adapter_url=session_url,       # the proxy URL the loop dials (ANTHROPIC_BASE_URL)
                    prompt=self._problem_text(task),
                    arn=arn,                       # the per-episode sandbox Runtime the harness drives
                    max_turns=self.max_turns,
                    max_tokens=self.max_tokens,
                )
            ep.update(agent_ms=round((time.perf_counter() - ta) * 1000.0, 1), turns=res.turns,
                      tool_calls=res.tool_calls, stop=res.stop_reason,
                      harness_where=res.where)
            ep.update(skills.demo_fields(res.transcript, skills_inst))

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
            reward = self._reward(g, evaluation)
            ep["reward"] = round(reward, 4)

            samples = await adapter.finish_session(
                sid, base_sample=base_sample, reward=float(reward),
                extra_metadata={"task_id": task.task_id, "solved": bool(g["solved"]),
                                "harness_worker": wid, "sandbox_session": sb._session_id,
                                "skills_version": skills_inst.version if skills_inst else "off"},
            )
            out, ep["fan_out"] = one_sample(samples, float(reward))
            return out
        except BaseException as exc:
            reason = _cancel_reason(exc)
            if reason:
                ep["cancelled"] = reason
            else:
                ep["infra_error"] = type(exc).__name__
            raise
        finally:
            ep["episode_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
            ep["session_s"] = round(time.perf_counter() - sb._tb_opened_at, 1)  # microVM-seconds paid
            print("[demo] episode_end " + " ".join(f"{k}={v}" for k, v in ep.items()), flush=True)
            await adapter.drop_session(sid, wait_timeout=30)
            self.release_engine(e)
            await self._close_session(sb)

    async def _run_coding_episode(self, worker, task, sampling_params, base_sample, sb, arn, preopened,
                                  wait_ms, idle_ms, cold, evaluation: bool = False):
        """A routed non-Terminal-Bench row (KernelBench on the g6 tier). Same fresh, pre-opened, gated
        session as a Terminal-Bench episode, but the CODING episode runs in it: the agent writes
        solution.py, then the hidden checker (dropped in under a random name, nonce'd PASSED line)
        builds ModelNew vs the reference Model on the GPU — correct + on CUDA = reward 1; the measured
        speedup is reported as speedup=… on the [demo] line."""
        from . import mini_claude_episode, router_trace
        from .agentcore_runtime_sandbox import _arn_tail

        e, engine = self.acquire_engine()
        try:
            router_trace.emit(task_id=task.task_id, rollout_id=getattr(base_sample, "index", "?"),
                              harness_worker=worker.wid, engine=engine,
                              engine_pinned=int(len(self._engine_urls) > 1),
                              adapter_session=getattr(base_sample, "session_id", "?"),
                              sandbox_id=getattr(sb, "sandbox_id", "?"), sandbox_session=sb._session_id,
                              runtime_arn=_arn_tail(arn), sandbox_image=task.image, family=task.family)
            print(f"[demo] worker={worker.wid} qdepth={worker.qdepth} engine={e} engine_inflight={self.inflight(e)} "
                  f"fresh_session=1 substrate={sb._tb_substrate} preopened={int(preopened)} "
                  f"open_wait_ms={wait_ms:.1f} preopen_idle_ms={idle_ms:.1f} open_attempts={sb._tb_open_attempts} "
                  + " ".join(f"{k}={v}" for k, v in cold.items()) + f" arn={_arn_tail(arn)}", flush=True)
            return await mini_claude_episode.run(self.args, base_sample, task, sampling_params, sb, engine,
                                                 fields={"substrate": sb._tb_substrate, "infra_error": "none"},
                                                 evaluation=evaluation)
        except BaseException as exc:
            reason = _cancel_reason(exc)
            tag = f"infra_error=none cancelled={reason}" if reason else f"infra_error={type(exc).__name__}"
            print(f"[demo] task={task.task_id} family={task.family} substrate={sb._tb_substrate} solved=0 "
                  f"{tag}", flush=True)
            raise
        finally:
            self.release_engine(e)
            await self._close_session(sb)

    # ---- public entry point: start the episode's cold start at enqueue time --
    async def submit(self, task, sampling_params, base_sample, evaluation: bool = False):
        await self._ensure_started()
        arn = self._arn_for(task)          # fail fast on an unmapped task, before queueing
        if not (getattr(base_sample, "metadata", None) or {}).get("tb_reserved"):
            # Not announced by the look-ahead source (eval, or no source wired): start its boot now.
            self._demand.append(arn)
        self._refill()
        return await self.run(lambda worker: self._run_episode(worker, task, sampling_params, base_sample, evaluation))

    # ---- shutdown: stop any still-pre-opened sessions + the keeper -----------
    async def aclose(self) -> None:
        self._demand, self._future = [], []
        for q in list(self._ready.values()):
            for pending in q:
                pending.cancel()   # still booting: abort the open (it stops its session) instead of
                await self._discard(pending)   # waiting minutes for a g6 boot just to close it
        self._ready.clear()
        if self._keeper is not None:
            await self._keeper.aclose()
