"""Fault tolerance for RL rollouts: sandboxes DIE during training, the trainer must not.

Why this exists: in agentic RL you run thousands of rollouts; each one boots a fresh sandbox
session. Sessions time out (AgentCore Code Interpreter default 900 s), microVMs get GC'd,
transport streams get severed, quotas throttle boots. If any of that crashes the rollout coro,
you lose the whole training step. The recovery contract here mirrors what slime's shipped stack
already does for E2B (repos/slime/slime/agent/sandbox.py `_rpc_retry` + examples/coding_agent_rl/
generate.py `boot_agent_sandbox` retries + `_abort_result`):

  1. RPC-level self-heal  — a terminated/severed session is transparently restarted and the
     op retried once (AgentCoreSandbox below; analogous to E2BSandbox._rpc_retry).
  2. Episode-level recovery — if a rollout fails mid-flight, recreate the sandbox and restart
     the episode on a fresh session, up to `max_boot_retries`, with backoff.
  3. Clean abort — if recovery is exhausted, return an ABORTED trajectory (reward 0), never
     raise. The trainer drops aborted samples from the batch and the step still runs.

AgentCore's own recovery semantics (see docs/fault-recovery.md for URLs + quotes):
  * A session's compute is an isolated microVM; it STOPS on 15-min idle, 8-h max lifetime,
    explicit stop, or a failed health check, and the microVM is destroyed + sanitized.
  * The logical session survives a stop: it transitions back to Active on the NEXT invocation
    and a fresh microVM is provisioned. Ephemeral in-VM state is lost unless session storage
    is configured — so a restart is a from-scratch re-run, which is why resilient_episode
    re-writes the workspace rather than trying to resume.
  * Health is GET /ping -> Healthy | HealthyBusy (HealthyBusy keeps the session alive).
  * 409 RetryableConflictException / 429 ThrottlingException / 5xx are transient -> retry with
    backoff (AWS SDKs auto-retry 409). 424 RuntimeClientError = your container 4xx/5xx.
  SDK surfaces used here: CodeInterpreter.get_session().status ∈ {READY, TERMINATED} (health),
  CodeInterpreter.invoke() auto-starts a session if none is active (code_interpreter_client.py).
"""

from __future__ import annotations

import asyncio
import logging
import random

from .harness import Trajectory, run_episode
from .task import CodingTask

logger = logging.getLogger(__name__)


class SandboxError(RuntimeError):
    """A sandbox-side failure (boot, exec transport, session death)."""


class TransientSandboxError(SandboxError):
    """Recoverable: retry / recreate the sandbox."""


class FatalSandboxError(SandboxError):
    """Not recoverable by retry (bad config, auth, quota-permanent)."""


# botocore/boto3 error names + message fragments that mean "the session is gone, get a new one".
_RECOVERABLE_MARKERS = (
    "terminated", "not found", "notfound", "expired", "session is not",
    "throttl", "timeout", "timed out", "connection", "unavailable",
    "internalserver", "serviceunavailable", "resourcenotfound",
)


def is_recoverable(exc: BaseException) -> bool:
    """Classify an exception as recoverable (retry on a fresh session) or fatal."""
    if isinstance(exc, FatalSandboxError):
        return False
    if isinstance(exc, TransientSandboxError):
        return True
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    if "accessdenied" in name or "accessdenied" in msg or "unauthorized" in msg:
        return False  # auth/permission → retrying won't help
    return any(m in name or m in msg for m in _RECOVERABLE_MARKERS)


async def resilient_episode(
    task: CodingTask,
    make_policy,          # () -> Policy   (fresh per attempt; policies can be stateful)
    make_sandbox,         # () -> Sandbox  (fresh session per attempt)
    *,
    sampling_params: dict | None = None,
    max_boot_retries: int = 3,
    backoff_base_sec: float = 0.5,
    backoff_cap_sec: float = 8.0,
) -> Trajectory:
    """Run one episode with automatic sandbox recovery.

    On a recoverable failure (boot or mid-episode), recreate the sandbox + policy and restart
    the episode on a fresh session. A microVM crash loses in-sandbox state, so recovery = a
    clean restart, not a resume — which is exactly right for a from-scratch coding task. If all
    attempts fail, return an ABORTED trajectory (reward 0) instead of raising.
    """
    sampling_params = sampling_params or {"max_new_tokens": 1024, "temperature": 1.0}
    last_err: BaseException | None = None

    for attempt in range(1, max_boot_retries + 1):
        try:
            sandbox = make_sandbox()
            async with sandbox as sb:
                traj = await run_episode(make_policy(), sb, task, sampling_params)
            traj.boot_attempts = attempt
            return traj
        except BaseException as e:  # noqa: BLE001 — we deliberately catch broadly, then classify
            last_err = e
            if not is_recoverable(e):
                logger.warning("[recovery] %s: fatal sandbox error, aborting: %r", task.task_id, e)
                break
            if attempt < max_boot_retries:
                backoff = min(backoff_cap_sec, backoff_base_sec * (2 ** (attempt - 1)))
                backoff = random.uniform(0, backoff)  # jitter, like E2BSandbox._rpc_retry
                logger.warning(
                    "[recovery] %s: recoverable sandbox failure (attempt %d/%d), "
                    "recreating in %.1fs: %r",
                    task.task_id, attempt, max_boot_retries, backoff, e,
                )
                await asyncio.sleep(backoff)
            else:
                logger.warning("[recovery] %s: exhausted %d boots, aborting", task.task_id, max_boot_retries)

    return _aborted(task, last_err, attempt)


def _aborted(task: CodingTask, err: BaseException | None, attempts: int) -> Trajectory:
    """Aborted-rollout trajectory: zero reward, marked so the trainer can drop it."""
    return Trajectory(
        task_id=task.task_id,
        reward=0.0,
        solved=False,
        status="aborted",
        abort_reason=f"{type(err).__name__}: {err}" if err else "unknown",
        boot_attempts=attempts,
    )


class FlakySandbox:
    """Test helper: wraps a real sandbox and injects failures to exercise recovery.

    Not for production — it's how the fault-recovery tests simulate session death without AWS.
      * fail_boots=k         -> the first k __aenter__ calls raise (transient boot failure)
      * fail_exec_after=n     -> the doomed session dies after its n-th exec (mid-episode death)
      * exec_death_count=m    -> how many *distinct sessions* die that way before one survives
      * fatal=True            -> raise a FatalSandboxError instead of a transient one

    Boot counting is class-level (so `fail_boots` spans the fresh instances a retry loop makes),
    but exec counting is PER INSTANCE with a one-shot death latch: only the first `exec_death_count`
    booted sessions die mid-episode; the next fresh session runs clean. That models a real session
    dying under load and recovery succeeding on a new one — not "every exec fails forever".
    """

    _boot_calls = 0
    _exec_deaths_done = 0

    def __init__(self, inner, *, fail_boots: int = 0, fail_exec_after: int | None = None,
                 exec_death_count: int = 1, fatal: bool = False) -> None:
        self._inner = inner
        self._fail_boots = fail_boots
        self._fail_exec_after = fail_exec_after
        self._exec_death_count = exec_death_count
        self._fatal = fatal
        self._exec_calls = 0        # per-instance
        self._doomed = False        # set at boot if this session is slated to die
        self.sandbox_id = inner.sandbox_id

    @classmethod
    def reset_counters(cls) -> None:
        cls._boot_calls = 0
        cls._exec_deaths_done = 0

    def _boom(self, where: str):
        exc = FatalSandboxError if self._fatal else TransientSandboxError
        return exc(f"injected {'fatal' if self._fatal else 'transient'} failure at {where}")

    async def __aenter__(self):
        type(self)._boot_calls += 1
        if type(self)._boot_calls <= self._fail_boots:
            raise self._boom(f"boot#{type(self)._boot_calls}")
        # This freshly-booted session is doomed only if we haven't used up the death budget.
        self._doomed = (
            self._fail_exec_after is not None
            and type(self)._exec_deaths_done < self._exec_death_count
        )
        await self._inner.__aenter__()
        self.sandbox_id = self._inner.sandbox_id
        return self

    async def __aexit__(self, *exc):
        return await self._inner.__aexit__(*exc)

    async def exec(self, cmd, **kw):
        self._exec_calls += 1
        if self._doomed and self._exec_calls > self._fail_exec_after:
            self._doomed = False
            type(self)._exec_deaths_done += 1
            raise self._boom(f"exec#{self._exec_calls} (session death)")
        return await self._inner.exec(cmd, **kw)

    async def write_file(self, *a, **kw):
        return await self._inner.write_file(*a, **kw)

    async def read_file(self, *a, **kw):
        return await self._inner.read_file(*a, **kw)
