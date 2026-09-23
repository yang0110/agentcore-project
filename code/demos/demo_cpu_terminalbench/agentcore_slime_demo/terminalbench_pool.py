"""Warm-SHARED harness pool for demo_cpu_terminalbench — the RELIABLE warm path for the image pool.

The demo's whole point is a LARGE image pool (one container image / one AgentCore Runtime per
Terminal-Bench task). Opening a fresh session per episode would pay a cold start (allocate microVM +
pull that task's image + boot) on the FIRST episode of every task, over and over. This pool instead
uses the one guaranteed warm lever AgentCore gives us — **session affinity** — the reliable way:

  1. ONE live runtimeSessionId per task (per Runtime ARN), reused across every episode routed to
     that task. AgentCore scopes a session per (arn, sessionId), so one warm microVM per task.
  2. Per-episode workdir ISOLATION via a sub_session: the server (SANDBOX_SHARED_MODE=cpu) resets a
     pristine copy of the task's baked workdir template per episode, so episodes share the warm
     microVM but not each other's workdir state.
  3. A process-wide SessionKeeper HEARTBEAT holds each task's session warm across slime's
     trainer->inference weight-sync stall (when generation goes silent and AgentCore would otherwise
     idle-reclaim the microVM). We register ONE keepalive "anchor" per task ARN; the keeper pings any
     anchor that goes idle past ping_fraction*idle_timeout.

This subclasses the harness-scaling demo's HarnessPool (W warm in-process workers + one adapter per
worker, so an episode is sticky to its worker => sticky to one SGLang engine) and swaps only the
sandbox tier: a warm-shared session routed to the episode's task, instead of a fresh round-robin
session. The agentic loop is the SAME MiniClaudeHarness (Anthropic Messages API -> AnthropicAdapter
-> policy under training; tools run in the sandbox), so this exercises the faithful tool-calling path
(needs --sglang-tool-call-parser qwen25 + --sglang-reasoning-parser qwen3).

Correctness escape hatch: warm reuse isolates the WORKDIR, not global OS state. For a task that
mutates global state (apt install, /etc, /root) and grades on it, set TB_SESSION_RECYCLE_EVERY=K to
mint a fresh session id for that task every K episodes (one amortized cold start per K), trading a
little warmth for a clean OS. Default 0 = never recycle (pure warm).

Wire with:
    --custom-generate-function-path agentcore_slime_demo.terminalbench_pool_generate.generate
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import threading

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
    """HarnessPool whose sandbox tier is a warm-SHARED session per task + a heartbeat keeper."""

    # Terminal-Bench tasks act through the shell, not by writing a solution.py that a checker runs.
    # Override the coding-shaped prompt so the agent uses its tools against the working directory.
    _AGENT_PROMPT = (
        "Read PROBLEM.md in the current directory and complete the task it describes. Use the bash "
        "tool to inspect the environment and make changes, and write_file to create or edit files. "
        "Verify your work with shell commands as you go. When the task is complete, stop."
    )

    def __init__(self, args) -> None:
        super().__init__(args)
        self.sandbox_kind = "agentcore-warm-shared"
        # Serverless microVM idle timeout is ~15 min; heartbeat at half that with margin.
        self.idle_timeout = float(os.environ.get("WARM_IDLE_TIMEOUT", "900"))
        self.recycle_every = int(os.environ.get("TB_SESSION_RECYCLE_EVERY", "0"))  # 0 = never
        self._keeper = None
        # Per-task (per-ARN) warm-session bookkeeping. The shared session id is OWNED here (not the
        # env constant) so we can recycle it per task without touching other tasks.
        self._anchors: dict[str, object] = {}        # arn -> live keepalive anchor sandbox
        self._session_ids: dict[str, str] = {}       # arn -> current shared runtimeSessionId
        self._ep_counts: dict[str, int] = {}         # arn -> episodes run since last recycle
        self._anchor_lock = asyncio.Lock()

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
            # Always-on idle heartbeat: hold any warm session that goes idle (e.g. across the
            # weight-sync stall) without gating real ops. This is what keeps the pool warm.
            self._keeper.keep_warm_begin()
            logger.info("[terminalbench_pool] SessionKeeper started (idle_timeout=%.0fs, recycle_every=%d)",
                        self.idle_timeout, self.recycle_every)

    def _new_session_id(self) -> str:
        # >=33 chars for AgentCore runtimeSessionId (token_hex(16)=32 + prefix clears it).
        return f"tbwarm-{secrets.token_hex(16)}"

    async def _ensure_anchor(self, arn: str, region: str) -> str:
        """Ensure a live keepalive session for this task's Runtime; return its shared session id.

        Opens ONE anchor sandbox (warms the microVM) and registers a pinger with the keeper so the
        session survives idle stalls. Recycles (fresh session id + new anchor) every recycle_every
        episodes when that knob is on."""
        from .agentcore_runtime_sandbox import _arn_tail
        from .sandbox import make_sandbox

        async with self._anchor_lock:
            count = self._ep_counts.get(arn, 0)
            recycle = self.recycle_every and count and count % self.recycle_every == 0
            if arn in self._anchors and not recycle:
                return self._session_ids[arn]

            if recycle and arn in self._anchors:
                # Retire the old warm session: unregister its heartbeat + stop it so the quota slot
                # frees, then fall through to open a fresh one.
                self._keeper.unregister(arn)
                old = self._anchors.pop(arn)
                try:
                    await old.__aexit__(None, None, None)
                except Exception:  # noqa: BLE001 — teardown; idle reclaim is the fallback
                    pass

            sid = self._new_session_id()
            anchor = make_sandbox(
                "agentcore-warm-shared", region=region, runtime_arn=arn,
                shared_session_id=sid, sub_session=f"anchor-{secrets.token_hex(8)}",
            )
            await anchor.__aenter__()          # warms the microVM (allocate + pull + boot + open)
            self._anchors[arn] = anchor
            self._session_ids[arn] = sid

            async def _ping(a=anchor) -> None:
                # A raw no-op invoke on the shared session resets AgentCore's idle clock.
                await a.exec("true", timeout=30)

            self._keeper.register(arn, _ping, self.idle_timeout)
            # DEMO-SPECIFIC metric ([demo] channel): a warm session per task came up / was recycled.
            # metrics.py value-counts warm_anchor and tracks live_sessions (= tasks kept warm).
            print(f"[demo] warm_anchor arn={_arn_tail(arn)} session={sid} "
                  f"recycled={int(bool(recycle))} live_sessions={self._keeper.live_sessions}",
                  flush=True)
            return sid

    # ---- the episode: warm-shared sandbox routed to THIS task ----------------
    async def _run_episode(self, wid: int, http, task, sampling_params, base_sample):
        from . import router_trace
        from .agentcore_runtime_sandbox import _arn_tail
        from .sandbox import _runtime_arn_for_image, make_sandbox

        adapter = self._adapters[wid]
        adapter_url = self._adapter_urls[wid]
        engine = self._engine_urls[wid]

        # Adapter Bearer session id == trajectory key; AgentCore runtimeSessionId needs len>=33.
        sid = getattr(base_sample, "session_id", None)
        if not sid or len(sid) < 33:
            sid = f"tb-{task.task_id}-{secrets.token_hex(16)}"
        base_sample.session_id = sid
        adapter.open_session(sid, sampling_defaults=sampling_params)

        # Route to THIS task's Runtime, then reuse its ONE warm shared session (kept alive by the
        # keeper anchor). A per-episode sub_session gives the isolated, freshly-reset workdir.
        arn = _runtime_arn_for_image(getattr(task, "image", task.task_id))
        if arn is None:
            raise RuntimeError(
                "no Runtime ARN for task "
                f"{task.task_id!r} — set AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE (see run.sh)."
            )
        shared_sid = await self._ensure_anchor(arn, self.region)
        self._ep_counts[arn] = self._ep_counts.get(arn, 0) + 1

        sandbox = make_sandbox(
            "agentcore-warm-shared", region=self.region, runtime_arn=arn,
            shared_session_id=shared_sid, sub_session=f"ep-{secrets.token_hex(16)}",
        )
        try:
            async with sandbox as sb:
                router_trace.emit(
                    task_id=task.task_id,
                    rollout_id=getattr(base_sample, "index", "?"),
                    harness_worker=wid,
                    engine=engine,
                    engine_pinned=int(len(self._engine_urls) > 0 and engine != ""),
                    adapter_session=sid,
                    sandbox_id=getattr(sb, "sandbox_id", "?"),
                    sandbox_session=getattr(sb, "_session_id", getattr(sb, "_sub", "?")),
                    runtime_arn=_arn_tail(arn),
                )
                # DEMO-SPECIFIC metric: which warm worker drove this, queue depth, and — the demo's
                # signal — that this episode REUSED an already-warm session (shared id + fresh sub).
                print(f"[demo] worker={wid} qdepth={self._queue.qsize() if self._queue else 0} "
                      f"warm_reuse=1 arn={_arn_tail(arn)} ep_on_session={self._ep_counts[arn]}",
                      flush=True)

                # Seed only the instruction; the task's environment is already baked + reset into the
                # workdir by the server's op=open. No run_checks.py (the oracle is the baked grade.sh).
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
                                "harness_worker": wid, "warm_session": shared_sid},
            )
        finally:
            await adapter.drop_session(sid, wait_timeout=30)

    # ---- shutdown: release the warm sessions + stop the keeper ---------------
    async def aclose(self) -> None:
        if self._keeper is not None:
            await self._keeper.aclose()
        for anchor in list(self._anchors.values()):
            try:
                await anchor.__aexit__(None, None, None)
            except Exception:  # noqa: BLE001 — teardown best-effort; idle reclaim is the fallback
                pass
        self._anchors.clear()
