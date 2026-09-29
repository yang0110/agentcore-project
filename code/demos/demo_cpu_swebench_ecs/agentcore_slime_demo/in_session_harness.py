"""The mini_claude loop runs INSIDE the sandbox session, never in the trainer (every demo, no opt-out).

DSec §6.2 pattern: one agent loop per sandbox, and the loop shares the sandbox's lifetime. The
trainer only (1) hosts slime's AnthropicAdapter, which records every turn's tokens, and (2) drives
the episode through the sandbox's ordinary ops:

    write_file .acs_h/runner.py + .acs_h/cfg.json     (session_runner.py, stdlib only)
    exec       nohup python3 .acs_h/runner.py ... &   (detached; the exec returns at once)
    exec       long-poll for .acs_h/done              (MINI_SESSION_POLL_SEC per call)
    read_file  .acs_h/status.json                     (turns, tool calls, stop, transcript)
    exec       kill + rm -rf .acs_h                   (before the hidden grader runs)

MINI_SESSION_MODE=foreground (demo_cpu_instance_pool) replaces the nohup + long-poll with ONE exec that
runs the runner to completion: on the jailed CPU farm every exec is a Landlock jail whose subreaper
kills the command's whole process tree when the exec returns, so a detached runner would die at once.
The loop's wall budget is then capped below INVOKE_READ_TIMEOUT (one invoke must outlive the loop).

The runner dials the adapter back over the VPC (ADAPTER_PUBLIC_HOST:port; the host SG admits the
runtime SG on the adapter port range, SG->SG). `InSessionHarness.run` has MiniClaudeHarness.run's
signature and result type, so call sites swap the object and pass the adapter's public URL.

There is NO in-trainer fallback: if the session cannot host the loop (no python3 on the image, or the
adapter port is unreachable from the session — a missing SG->SG rule, or a backend with no route back
such as the Code Interpreter) the episode raises HarnessPlacementError. Fix the image or the network;
the loop never silently moves into the trainer. HARNESS_WHERE=trainer is rejected.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import socket
import time
from pathlib import Path

from .mini_agent import _MAX_TOOL_OUTPUT_CHARS, TOOLS, MiniAgentResult, _system_prompt

logger = logging.getLogger(__name__)

_RUNNER_SRC = (Path(__file__).with_name("session_runner.py")).read_text()
_DIR = ".acs_h"


class HarnessPlacementError(RuntimeError):
    """The sandbox session cannot host the agent loop (no python3, or no route back to the adapter)."""


def harness_where() -> str:
    """Always ``session``: the agent loop runs in the sandbox session in every demo. HARNESS_WHERE is
    accepted only as ``session`` (or unset); ``trainer`` was removed 2026-09-28."""
    where = os.environ.get("HARNESS_WHERE", "").strip().lower()
    if where not in ("", "session"):
        raise HarnessPlacementError(f"HARNESS_WHERE={where!r} is not supported: the agent loop always runs "
                                    "in the sandbox session (the in-trainer harness was removed)")
    return "session"


def _public_host() -> str:
    host = os.environ.get("ADAPTER_PUBLIC_HOST", "").strip()
    if host:
        return host
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)   # no packet is sent; picks the egress IP
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def adapter_bind(index: int) -> tuple[str, int]:
    """(host, port) to bind the index-th AnthropicAdapter on. It must be reachable from the sandbox:
    0.0.0.0 on ADAPTER_PORT_BASE+index (the range the SG rule opens)."""
    base = int(os.environ.get("ADAPTER_PORT_BASE", "18100"))
    span = int(os.environ.get("ADAPTER_PORT_SPAN", "64"))
    if index >= span:
        raise RuntimeError(f"adapter #{index} is outside ADAPTER_PORT_BASE..+{span} (the SG-opened range)")
    return os.environ.get("ADAPTER_BIND_HOST", "0.0.0.0"), base + index


def adapter_urls(port: int) -> tuple[str, str]:
    """(url the trainer uses, url the sandbox session dials) for an adapter bound on ``port``."""
    return f"http://127.0.0.1:{port}", f"http://{_public_host()}:{port}"


class InSessionHarness:
    """MiniClaudeHarness, but the loop runs in the sandbox session (see the module docstring)."""

    name = "mini_claude"
    in_session = True   # callers hand it the adapter URL the SESSION dials, not the loopback one

    def __init__(self, *, max_turns: int = 12, request_timeout_sec: int = 300,
                 exec_timeout_sec: int = 120, system_prompt: str | None = None) -> None:
        self.max_turns = max_turns
        self.request_timeout_sec = request_timeout_sec
        self.exec_timeout_sec = exec_timeout_sec
        self.system_prompt = system_prompt

    async def run(self, sb, *, session_id: str, adapter_url: str, prompt: str, max_tokens: int = 1024,
                  model_label: str = "slime-actor", http=None) -> MiniAgentResult:
        """``adapter_url`` = the URL the SESSION dials (adapter_urls()[1]). ``http`` is unused (the
        runner makes its own requests); kept so the signature matches MiniClaudeHarness.run."""
        return await self._run_in_session(sb, session_id=session_id, adapter_url=adapter_url,
                                          prompt=prompt, max_tokens=max_tokens, model_label=model_label)

    async def _run_in_session(self, sb, *, session_id, adapter_url, prompt, max_tokens, model_label):
        cfg = {
            "adapter_url": adapter_url, "session_id": session_id, "prompt": prompt, "model": model_label,
            "max_tokens": int(max_tokens), "system": _system_prompt(self.system_prompt), "tools": TOOLS,
            "max_turns": self.max_turns, "exec_timeout": self.exec_timeout_sec,
            "request_timeout": self.request_timeout_sec, "max_tool_chars": _MAX_TOOL_OUTPUT_CHARS,
            "connect_timeout": int(os.environ.get("MINI_SESSION_CONNECT_SEC", "5")),
            "wall_sec": int(os.environ.get("MINI_SESSION_WALL_SEC",
                                           str(self.max_turns * (self.request_timeout_sec + self.exec_timeout_sec)))),
        }
        foreground = os.environ.get("MINI_SESSION_MODE", "detach").strip().lower() == "foreground"
        if foreground:   # the whole loop lives inside one invoke: keep it under the read timeout
            cfg["wall_sec"] = min(cfg["wall_sec"], int(os.environ.get("INVOKE_READ_TIMEOUT", "900")) - 90)
        await sb.write_file(f"{_DIR}/runner.py", _RUNNER_SRC)
        await sb.write_file(f"{_DIR}/cfg.json", json.dumps(cfg))
        if foreground:
            return await self._run_foreground(sb, cfg, session_id=session_id, adapter_url=adapter_url,
                                              prompt=prompt)
        code, _out, err = await sb.exec(
            f"command -v python3 >/dev/null 2>&1 || exit 42; "
            f"nohup python3 {_DIR}/runner.py {_DIR}/cfg.json > {_DIR}/log 2>&1 < /dev/null & "
            f"echo $! > {_DIR}/pid", timeout=60)
        if code != 0:
            await self._cleanup(sb)
            raise HarnessPlacementError(
                f"session={session_id} cannot start the in-session runner (exit {code}"
                f"{': no python3 on the sandbox image' if code == 42 else ': ' + err[:200]})")

        poll = int(os.environ.get("MINI_SESSION_POLL_SEC", "25"))
        hard_deadline = time.monotonic() + cfg["wall_sec"] + 120
        state = "done"
        try:
            while True:
                code, _out, _err = await sb.exec(
                    f"timeout {poll} bash -c 'while [ ! -f {_DIR}/done ]; do "
                    f"kill -0 $(cat {_DIR}/pid) 2>/dev/null || exit 3; sleep 0.5; done'", timeout=poll + 30)
                if code == 0:
                    break
                if code == 3:                        # runner gone without `done` (killed by the agent?)
                    state = "runner_died"
                    break
                if time.monotonic() > hard_deadline:
                    state = "poll_deadline"
                    break
        except BaseException:
            # A failed/cancelled poll must not orphan the detached runner: on a pooled or shared
            # session it would keep calling the adapter next to the following episode.
            with contextlib.suppress(Exception):
                await self._cleanup(sb)
            raise
        return await self._finish(sb, state, session_id=session_id, adapter_url=adapter_url, prompt=prompt)

    async def _run_foreground(self, sb, cfg, *, session_id, adapter_url, prompt):
        """One exec runs the runner to completion (the jailed farm kills anything an exec leaves behind)."""
        code, _out, err = await sb.exec(
            f"command -v python3 >/dev/null 2>&1 || exit 42; "
            f"python3 {_DIR}/runner.py {_DIR}/cfg.json > {_DIR}/log 2>&1 < /dev/null", timeout=cfg["wall_sec"] + 60)
        if code == 42:
            await self._cleanup(sb)
            raise HarnessPlacementError(f"session={session_id} cannot start the in-session runner "
                                        "(exit 42: no python3 on the sandbox image)")
        state = "done" if code == 0 else ("poll_deadline" if code == 124 else "runner_died")
        if state != "done":
            logger.warning("[in_session] session=%s foreground runner exit=%s %s", session_id, code, (err or "")[:200])
        return await self._finish(sb, state, session_id=session_id, adapter_url=adapter_url, prompt=prompt)

    async def _finish(self, sb, state, *, session_id, adapter_url, prompt):
        try:
            status = json.loads(await sb.read_file(f"{_DIR}/status.json") or "{}")
        except Exception:  # noqa: BLE001 — partial/missing status -> counters default to 0
            status = {}
        await self._cleanup(sb)

        if status.get("stop") == "unreachable":
            raise HarnessPlacementError(
                f"session={session_id}: adapter {adapter_url} unreachable from the session ({status.get('error')}) "
                "— check networkMode=VPC + the SG->SG rule on ADAPTER_PORT_BASE.. (docs/agentcore-networking.md)")
        if status.get("stop") == "error":
            raise RuntimeError(f"in-session runner failed: {status.get('error')}")
        res = MiniAgentResult(turns=int(status.get("turns", 0)), tool_calls=int(status.get("tool_calls", 0)),
                              stop_reason=status.get("stop", "none") if state == "done" else state,
                              last_exit_code=status.get("last_exit_code"),
                              transcript=status.get("transcript") or [{"role": "user", "content": prompt}])
        res.where = "session"
        logger.info("[in_session] session=%s turns=%d tool_calls=%d stop=%s last_exit=%s",
                    session_id, res.turns, res.tool_calls, res.stop_reason, res.last_exit_code)
        return res

    @staticmethod
    async def _cleanup(sb) -> None:
        """Stop the runner and remove its files before anything grades the working directory."""
        try:
            await sb.exec(f"[ -f {_DIR}/pid ] && kill -9 $(cat {_DIR}/pid) 2>/dev/null; rm -rf {_DIR}", timeout=30)
        except Exception:  # noqa: BLE001 — teardown only; the session is stopped right after
            pass


def make_harness(**kw):
    """The harness every demo uses: the loop in the sandbox session."""
    harness_where()          # rejects HARNESS_WHERE=trainer up front
    return InSessionHarness(**kw)
