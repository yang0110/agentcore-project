"""HARNESS_WHERE=session: run the mini_claude loop INSIDE the sandbox session, not in the trainer.

DSec §6.2 pattern: one agent loop per sandbox, and the loop shares the sandbox's lifetime. The
trainer only (1) hosts slime's AnthropicAdapter, which records every turn's tokens, and (2) drives
the episode through the sandbox's ordinary ops:

    write_file .acs_h/runner.py + .acs_h/cfg.json     (session_runner.py, stdlib only)
    exec       nohup python3 .acs_h/runner.py ... &   (detached; the exec returns at once)
    exec       long-poll for .acs_h/done              (MINI_SESSION_POLL_SEC per call)
    read_file  .acs_h/status.json                     (turns, tool calls, stop, transcript)
    exec       kill + rm -rf .acs_h                   (before the hidden grader runs)

The runner dials the adapter back over the VPC (ADAPTER_PUBLIC_HOST:port; the host SG admits the
runtime SG on the adapter port range, SG->SG). `InSessionHarness.run` has MiniClaudeHarness.run's
signature and result type, so call sites swap the object and pass the adapter's public URL.

Fallback to the in-trainer loop (same adapter, local URL) when the session cannot host it: no
python3 on the image, or the adapter port is unreachable from the session (e.g. Code Interpreter,
which has no path back to the trainer). The adapter has recorded nothing in either case, so the
fallback starts from a clean session. `result.where` says which one ran.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import time
from pathlib import Path

from .mini_agent import _MAX_TOOL_OUTPUT_CHARS, TOOLS, MiniAgentResult, MiniClaudeHarness, _system_prompt

logger = logging.getLogger(__name__)

_RUNNER_SRC = (Path(__file__).with_name("session_runner.py")).read_text()
_DIR = ".acs_h"


def harness_where() -> str:
    """session (default) | trainer. Code Interpreter (SANDBOX=agentcore) has no route back to the
    adapter, so it defaults to the trainer; everything else (own Runtimes, farms, local) to session."""
    where = os.environ.get("HARNESS_WHERE", "").strip().lower()
    if where in ("session", "trainer"):
        return where
    return "trainer" if os.environ.get("SANDBOX", "agentcore") == "agentcore" else "session"


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
    """(host, port) to bind the index-th AnthropicAdapter on. In session mode it must be reachable
    from the sandbox: 0.0.0.0 on ADAPTER_PORT_BASE+index (the range the SG rule opens). Otherwise
    loopback on an ephemeral port, as before."""
    if harness_where() != "session":
        return "127.0.0.1", 0
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
        self._fallback = MiniClaudeHarness(max_turns=max_turns, request_timeout_sec=request_timeout_sec,
                                           exec_timeout_sec=exec_timeout_sec, system_prompt=system_prompt)

    async def run(self, sb, *, session_id: str, adapter_url: str, prompt: str, max_tokens: int = 1024,
                  model_label: str = "slime-actor", http=None, local_adapter_url: str | None = None,
                  ) -> MiniAgentResult:
        """``adapter_url`` = the URL the SESSION dials; ``local_adapter_url`` = the trainer's own
        URL for the fallback (defaults to adapter_url)."""
        res = await self._run_in_session(sb, session_id=session_id, adapter_url=adapter_url,
                                         prompt=prompt, max_tokens=max_tokens, model_label=model_label)
        if res is not None:
            return res
        res = await self._fallback.run(sb, session_id=session_id, adapter_url=local_adapter_url or adapter_url,
                                       prompt=prompt, max_tokens=max_tokens, model_label=model_label, http=http)
        res.where = "trainer_fallback"
        return res

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
        await sb.write_file(f"{_DIR}/runner.py", _RUNNER_SRC)
        await sb.write_file(f"{_DIR}/cfg.json", json.dumps(cfg))
        code, _out, err = await sb.exec(
            f"command -v python3 >/dev/null 2>&1 || exit 42; "
            f"nohup python3 {_DIR}/runner.py {_DIR}/cfg.json > {_DIR}/log 2>&1 < /dev/null & "
            f"echo $! > {_DIR}/pid", timeout=60)
        if code != 0:
            logger.warning("[in_session] session=%s cannot start the runner (exit %s%s); loop falls back "
                           "to the trainer", session_id, code, ", no python3" if code == 42 else f": {err[:200]}")
            await self._cleanup(sb)
            return None

        poll = int(os.environ.get("MINI_SESSION_POLL_SEC", "25"))
        hard_deadline = time.monotonic() + cfg["wall_sec"] + 120
        state = "done"
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
        try:
            status = json.loads(await sb.read_file(f"{_DIR}/status.json") or "{}")
        except Exception:  # noqa: BLE001 — partial/missing status -> counters default to 0
            status = {}
        await self._cleanup(sb)

        if status.get("stop") == "unreachable":
            logger.warning("[in_session] session=%s adapter %s unreachable from the session (%s); loop "
                           "falls back to the trainer", session_id, adapter_url, status.get("error"))
            return None
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
    """InSessionHarness when HARNESS_WHERE resolves to session, else the in-trainer MiniClaudeHarness."""
    return InSessionHarness(**kw) if harness_where() == "session" else MiniClaudeHarness(**kw)

