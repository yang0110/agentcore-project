"""`harness_runtime_agent` — the harness as its OWN AgentCore Runtime (the decoupled target).

This is the *harness* half of the disaggregated design (docs: the harness runs on its own AgentCore
Runtime microVM, the sandbox is a SEPARATE Runtime microVM — 2 microVMs per episode). A
`BedrockAgentCoreApp` runs the agent loop here and drives the sandbox over `invoke_agent_runtime`,
so the sandbox session's lifecycle + the hidden grader stay owned by the trainer.

Wiring per rollout (driven by `decoupled_cli_harness.DecoupledCliHarness`):
    invoke_agent_runtime(harness ARN, harness_session_id, payload={
        "harness_kind": "claude",
        "prompt": ..., "adapter_url": ..., "session_id": ...,   # session_id = adapter Bearer key
        "sandbox_runtime_arn": ..., "sandbox_session_id": ...,  # reach the SAME sandbox the trainer opened
        "workdir": ..., "max_turns": ..., "model": "slime-actor", "time_budget_sec": ...,
    })
  The harness:
    1. runs the agent loop, calling the LLM through the PROXY (adapter_url), never SGLang direct;
    2. dispatches each tool call to the SANDBOX Runtime over invoke_agent_runtime (MCP bridge);
    3. returns a small summary {turns, tool_calls, stop_reason, ...} — the trajectory itself is
       captured proxy-side by slime's AnthropicAdapter, keyed on session_id (the Bearer token).

IMPORTANT: this Runtime's env MUST set AGENTCORE_STOP_SESSION=0 so the harness-side
`AgentCoreRuntimeSandbox.__aexit__` is a no-op — the trainer owns the sandbox session (it opened it,
heartbeats it, grades against it, and stops it). If the harness stopped it, the grader would lose
the microVM.

Two harness flavors, selected by HARNESS_KIND / payload "harness_kind":
  * "mini"   — reuse MiniClaudeHarness (the Python mimic) but with tools bound to the remote sandbox.
  * "claude" — run the real `claude -p` CLI, built-ins disabled, MCP bridge → sandbox (sandbox_mcp.py).
  * "noop"   — cheap liveness/warm ping used by the warm harness-session pool.
"""

from __future__ import annotations

import os
import re
import secrets

from .sandbox_mcp import MCP_TOOL_NAMES

try:
    from bedrock_agentcore.runtime import BedrockAgentCoreApp
except Exception:  # pragma: no cover - only importable inside the deployed container
    BedrockAgentCoreApp = None  # type: ignore


async def _run_mini(request: dict) -> dict:
    """Run the mini agent loop with tools bound to the *remote* sandbox Runtime."""
    from .agentcore_runtime_sandbox import AgentCoreRuntimeSandbox

    prompt = request["prompt"]
    adapter_url = request["adapter_url"]        # the PROXY, not SGLang
    session_id = request["session_id"]
    workdir = request.get("workdir", "/workspace")

    # The harness's tools execute in a SEPARATE sandbox Runtime, reached by ARN.
    sandbox = AgentCoreRuntimeSandbox(
        runtime_arn=request.get("sandbox_runtime_arn"),
        session_id=request.get("sandbox_session_id") or session_id,
        workdir=workdir,
    )

    # MiniClaudeHarness is OUR ~1-file agent (agentcore_slime_demo/mini_agent.py); import lazily so
    # this module still loads for unit-testing `handle` without the mini_agent deps.
    from .mini_agent import MiniClaudeHarness

    max_tokens = int(request.get("max_tokens") or os.environ.get("MINI_AGENT_MAX_TOKENS", "1024"))
    max_turns = int(request.get("max_turns") or os.environ.get("MINI_AGENT_MAX_TURNS", "8"))
    async with sandbox as sb:   # __aexit__ is a no-op here (AGENTCORE_STOP_SESSION=0) — trainer owns sb
        harness = MiniClaudeHarness(
            max_turns=max_turns,
            exec_timeout_sec=int(os.environ.get("TB_EXEC_TIMEOUT", "180")),
            system_prompt=request.get("system_prompt"),   # the TB persona, passed from the trainer
        )
        result = await harness.run(
            sb, session_id=session_id, adapter_url=adapter_url, prompt=prompt, max_tokens=max_tokens,
        )
    return {
        "ok": True,
        "harness_kind": "mini",
        "turns": result.turns,
        "stop_reason": result.stop_reason,
        "tool_calls": result.tool_calls,
        "last_exit_code": result.last_exit_code,
    }


# Claude Code built-in tools we DISABLE so every file/shell action goes through the MCP bridge
# (mcp__sandbox__*) into the remote sandbox Runtime, not the harness host FS. (Bash/Edit/Write/Read
# plus the search/web built-ins that could touch the harness host or the internet.) We ALSO disable
# the sub-agent / planning tools (Task, Agent, Explore, TodoWrite): a small/weak policy driven by
# Claude Code's system prompt kept emitting `Explore`/`Task` calls which SGLang rejects as
# "undefined function" — burning the turn budget and producing ZERO valid tool calls, so reward was
# structurally 0. Removing them from the offered set stops the model reaching for them.
_CLAUDE_DISALLOWED = (
    "Bash Edit Write Read NotebookEdit WebFetch WebSearch Glob Grep "
    "Task Agent Explore TodoWrite BashOutput KillShell SlashCommand"
)

# Steering system prompt (appended to Claude Code's own). The base model is weak and gravitates to
# the native tool names it was trained on (Bash/Read/Write) or the `Explore` sub-agent; here those
# are disabled and only the three MCP-bridged sandbox tools exist. Point the model at them
# explicitly so it actually writes a file and runs it in the sandbox instead of hallucinating tools.
_CLAUDE_STEER = (
    "You are solving a task inside a remote sandbox. You have EXACTLY three tools, all named with the "
    "prefix `mcp__sandbox__`:\n"
    "  - `mcp__sandbox__write_file(path, content)` — create/overwrite a file in the sandbox workspace;\n"
    "  - `mcp__sandbox__bash(command)` — run a shell command in the sandbox (compile, run, test);\n"
    "  - `mcp__sandbox__read_file(path)` — read a file back.\n"
    "Use ONLY these three tools. Do NOT call `Bash`, `Read`, `Write`, `Edit`, `Explore`, `Task`, "
    "`TodoWrite`, or any other tool — they DO NOT EXIST here and every such call fails and wastes a "
    "turn. To solve the task: write your solution with `mcp__sandbox__write_file`, then compile and "
    "run it with `mcp__sandbox__bash`. Be direct — do not plan or explore; act immediately."
)


def _serve_mcp_in_thread(sb, workdir: str):
    """Serve sandbox_mcp (bridges Claude's tools to the remote sandbox) on a local port.

    FastMCP exposes a Starlette/ASGI app (streamable_http_app), so we run it under uvicorn in a
    daemon thread. Returns (base URL of the /mcp endpoint the CLI connects to over localhost, server).
    """
    import socket
    import threading

    import uvicorn

    from .sandbox_mcp import build_sandbox_mcp

    mcp = build_sandbox_mcp(sb, workdir=workdir)
    app = mcp.streamable_http_app()

    # Grab a free localhost port (the CLI + server share this host, so 127.0.0.1 is enough).
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, name="sandbox-mcp", daemon=True).start()
    return f"http://127.0.0.1:{port}/mcp", server


def _count_stream(text: str) -> tuple[int, int]:
    """(assistant turns, sandbox tool calls) parsed from the CLI's --output-format stream-json tail.

    Best-effort and cheap: count assistant message events and occurrences of the three
    mcp__sandbox__* tool names. tool_calls==0 is the known-risk diagnostic (a weak policy that never
    emits a valid sandbox tool call), so surfacing it matters more than exactness.
    """
    if not text:
        return 0, 0
    turns = len(re.findall(r'"type"\s*:\s*"assistant"', text))
    tool_calls = sum(text.count(name) for name in MCP_TOOL_NAMES)
    return turns, tool_calls


async def _run_claude(request: dict) -> dict:
    """Decoupled CLI path: run the real `claude -p` here (the harness Runtime), built-ins disabled,
    its ONLY tools bridged via MCP to the remote sandbox Runtime. The LLM calls go through the proxy
    (adapter_url) exactly like the mini path, so the trajectory is captured proxy-side."""
    import asyncio
    import json
    import shlex

    from .agentcore_runtime_sandbox import AgentCoreRuntimeSandbox

    prompt = request["prompt"]
    adapter_url = request["adapter_url"]          # the PROXY, dialed back over the VPC
    session_id = request["session_id"]            # also the adapter Bearer key (trajectory id)
    workdir = request.get("workdir", "/workspace")
    model_label = request.get("model") or os.environ.get("ANTHROPIC_MODEL", "slime-actor")
    time_budget = int(request.get("time_budget_sec") or os.environ.get("AGENT_TIME_BUDGET_SEC", "900"))

    sandbox = AgentCoreRuntimeSandbox(
        runtime_arn=request.get("sandbox_runtime_arn"),
        session_id=request.get("sandbox_session_id") or session_id,
        workdir=workdir,
    )

    # A per-episode cwd so a REUSED warm harness session carries no ~/.claude/projects/<cwd> history
    # or stray files between episodes (claude -p keys session/history by cwd). Cleaned in `finally`.
    ep_cwd = f"/tmp/ep-{secrets.token_hex(8)}"

    async with sandbox as sb:   # __aexit__ is a no-op here (AGENTCORE_STOP_SESSION=0) — trainer owns sb
        mcp_url, server = _serve_mcp_in_thread(sb, workdir)
        mcp_config = json.dumps({"mcpServers": {"sandbox": {"type": "http", "url": mcp_url}}})
        # Same launch contract as slime's ClaudeCodeHarness, but run LOCALLY (subprocess) with the
        # sandbox MCP as the only toolset and every built-in tool disabled. Claude Code's native
        # --max-turns bounds assistant<->tool cycles; capping it shortens trajectories.
        max_turns = int(request.get("max_turns") or os.environ.get("CLAUDE_MAX_TURNS", "8"))
        cmd = (
            f"/usr/local/bin/claude -p {shlex.quote(prompt)} "
            "--permission-mode bypassPermissions --output-format stream-json "
            "--include-partial-messages --verbose "
            f"--max-turns {max_turns} "
            f"--append-system-prompt {shlex.quote(_CLAUDE_STEER)} "
            f"--mcp-config {shlex.quote(mcp_config)} "
            f"--allowedTools {' '.join(MCP_TOOL_NAMES)} "
            f"--disallowedTools {_CLAUDE_DISALLOWED}"
        )
        env = {
            **os.environ,
            "ANTHROPIC_BASE_URL": adapter_url,
            "ANTHROPIC_AUTH_TOKEN": session_id,
            "ANTHROPIC_MODEL": model_label,
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
            "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
            "IS_SANDBOX": "1",
        }
        exit_code, timed_out, out = 0, False, b""
        try:
            os.makedirs(ep_cwd, exist_ok=True)
            proc = await asyncio.create_subprocess_shell(
                cmd, cwd=ep_cwd, env=env,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            )
            try:
                out, _ = await asyncio.wait_for(proc.communicate(), timeout=time_budget)
                exit_code = proc.returncode or 0
            except asyncio.TimeoutError:
                proc.kill()
                timed_out = True
        finally:
            server.should_exit = True
            import shutil
            shutil.rmtree(ep_cwd, ignore_errors=True)

    tail = out.decode(errors="replace") if out else ""
    turns, tool_calls = _count_stream(tail)
    return {
        "ok": not timed_out,
        "harness_kind": "claude",
        "agent_exit_code": exit_code,
        "timed_out": timed_out,
        "turns": turns,
        "tool_calls": tool_calls,
        "stop_reason": "timeout" if timed_out else ("error" if exit_code else "end_turn"),
        "tail": tail[-2000:],
    }


async def handle_async(request: dict) -> dict:
    kind = (request or {}).get("harness_kind") or os.environ.get("HARNESS_KIND", "mini")
    if kind == "noop":                              # warm-pool open / heartbeat ping
        return {"ok": True, "harness_kind": "noop"}
    if kind == "mini":
        return await _run_mini(request)
    if kind == "claude":
        return await _run_claude(request)
    return {"error": f"unknown harness_kind {kind!r}"}


if BedrockAgentCoreApp is not None:
    app = BedrockAgentCoreApp()

    @app.entrypoint
    async def entrypoint(request):  # noqa: ANN001
        return await handle_async(request)

    if __name__ == "__main__":
        app.run()
