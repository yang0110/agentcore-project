"""`harness_runtime_agent` — the harness as its OWN AgentCore Runtime (Runtime #1).

STATUS: untested scaffolding for the two-Runtime target (docs/harness-sandbox-decoupling.md §7).
This is the *harness* half of the disaggregated target: a `BedrockAgentCoreApp` that runs the
agent loop on its own AgentCore Runtime, deployed separately from the sandbox Runtime (#2). It is
the AgentCore-hosted replacement for the two current shortcuts — the in-process `MiniClaudeHarness`
(A3/B3) and the in-sandbox Claude Code CLI (A2/B2).

Wiring per rollout (driven by agentcore_runtime_generate.generate):
    invoke_agent_runtime(harness ARN, session_id, payload={
        "prompt": ..., "adapter_url": ..., "session_id": ..., "model": ...,
        "sandbox_runtime_arn": ...,        # so the harness can reach Runtime #2
    })
  The harness:
    1. runs the agent loop, calling the LLM through the PROXY (adapter_url), never SGLang direct;
    2. dispatches each tool call to the SANDBOX Runtime (#2) over invoke_agent_runtime;
    3. returns {"turns": n, "stopped": reason} — the trajectory itself is captured proxy-side.

Two harness flavors, selected by HARNESS_KIND:
  * "mini"  — reuse MiniClaudeHarness (A3/B3 loop) but with tools bound to the remote sandbox.
  * "claude"— run the real `claude -p` CLI (A2/B2), built-ins disabled, MCP bridge → sandbox.
             (Only the mini path is sketched here; the claude path reuses sandbox_mcp.py.)

Nothing here has run against AWS. Deploy exactly like sandbox_server_agent.py (its own image +
`gpu_deploy.py runtime-on-cp`), export the ARN as AGENTCORE_HARNESS_RUNTIME_ARN.
"""

from __future__ import annotations

import os

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

    # The harness's tools execute in a SEPARATE sandbox Runtime (#2), reached by ARN.
    sandbox = AgentCoreRuntimeSandbox(
        runtime_arn=request.get("sandbox_runtime_arn"),
        session_id=request.get("sandbox_session_id") or session_id,
        workdir=workdir,
    )

    # MiniClaudeHarness is OUR ~1-file agent (agentcore_slime_demo/mini_agent.py), not a slime
    # class; import lazily so this module still loads for unit-testing `handle` without deps.
    from .mini_agent import MiniClaudeHarness

    max_tokens = int(request.get("max_tokens") or os.environ.get("MINI_AGENT_MAX_TOKENS", "1024"))
    # Per-episode turn cap: prefer the value the generate hook passes in the payload, else env, else 3.
    max_turns = int(request.get("max_turns") or os.environ.get("MINI_AGENT_MAX_TURNS", "3"))
    async with sandbox as sb:
        harness = MiniClaudeHarness(max_turns=max_turns)
        result = await harness.run(
            sb, session_id=session_id, adapter_url=adapter_url, prompt=prompt, max_tokens=max_tokens,
        )
    # The return value is JSON-serialized by BedrockAgentCoreApp, so hand back plain fields only
    # (MiniAgentResult is a dataclass and would raise "not JSON serializable"). The trajectory
    # itself is captured proxy-side; the caller (agentcore_runtime_generate) only needs a summary.
    return {
        "ok": True,
        "turns": result.turns,
        "stop_reason": result.stop_reason,
        "tool_calls": result.tool_calls,
        "last_exit_code": result.last_exit_code,
    }


# Claude Code built-in tools we DISABLE so every file/shell action goes through the MCP bridge
# (mcp__sandbox__*) into the remote sandbox Runtime, not the harness host FS. (Bash/Edit/Write/Read
# plus the search/web built-ins that could touch the harness host or the internet.) We ALSO disable
# the sub-agent / planning tools (Task, Agent, Explore, TodoWrite): a small policy driven by Claude
# Code's system prompt kept emitting `Explore`/`Task` calls (74x in the first b2 run) which SGLang
# rejects as "undefined function" — burning the turn budget and producing ZERO valid tool calls, so
# reward was structurally 0. Removing them from the offered set stops the model reaching for them.
_CLAUDE_DISALLOWED = (
    "Bash Edit Write Read NotebookEdit WebFetch WebSearch Glob Grep "
    "Task Agent Explore TodoWrite BashOutput KillShell SlashCommand"
)

# Steering system prompt (appended to Claude Code's own). The base model is weak and gravitates to the
# native tool names it was trained on (Bash/Read/Write) or the `Explore` sub-agent; here those are
# disabled and only the three MCP-bridged sandbox tools exist. Point the model at them explicitly so
# it actually writes a file and runs it in the sandbox instead of hallucinating undefined tools.
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
    daemon thread — slime's aiohttp run_app_in_thread can't host an ASGI app. Returns the base URL
    of the /mcp endpoint the CLI connects to over localhost."""
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


async def _run_claude(request: dict) -> dict:
    """Two-Runtime CLI path: run the real `claude -p` here (Runtime #1), built-ins disabled, its
    ONLY tools bridged via MCP to the remote sandbox Runtime (#2). The LLM calls go through the
    proxy (adapter_url) exactly like the mini path, so the trajectory is captured proxy-side."""
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

    async with sandbox as sb:
        mcp_url, server = _serve_mcp_in_thread(sb, workdir)
        mcp_config = json.dumps({"mcpServers": {"sandbox": {"type": "http", "url": mcp_url}}})
        # Same launch contract as slime's ClaudeCodeHarness, but run LOCALLY (subprocess) with the
        # sandbox MCP as the only toolset and every built-in tool disabled.
        # Per-episode agentic-turn cap (payload > env > 3). Claude Code's native --max-turns bounds
        # assistant<->tool cycles; capping it shortens trajectories (cheaper rollout + ref/actor fwd).
        max_turns = int(request.get("max_turns") or os.environ.get("CLAUDE_MAX_TURNS", "3"))
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
        proc = await asyncio.create_subprocess_shell(
            cmd, cwd="/tmp", env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        exit_code, timed_out = 0, False
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=time_budget)
            exit_code = proc.returncode or 0
        except asyncio.TimeoutError:
            proc.kill()
            out, timed_out = b"", True
        finally:
            server.should_exit = True

    return {
        "ok": not timed_out,
        "harness_kind": "claude",
        "agent_exit_code": exit_code,
        "timed_out": timed_out,
        "tail": (out.decode(errors="replace")[-2000:] if out else ""),
    }


async def handle_async(request: dict) -> dict:
    kind = (request or {}).get("harness_kind") or os.environ.get("HARNESS_KIND", "mini")
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
