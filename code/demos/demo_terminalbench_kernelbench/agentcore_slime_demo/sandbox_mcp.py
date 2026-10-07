"""`sandbox_mcp` — expose a live `Sandbox` as MCP tools for a black-box CLI harness.

Used by the decoupled `claude_cli` harness (harness_runtime_agent.py): the real Claude Code CLI
runs with its built-in Bash/Edit/Write/Read tools DISABLED and only these `mcp__sandbox__*` tools
enabled, so every tool action executes in the *remote* sandbox (an `AgentCoreRuntimeSandbox` on a
SEPARATE AgentCore Runtime) instead of on the harness host. 2 microVMs per episode: the harness
Runtime (this process) + the sandbox Runtime (where `sb` runs).

The tool set deliberately mirrors the `Sandbox` protocol (sandbox.py). One MCP server per episode
wraps the *already-open* sandbox session, so the agent's tool calls and the hidden reward checker
share a single sandbox session (the one the trainer opened + grades).

`build_sandbox_mcp(sb, workdir)` returns a FastMCP whose `streamable_http_app()` is served on
localhost; the CLI is pointed at it with
`--mcp-config '{"mcpServers":{"sandbox":{"type":"http","url":"http://127.0.0.1:<port>/mcp"}}}'`.
"""

from __future__ import annotations


def build_sandbox_mcp(sb, workdir: str = "/workspace"):
    """Wrap a live Sandbox as an MCP server exposing bash/write_file/read_file."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("agentcore-sandbox")

    def _abs(p: str) -> str:
        return p if p.startswith("/") else f"{workdir}/{p}"

    @mcp.tool()
    async def bash(command: str, timeout: int = 120) -> str:
        """Run a shell command in the remote sandbox; returns exit code + stdout + stderr."""
        code, out, err = await sb.exec(command, timeout=timeout)
        return f"exit={code}\n--- stdout ---\n{out}\n--- stderr ---\n{err}"

    @mcp.tool()
    async def write_file(path: str, content: str) -> str:
        """Write a file in the sandbox workspace (relative paths resolve against the workdir)."""
        await sb.write_file(_abs(path), content)
        return f"wrote {path}"

    @mcp.tool()
    async def read_file(path: str) -> str:
        """Read a file from the sandbox workspace."""
        return await sb.read_file(_abs(path))

    return mcp


# The three tool names Claude Code must be restricted to (built-ins disabled).
MCP_TOOL_NAMES = (
    "mcp__sandbox__bash",
    "mcp__sandbox__write_file",
    "mcp__sandbox__read_file",
)
