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

    async with sandbox as sb:
        harness = MiniClaudeHarness()
        result = await harness.run(sb, session_id=session_id, adapter_url=adapter_url, prompt=prompt)
    return {"ok": True, "result": result}


async def handle_async(request: dict) -> dict:
    kind = (request or {}).get("harness_kind") or os.environ.get("HARNESS_KIND", "mini")
    if kind == "mini":
        return await _run_mini(request)
    # "claude" flavor: run `claude -p` with sandbox_mcp.py as the only toolset. The CLI must be
    # baked into this harness Runtime's image; the MCP server bridges to the sandbox Runtime.
    return {"error": f"harness_kind {kind!r} not yet implemented in scaffolding"}


if BedrockAgentCoreApp is not None:
    app = BedrockAgentCoreApp()

    @app.entrypoint
    async def entrypoint(request):  # noqa: ANN001
        return await handle_async(request)

    if __name__ == "__main__":
        app.run()
