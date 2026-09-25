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

Harness flavor, selected by HARNESS_KIND:
  * "mini"  — reuse MiniClaudeHarness (A3/B3 loop) but with tools bound to the remote sandbox.
             (The old "claude" flavor — the real `claude -p` CLI for A2/B2 — has been removed.)

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


async def handle_async(request: dict) -> dict:
    kind = (request or {}).get("harness_kind") or os.environ.get("HARNESS_KIND", "mini")
    if kind == "mini":
        return await _run_mini(request)
    return {"error": f"unknown harness_kind {kind!r}"}


if BedrockAgentCoreApp is not None:
    app = BedrockAgentCoreApp()

    @app.entrypoint
    async def entrypoint(request):  # noqa: ANN001
        return await handle_async(request)

    if __name__ == "__main__":
        app.run()
