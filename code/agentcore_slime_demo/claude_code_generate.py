"""Real-harness path: Claude Code CLI as the agent, slime policy as the brain, AgentCore as hands.

This is the faithful "Claude Code as the harness between" wiring from docs/demo-plan.md §1,
and the AgentCore analogue of slime's examples/coding_agent_rl/generate.py — the ONLY change
from that shipped example is the sandbox class:

    from slime.agent.sandbox import E2BSandbox      # slime's reference backend
    from .sandbox import AgentCoreSandbox            # <-- swapped in here

Everything else is slime's real stack, unchanged (Agent Lightning's "train with real harnesses,
ZERO changes"): ClaudeCodeHarness installs + runs `claude -p`, wiring ANTHROPIC_BASE_URL to
slime's AnthropicAdapter, so every "Anthropic" completion is served by the slime policy (SGLang)
and its tokens are captured for training. Tool calls (Bash/Edit) execute inside the AgentCore
sandbox microVM. Reward comes from running the task's checker in that same sandbox.

NOTE (target architecture — see docs/harness-sandbox-decoupling.md §7): this path CO-LOCATES the
harness INSIDE the sandbox (slime's ClaudeCodeHarness runs `claude -p` in `sb`), which is a
BRING-UP SHORTCUT, not the target. The decoupled + disaggregated target for A2/A3/B2/B3 alike is
TWO separate AgentCore Runtimes: Claude Code on its OWN harness Runtime, its built-in tools
disabled and redirected via an MCP bridge (§3) to a SEPARATE sandbox Runtime (Code Interpreter for
A2, a real GPU Runtime instance for B2). Co-location here is what to replace, not the end state.

Wire into slime:
    --custom-generate-function-path agentcore_slime_demo.claude_code_generate.generate

REQUIREMENTS (this path does NOT run offline):
  * slime installed; a running SGLang router; a policy checkpoint.
  * Claude Code + Node 22 tarballs (SLIME_AGENT_NODE_TARBALL / SLIME_AGENT_CC_TARBALL).
  * ADAPTER_PUBLIC_HOST set to an address the sandbox can dial back (demo-plan.md §3).
  * A sandbox that can host Claude Code: a **custom AgentCore Runtime container** (useradd/
    setsid/npm/multi-user), i.e. demo-plan.md Option B — the restricted Code Interpreter
    cannot run a full CLI harness. `AgentCoreSandbox` here targets that container's exec API.
"""

from __future__ import annotations

import logging
import os
import secrets
import threading
import time
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

_WORKDIR = os.environ.get("AGENTCORE_WORKDIR", "/workspace")

# One SHARED adapter per (bind host, port) for the whole rollout — the same pattern the two-Runtime
# hook (agentcore_runtime_generate) uses. generate() runs per-sample CONCURRENTLY, so binding a NEW
# adapter on a fixed port each call collides ([Errno 98]); binding on an ephemeral port (0) breaks
# the sandbox dial-back, which must hit a PINNED port (ADAPTER_PORT) so a single narrow security-
# group ingress rule stays stable (the sandbox microVM reaches the proxy over the private VPC, no
# public port). The AnthropicAdapter already multiplexes concurrent episodes by session_id (the
# Bearer session id), so we construct + start it ONCE and every episode opens/finishes/drops its
# own session on it.
_SHARED_ADAPTER: dict[tuple, tuple] = {}
_SHARED_LOCK = threading.Lock()


def _shared_adapter(args, bind_host: str, port: int):
    """Return (adapter, handle) for (bind_host, port), constructing + starting it once."""
    from slime.agent.adapters import AnthropicAdapter
    from slime.agent.aiohttp_threaded import run_app_in_thread
    from slime.utils.processing_utils import load_tokenizer

    key = (bind_host, port)
    with _SHARED_LOCK:
        hit = _SHARED_ADAPTER.get(key)
        if hit is not None:
            return hit
        tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        adapter = AnthropicAdapter(
            tokenizer=tokenizer,
            sglang_url=f"http://{args.sglang_router_ip}:{args.sglang_router_port}",
            tool_parser=getattr(args, "sglang_tool_call_parser", None) or None,
            reasoning_parser=getattr(args, "sglang_reasoning_parser", None) or None,
        )
        handle = run_app_in_thread(
            adapter.app, host=bind_host, port=port, thread_name="anthropic-adapter-cli",
        )
        _SHARED_ADAPTER[key] = (adapter, handle)
        return adapter, handle

_AGENT_PROMPT = (
    "Read PROBLEM.md in this directory. Write a file named solution.py that satisfies it. "
    "Then run `python3 run_checks.py` and iterate until it exits 0 (all checks pass)."
)


@asynccontextmanager
async def _boot_sandbox(harness):
    """Open the sandbox that will host the Claude Code CLI and (unless prebaked) install it.

    Two backends, selected by SANDBOX:
      * "agentcore" / "aws" (default) — the managed Code Interpreter (AgentCoreSandbox). The
        toolchain (Node 22 + claude CLI) is streamed in at runtime via harness.install_cli.
      * "agentcore-runtime" / "runtime" — the sandbox as its OWN custom AgentCore Runtime
        (AgentCoreRuntimeSandbox), driven over invoke_agent_runtime. This is the A2/B2 target:
        the custom image (demo_a2/demo_b2 Dockerfile) BAKES node+claude in, so install_cli is
        skipped — and MUST be, because that path streams the multi-MB node/claude tarballs as
        text through write_file, which corrupts binaries over the runtime transport. The ARN
        comes from AGENTCORE_SANDBOX_RUNTIME_ARN.
    """
    kind = os.environ.get("SANDBOX", "agentcore").lower()
    if kind in ("agentcore-runtime", "runtime"):
        from .agentcore_runtime_sandbox import AgentCoreRuntimeSandbox

        sb = AgentCoreRuntimeSandbox(
            runtime_arn=os.environ.get("AGENTCORE_SANDBOX_RUNTIME_ARN"),
            region=os.environ.get("AGENTCORE_REGION"),
            workdir=_WORKDIR,
        )
        prebaked = True   # node+claude baked into the custom image; never stream them in
    else:
        from .sandbox import AgentCoreSandbox

        sb = AgentCoreSandbox(region=os.environ.get("AGENTCORE_REGION"), workdir=_WORKDIR)
        # Allow forcing the prebaked path even on Code Interpreter (rare; e.g. a prewarmed image).
        prebaked = os.environ.get("AGENTCORE_CLI_PREBAKED") == "1"

    await sb.__aenter__()
    try:
        if not prebaked:
            await harness.install_cli(sb)   # Node 22 + claude CLI from host tarballs
        yield sb
    finally:
        await sb.__aexit__(None, None, None)


async def _prepare_workspace(sb, task) -> None:
    await sb.exec(f"mkdir -p {_WORKDIR}", check=True, timeout=60)
    await sb.write_file(f"{_WORKDIR}/PROBLEM.md", task.initial_user_prompt())
    await sb.write_file(f"{_WORKDIR}/run_checks.py", task.runner_source())


async def _score(sb, task) -> float:
    exit_code, _out, _err = await sb.exec("python3 run_checks.py", timeout=120)
    return task.reward(exit_code)


async def generate(args, base_sample, sampling_params, evaluation: bool = False):
    """Per-sample rollout using the real Claude Code harness. Returns list[Sample]."""
    # Lazy imports so this module loads without slime present; the offline demo never
    # touches this path.
    from slime.agent.harness import ClaudeCodeHarness

    from .generate import _task_from_sample

    task = _task_from_sample(base_sample)
    harness = ClaudeCodeHarness()

    adapter_public_host = os.environ.get("ADAPTER_PUBLIC_HOST")
    if not adapter_public_host:
        raise RuntimeError("ADAPTER_PUBLIC_HOST not set (see demo-plan.md §3).")

    # ONE shared adapter on a PINNED port for the whole rollout (see _shared_adapter). The sandbox
    # microVM dials this port back over the SG-gated private VPC, so it must be stable across every
    # concurrent episode — an ephemeral port would break the single narrow ingress rule.
    adapter, handle = _shared_adapter(
        args,
        os.environ.get("ADAPTER_BIND_HOST", "0.0.0.0"),
        int(os.environ.get("ADAPTER_PORT", "18021")),
    )
    adapter_url = f"http://{adapter_public_host}:{handle.port}"

    session_id = base_sample.session_id or f"acdemo-{task.task_id}-{secrets.token_hex(6)}"
    base_sample.session_id = session_id
    adapter.open_session(session_id, sampling_defaults=sampling_params)

    t0 = time.time()
    try:
        async with _boot_sandbox(harness) as sb:
            await _prepare_workspace(sb, task)
            agent_exit_code = await harness.run(
                sb,
                workdir=_WORKDIR,
                session_id=session_id,
                adapter_url=adapter_url,
                time_budget_sec=int(os.environ.get("AGENT_TIME_BUDGET_SEC", "900")),
                prompt=_AGENT_PROMPT,
            )
            reward = await _score(sb, task)

        logger.info("[claude_code_generate] %s reward=%.1f exit=%d elapsed=%.1fs",
                    task.task_id, reward, agent_exit_code, time.time() - t0)

        samples = await adapter.finish_session(
            session_id, base_sample=base_sample, reward=float(reward),
            extra_metadata={"task_id": task.task_id, "solved": reward == 1.0,
                            "agent_exit_code": agent_exit_code},
        )
        return samples
    finally:
        await adapter.drop_session(session_id, wait_timeout=30)
