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
import time
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

_WORKDIR = os.environ.get("AGENTCORE_WORKDIR", "/workspace")

_AGENT_PROMPT = (
    "Read PROBLEM.md in this directory. Write a file named solution.py that satisfies it. "
    "Then run `python3 run_checks.py` and iterate until it exits 0 (all checks pass)."
)


def _maybe_gpu_gate(sb):
    """Optionally wrap the sandbox so GPU-touching exec calls route through a GpuPool.

    Enabled by AGENTCORE_GPU_GATE (off | heuristic | all), default 'off'. On a dense GPU
    Runtime instance holding many concurrent rollouts, set 'heuristic' so Claude Code's
    Bash tool calls that touch CUDA get admission-controlled + a device/memory lease, or
    'all' to gate every command. The pool is sized from GPU_POOL_* env (gpu_pool.from_env).
    Off by default so the CPU microVM path (Code Interpreter) is unaffected. See
    docs/gpu-density.md. Install/toolchain commands run BEFORE this wrap, so they are never
    gated (they are one-time and not GPU work)."""
    mode = os.environ.get("AGENTCORE_GPU_GATE", "off").lower()
    if mode == "off":
        return sb
    from .gpu_pool import GpuGatedSandbox, GpuPool

    pool = GpuPool.from_env()
    logger.info("[claude_code_generate] GPU gating=%s | %s", mode, pool.describe())
    # Also force-gate the reward checker (imports torch internally, invisible to heuristic).
    return GpuGatedSandbox(sb, pool, gate_all=(mode == "all"), always_gate=("run_checks.py",))


@asynccontextmanager
async def _boot_sandbox(harness):
    """Open an AgentCore sandbox and install the Claude Code toolchain into it."""
    from .sandbox import AgentCoreSandbox

    sb = AgentCoreSandbox(region=os.environ.get("AGENTCORE_REGION"), workdir=_WORKDIR)
    await sb.__aenter__()
    try:
        await harness.install_cli(sb)   # Node 22 + claude CLI from host tarballs (never gated)
        yield _maybe_gpu_gate(sb)       # gate GPU exec calls on a dense GPU instance
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
    from slime.agent.adapters import AnthropicAdapter
    from slime.agent.aiohttp_threaded import run_app_in_thread
    from slime.agent.harness import ClaudeCodeHarness
    from slime.utils.processing_utils import load_tokenizer

    from .generate import _task_from_sample

    task = _task_from_sample(base_sample)
    harness = ClaudeCodeHarness()

    adapter_public_host = os.environ.get("ADAPTER_PUBLIC_HOST")
    if not adapter_public_host:
        raise RuntimeError("ADAPTER_PUBLIC_HOST not set (see demo-plan.md §3).")

    tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
    adapter = AnthropicAdapter(
        tokenizer=tokenizer,
        sglang_url=f"http://{args.sglang_router_ip}:{args.sglang_router_port}",
        tool_parser=getattr(args, "sglang_tool_call_parser", None) or None,
        reasoning_parser=getattr(args, "sglang_reasoning_parser", None) or None,
    )
    handle = run_app_in_thread(
        adapter.app, host=os.environ.get("ADAPTER_BIND_HOST", "0.0.0.0"),
        # 0 => ephemeral: generate() runs per-sample concurrently, so a fixed port collides
        # ([Errno 98]). adapter_url below uses the real handle.port.
        port=int(os.environ.get("ADAPTER_PORT", "0")), thread_name="anthropic-adapter",
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
