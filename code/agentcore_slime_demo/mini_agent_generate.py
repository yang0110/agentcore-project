"""slime custom-generate entry point for the MINI harness (Demos A3 / B3).

Same job as claude_code_generate.generate — run one agentic episode and turn its multi-turn
conversation into loss-masked training samples — but the agent is our own ~1-file
:class:`MiniClaudeHarness` (mimic-Claude-Code) instead of the real Claude Code CLI.

Wire into slime:
    --custom-generate-function-path agentcore_slime_demo.mini_agent_generate.generate

The wiring is identical to the Claude Code path (Agent Lightning decoupling): stand up slime's
`AnthropicAdapter` in front of SGLang, open a session, run the agent (whose model calls go
through the adapter so the policy serves them and the adapter records the trajectory), score the
result by running the task's checker in the sandbox, then `finish_session` -> list[Sample].

The ONE difference that makes A3/B3 simpler than A2/B2 today: the agent runs in THIS process, not
in the sandbox, so there is NO CLI/Node to install. The sandbox is the vanilla AgentCore Code
Interpreter (A3) or GPU Runtime (B3) via `make_sandbox` — no custom multi-user container, and
`SANDBOX=local` even works for an offline smoke test against a local adapter.

NOTE (target architecture — see docs/harness-sandbox-decoupling.md §7): running the harness
IN-PROCESS is a BRING-UP SHORTCUT, not the intended deployment. The decoupled + disaggregated
target for A2/A3/B2/B3 alike is TWO separate AgentCore Runtimes: the harness deployed as its own
AgentCore Runtime (a BedrockAgentCoreApp wrapping MiniClaudeHarness) whose tool calls invoke a
SEPARATE sandbox Runtime (Code Interpreter for A3, a real GPU Runtime instance for B3). The
in-process path here is what to replace, not the end state.

REQUIREMENTS (does not run fully offline — it needs a served policy):
  * slime installed; a running SGLang router; a policy checkpoint.
  * ADAPTER_PUBLIC_HOST — an address the sandbox... (see NOTE) — actually only needed if the
    sandbox must dial back. The mini agent runs in-process and reaches the adapter over
    localhost, so ADAPTER_PUBLIC_HOST defaults to 127.0.0.1 here (unlike the CLI path, where the
    CLI runs INSIDE the sandbox and must dial back to the trainer host).
"""

from __future__ import annotations

import logging
import os
import secrets
import time

logger = logging.getLogger(__name__)

_WORKDIR = os.environ.get("AGENTCORE_WORKDIR", "")  # "" => sandbox native writable CWD (Code Interpreter)

_AGENT_PROMPT = (
    "Read PROBLEM.md in this directory (use the read_file tool). Write a file named solution.py "
    "that satisfies it. Then run `python3 run_checks.py` and iterate until it exits 0 (all checks "
    "pass)."
)


def _maybe_gpu_gate(sb):
    """Wrap the sandbox so GPU-touching exec calls route through a process-wide GpuPool.

    Enabled by AGENTCORE_GPU_GATE (off | heuristic | all), default 'off'. Demo B3 sets 'all' so
    the KernelBench checker (`python3 run_checks.py` imports torch internally) is admission-
    controlled onto the env GPUs; Demo A3's CPU checks leave it 'off'. Mirrors
    claude_code_generate._maybe_gpu_gate."""
    mode = os.environ.get("AGENTCORE_GPU_GATE", "off").lower()
    if mode in ("off", "", "0", "false"):
        return sb
    from .gpu_pool import GpuGatedSandbox, GpuPool

    pool = GpuPool.from_env()
    logger.info("[mini_agent_generate] GPU gating=%s | %s", mode, pool.describe())
    return GpuGatedSandbox(sb, pool, gate_all=(mode == "all"), always_gate=("run_checks.py",))


async def _prepare_workspace(sb, task) -> None:
    """Drop the problem statement + checker into the sandbox (relative paths land in the CWD)."""
    if _WORKDIR:
        await sb.exec(f"mkdir -p {_WORKDIR}", check=True, timeout=60)
    await sb.write_file("PROBLEM.md", task.initial_user_prompt())
    await sb.write_file("run_checks.py", task.runner_source())


async def _score(sb, task) -> float:
    exit_code, _out, _err = await sb.exec("python3 run_checks.py", timeout=120)
    return task.reward(exit_code)


async def generate(args, base_sample, sampling_params, evaluation: bool = False):
    """Per-sample rollout using the mini Claude-Code-like harness. Returns list[Sample]."""
    # Lazy slime imports so this module loads without slime present (offline tooling/tests).
    from slime.agent.adapters import AnthropicAdapter
    from slime.agent.aiohttp_threaded import run_app_in_thread
    from slime.utils.processing_utils import load_tokenizer

    from .generate import _task_from_sample
    from .mini_agent import MiniClaudeHarness
    from .sandbox import make_sandbox

    task = _task_from_sample(base_sample)

    tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
    adapter = AnthropicAdapter(
        tokenizer=tokenizer,
        sglang_url=f"http://{args.sglang_router_ip}:{args.sglang_router_port}",
        tool_parser=getattr(args, "sglang_tool_call_parser", None) or None,
        reasoning_parser=getattr(args, "sglang_reasoning_parser", None) or None,
    )
    handle = run_app_in_thread(
        adapter.app,
        host=os.environ.get("ADAPTER_BIND_HOST", "0.0.0.0"),
        # 0 => ephemeral port. generate() runs once PER SAMPLE concurrently (RB*NS episodes at
        # once); a fixed port makes every episode after the first fail with [Errno 98] address in
        # use. run_app_in_thread reports the real port via handle.port, used for adapter_url below.
        port=int(os.environ.get("ADAPTER_PORT", "0")),
        thread_name="anthropic-adapter-mini",
    )
    # The mini agent runs in-process, so it reaches the adapter over localhost; ADAPTER_PUBLIC_HOST
    # is honored only for parity with the CLI path (e.g. adapter bound on a routable NIC).
    adapter_host = os.environ.get("ADAPTER_PUBLIC_HOST") or "127.0.0.1"
    adapter_url = f"http://{adapter_host}:{handle.port}"

    session_id = base_sample.session_id or f"mini-{task.task_id}-{secrets.token_hex(6)}"
    base_sample.session_id = session_id
    adapter.open_session(session_id, sampling_defaults=sampling_params)

    harness = MiniClaudeHarness(max_turns=int(os.environ.get("MINI_AGENT_MAX_TURNS", "12")))
    max_tokens = int(getattr(args, "rollout_max_response_len", 0) or 1024)

    t0 = time.time()
    try:
        sandbox = make_sandbox(os.environ.get("SANDBOX", "agentcore"),
                               region=os.environ.get("AGENTCORE_REGION"))
        async with sandbox as raw_sb:
            sb = _maybe_gpu_gate(raw_sb)
            await _prepare_workspace(sb, task)
            run_result = await harness.run(
                sb, session_id=session_id, adapter_url=adapter_url,
                prompt=_AGENT_PROMPT, max_tokens=max_tokens,
            )
            reward = await _score(sb, task)

        logger.info("[mini_agent_generate] %s reward=%.1f turns=%d stop=%s elapsed=%.1fs",
                    task.task_id, reward, run_result.turns, run_result.stop_reason, time.time() - t0)

        samples = await adapter.finish_session(
            session_id, base_sample=base_sample, reward=float(reward),
            extra_metadata={"task_id": task.task_id, "solved": reward == 1.0,
                            "num_turns": run_result.turns, "tool_calls": run_result.tool_calls,
                            "stop_reason": run_result.stop_reason},
        )
        return samples
    finally:
        await adapter.drop_session(session_id, wait_timeout=30)
