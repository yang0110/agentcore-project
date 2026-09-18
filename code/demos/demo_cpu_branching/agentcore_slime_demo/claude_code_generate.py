"""demo_cpu_branching — Claude Code CLI as the harness, one episode → MANY training samples.

This is the fan-out path. The in-process `run_episode` harness of the other demos produces exactly
ONE flattened Sample per episode. Here we swap it for the **real Claude Code CLI** wired to slime's
`AnthropicAdapter`, and we deliberately provoke the harness behaviors that split one rollout into
several trainable branches — so `generate()` returns a **list[Sample]** (the fan-out), not a bare
Sample.

Wire into slime:
    --custom-generate-function-path agentcore_slime_demo.claude_code_generate.generate

TOPOLOGY (co-located harness + REMOTE sandbox):
  * The Claude Code CLI runs INSIDE the sandbox microVM, its "Anthropic" endpoint pointed at slime's
    AnthropicAdapter (ANTHROPIC_BASE_URL = http://$ADAPTER_PUBLIC_HOST:$ADAPTER_PORT). Every model
    call is therefore served by the slime policy (SGLang) and every emitted token is captured for
    training. The adapter runs in THIS training process (co-located with slime).
  * Tools (Bash/Edit/Read) execute in the sandbox — a REMOTE CPU AgentCore Runtime microVM
    (SANDBOX=agentcore-runtime, AgentCoreRuntimeSandbox), reached over invoke_agent_runtime.

WHERE THE FAN-OUT COMES FROM (what `TrajectoryManager` records as separate branches):
  The adapter feeds every /messages turn into a shared `TrajectoryManager` keyed by session id. The
  per-sid tree forks — and each leaf becomes one Sample in `finish_session` — whenever the CLI does
  either of the two things this demo intentionally encourages (see `_AGENT_PROMPT`):
    1. **Sub-agents.** When Claude Code spawns a sub-agent (a Task tool call), the sub-agent's turns
       arrive under the SAME session id but on a message prefix that diverges from the parent's, so
       `record_turn` mounts them on a NEW branch of the tree. Parent chain + sub-agent chain = two
       leaves = two Samples.
    2. **Context summarization / compaction.** When the CLI compacts context (replacing a long
       message history with a summary), the next turn's prompt prefix no longer matches the recorded
       parent chain, so it mounts as a fork rather than extending the old chain. We set the
       adapter's `fork_threshold_tokens` LOW (BRANCHING_FORK_THRESHOLD, default 512) so ordinary
       compactions are recorded as forks instead of being merged back — making the fan-out reliable
       enough to train on.
  `AnthropicAdapter.finish_session(sid, base_sample=..., reward=...)` linearizes the tree: one Sample
  per leaf, EACH stamped with the SAME `rollout_id = base_sample.rollout_id or base_sample.index`
  (slime/agent/trajectory.py:get_trajectory) and EACH carrying the episode's full outcome reward
  (reward is assigned in full to every sibling, not split). Those siblings are what the
  rollout-level advantage (agentcore_slime_demo.rollout_advantage) and rollout-level token-mean loss
  (calculate-per-token-loss OFF + shared rollout_id => rollout_mask_sums denominator) reconcile.

MODE REQUIREMENT: returning list[Sample] puts slime on the FAN-OUT shape (generate_and_rm_group
returns list[list[Sample]]). The stock / 1-step-async driver tolerates that nesting everywhere via
`isinstance(group[0], list)` guards; the FULLY-ASYNC driver does NOT (its sort key crashes on the
nested shape). So this demo MUST run MODE=A. run.sh enforces it.

REQUIREMENTS (this path does NOT run offline):
  * slime installed; a running SGLang router; a policy checkpoint.
  * A sandbox image with node 22 + the claude CLI BAKED IN (deploy/Dockerfile.sandbox.cli) — the
    prebaked path. Streaming the multi-MB node/claude tarballs through write_file corrupts the
    binaries over the runtime transport, so install_cli is skipped for agentcore-runtime.
  * ADAPTER_PUBLIC_HOST set to an address the sandbox can dial back (a private VPC IP of this box).
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

# One SHARED adapter per (bind host, port) for the whole rollout. generate() runs per-sample
# CONCURRENTLY, so binding a NEW adapter on a fixed port each call collides ([Errno 98]); binding on
# an ephemeral port (0) breaks the sandbox dial-back, which must hit a PINNED port (ADAPTER_PORT) so
# a single narrow security-group ingress rule stays stable. The AnthropicAdapter multiplexes
# concurrent episodes by session_id, so we construct + start it ONCE and every episode
# opens/finishes/drops its own session on it.
_SHARED_ADAPTER: dict[tuple, tuple] = {}
_SHARED_LOCK = threading.Lock()

# Low fork threshold => context compactions are recorded as tree FORKS (siblings) rather than merged
# back into the parent chain. This is what makes summarization-driven fan-out reliable enough to
# train on. See the module docstring, item (2).
_FORK_THRESHOLD = int(os.environ.get("BRANCHING_FORK_THRESHOLD", "512"))


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
            # LOW threshold => record compactions as forks (fan-out), not merges. See module docstring.
            fork_threshold_tokens=_FORK_THRESHOLD,
        )
        handle = run_app_in_thread(
            adapter.app, host=bind_host, port=port, thread_name="anthropic-adapter-branching",
        )
        _SHARED_ADAPTER[key] = (adapter, handle)
        return adapter, handle


# The prompt deliberately steers the CLI toward the two fan-out behaviors (sub-agent + compaction).
# It is legitimate agent guidance — "verify with a sub-agent", "summarize before continuing" — not a
# hack: these are exactly the real harness behaviors §"When one episode becomes many samples"
# describes, and the demo's point is to make them happen every episode so the multi-sample plumbing
# is exercised on every rollout.
_AGENT_PROMPT = (
    "Read PROBLEM.md in this directory: it contains a buggy function and its failing tests. "
    "Write a file `solution.py` in this directory holding the FULL corrected function, then run "
    "`python3 run_checks.py` (it imports your solution.py and runs the asserts) until it exits 0.\n"
    "Work in this order, and follow it even for simple bugs:\n"
    "  1. Investigate the failure. Spawn a sub-agent (the Task tool) to write a first solution.py "
    "     and run the checker, and report back the exact failing assertion — so your main context "
    "     stays focused on the fix.\n"
    "  2. Once you have the sub-agent's report, SUMMARIZE what you learned in one short paragraph "
    "     before you edit, then write the corrected solution.py yourself.\n"
    "  3. Re-run `python3 run_checks.py` and iterate until it exits 0."
)


@asynccontextmanager
async def _boot_sandbox():
    """Open the sandbox that hosts the Claude Code CLI.

    SANDBOX=agentcore-runtime (this demo's default) => AgentCoreRuntimeSandbox over a REMOTE CPU
    AgentCore Runtime whose image BAKES node+claude (prebaked; install_cli is skipped and MUST be —
    streaming the tarballs as text through write_file corrupts the binaries). The ARN comes from
    AGENTCORE_SANDBOX_RUNTIME_ARN. SANDBOX=agentcore falls back to the managed Code Interpreter with
    a runtime CLI install (dev only; the microVM cannot host a full multi-user CLI harness well)."""
    kind = os.environ.get("SANDBOX", "agentcore-runtime").lower()
    if kind in ("agentcore-runtime", "runtime"):
        from .agentcore_runtime_sandbox import AgentCoreRuntimeSandbox

        sb = AgentCoreRuntimeSandbox(
            runtime_arn=os.environ.get("AGENTCORE_SANDBOX_RUNTIME_ARN"),
            region=os.environ.get("AGENTCORE_REGION"),
            workdir=_WORKDIR,
        )
        prebaked = True
    else:
        from .sandbox import AgentCoreSandbox

        sb = AgentCoreSandbox(region=os.environ.get("AGENTCORE_REGION"), workdir=_WORKDIR)
        prebaked = os.environ.get("AGENTCORE_CLI_PREBAKED") == "1"

    await sb.__aenter__()
    try:
        yield sb, prebaked
    finally:
        # __aexit__ stops the session (StopRuntimeSession for agentcore-runtime, client.stop for the
        # Code Interpreter) — see agentcore_runtime_sandbox / sandbox. One session per episode.
        await sb.__aexit__(None, None, None)


async def _prepare_workspace(sb, task) -> None:
    await sb.exec(f"mkdir -p {_WORKDIR}", check=True, timeout=60)
    await sb.write_file(f"{_WORKDIR}/PROBLEM.md", task.initial_user_prompt())
    await sb.write_file(f"{_WORKDIR}/run_checks.py", task.runner_source())
    # Any family-specific seed files (the buggy module + its test for SWE) land next to the checker.
    for name, content in getattr(task, "setup_files", lambda: {})().items():
        await sb.write_file(f"{_WORKDIR}/{name}", content)


async def _score(sb, task) -> float:
    exit_code, _out, _err = await sb.exec("python3 run_checks.py", timeout=120)
    return task.reward(exit_code)


async def generate(args, base_sample, sampling_params, evaluation: bool = False):
    """Per-sample rollout using the real Claude Code harness. Returns list[Sample] (the fan-out).

    Each returned Sample is one leaf of the episode's trajectory tree (parent chain, sub-agent
    chains, post-compaction chains), all sharing base_sample's rollout_id and the episode's outcome
    reward. See the module docstring for how the branches arise and how they are reconciled."""
    from slime.agent.harness import ClaudeCodeHarness

    from .generate import _task_from_sample

    task = _task_from_sample(base_sample)
    harness = ClaudeCodeHarness()

    adapter_public_host = os.environ.get("ADAPTER_PUBLIC_HOST")
    if not adapter_public_host:
        raise RuntimeError(
            "ADAPTER_PUBLIC_HOST not set — the sandbox microVM must be able to dial the "
            "AnthropicAdapter back on a private VPC address of this training box."
        )

    adapter, handle = _shared_adapter(
        args,
        os.environ.get("ADAPTER_BIND_HOST", "0.0.0.0"),
        int(os.environ.get("ADAPTER_PORT", "18021")),
    )
    adapter_url = f"http://{adapter_public_host}:{handle.port}"

    session_id = getattr(base_sample, "session_id", None) or f"branch-{task.task_id}-{secrets.token_hex(6)}"
    base_sample.session_id = session_id
    adapter.open_session(session_id, sampling_defaults=sampling_params)

    t0 = time.time()
    try:
        async with _boot_sandbox() as (sb, prebaked):
            await _prepare_workspace(sb, task)
            if not prebaked:
                await harness.install_cli(sb)  # dev-only Code-Interpreter path
            agent_exit_code = await harness.run(
                sb,
                workdir=_WORKDIR,
                session_id=session_id,
                adapter_url=adapter_url,
                time_budget_sec=int(os.environ.get("AGENT_TIME_BUDGET_SEC", "900")),
                prompt=_AGENT_PROMPT,
            )
            reward = await _score(sb, task)

        samples = await adapter.finish_session(
            session_id,
            base_sample=base_sample,
            reward=float(reward),
            extra_metadata={
                "task_id": task.task_id,
                "solved": reward == 1.0,
                "agent_exit_code": agent_exit_code,
            },
        )
        logger.info(
            "[branching] %s reward=%.1f exit=%d fan_out=%d elapsed=%.1fs",
            task.task_id, reward, agent_exit_code, len(samples), time.time() - t0,
        )
        return samples
    finally:
        await adapter.drop_session(session_id, wait_timeout=30)
