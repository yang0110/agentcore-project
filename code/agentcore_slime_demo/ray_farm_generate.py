"""`ray_farm_generate.generate` — slime hook for Demo B4 (dense GPU-shared REMOTE g6 sandbox).

B4's premise: like all of series B, the sandbox is a REMOTE g6 AgentCore Runtime #2 — but instead
of B3's one-session-per-g6 (which cannot share a GPU, so N concurrent rollouts want N g6
instances), several concurrent KernelBench sandboxes share ONE g6 instance and let **Ray split its
single GPU** between them (``sandbox_ray_farm.RayGpuFarm`` runs INSIDE that g6). The trainer box
gives all 8 of its GPUs to slime (4 train + 4 infer), exactly like B2/B3 — the shared GPU is the
remote g6's, not a local one.

How the sharing is wired (AgentCore maps 1 runtimeSessionId -> 1 instance):
  * every B4 episode invokes the g6 under ONE shared ``runtimeSessionId`` (``_SHARED_RUNTIME_SESSION``,
    one per rollout process), so they all land on the SAME g6 instance;
  * each episode carries a per-episode ``sub_session``; the g6 server leases one fractional-GPU Ray
    actor per sub_session, so up to ``B4_MAX_SANDBOXES`` episodes run on the one GPU concurrently.

What stays the same as B3:
  * the mini Claude-Code-like harness (mini_agent.MiniClaudeHarness) drives write -> run -> fix;
  * every LLM call goes through slime's ``AnthropicAdapter`` proxy, keyed by session_id, so the
    trajectory is captured proxy-side and ``finish_session`` drains it into GRPO samples;
  * the KernelBench checker (``python3 run_checks.py``) runs on a GPU in the sandbox; reward =
    task.reward(exit_code).

What differs from B3:
  * the sandbox is a ``RayFarmRuntimeSandbox`` (a sub_session slot on the shared g6 farm), NOT a
    dedicated one-session-per-instance ``AgentCoreRuntimeSandbox``;
  * the mini harness runs IN-PROCESS here (not on a separate harness Runtime #1), so it dials the
    proxy over localhost — no VPC dial-back, no ADAPTER_PUBLIC_HOST, no harness Runtime to deploy.
    B4 therefore adds exactly ONE thing over Demo B: the Ray GPU-sharing farm inside the g6 sandbox.

Wire with:
  --custom-generate-function-path agentcore_slime_demo.ray_farm_generate.generate
and point AGENTCORE_SANDBOX_RUNTIME_ARN at the B4 GPU-shared sandbox Runtime (its g6 image bakes
`ray` + the B4_* farm config). See demo_b4/run.sh.
"""

from __future__ import annotations

import os
import secrets

_AGENT_PROMPT = (
    "Read PROBLEM.md in this directory. Write a file named solution.py that satisfies it. "
    "Then run `python3 run_checks.py` and iterate until it exits 0 (all checks pass)."
)

# ONE shared runtimeSessionId per rollout PROCESS: every episode invokes the g6 under it, so they
# all reuse the SAME g6 instance (whose GPU the farm shares). AgentCore requires length in [33,128]:
# "b4farm-" (7) + token_hex(16) (32) = 39. Override with B4_SHARED_RUNTIME_SESSION to pin it.
_SHARED_RUNTIME_SESSION = os.environ.get("B4_SHARED_RUNTIME_SESSION") or f"b4farm-{secrets.token_hex(16)}"


async def generate(args, base_sample, sampling_params, evaluation: bool = False):
    # Local imports so the module loads without slime present (offline tooling/tests).
    from .agentcore_runtime_generate import _shared_adapter
    from .agentcore_runtime_sandbox import RayFarmRuntimeSandbox
    from .generate import _task_from_sample
    from .mini_agent import MiniClaudeHarness

    task = _task_from_sample(base_sample)
    workdir = os.environ.get("SANDBOX_WORKDIR", "/workspace")

    # The harness is IN-PROCESS, so it dials the adapter over localhost (no public port / SG rule).
    # One shared adapter per (host, port), multiplexed by session_id (see agentcore_runtime_generate).
    adapter, handle = _shared_adapter(
        args,
        os.environ.get("ADAPTER_BIND_HOST", "127.0.0.1"),
        int(os.environ.get("ADAPTER_PORT", "0")),
    )
    adapter_url = f"http://127.0.0.1:{handle.port}"

    # The adapter Bearer session key (the trajectory id); unique per episode.
    session_id = base_sample.session_id
    if not session_id:
        session_id = f"b4-{task.task_id}-{secrets.token_hex(16)}"
    base_sample.session_id = session_id
    adapter.open_session(session_id, sampling_defaults=sampling_params)

    max_turns = int(os.environ.get("EPISODE_MAX_TURNS", "3"))
    max_tokens = int(os.environ.get("MINI_AGENT_MAX_TOKENS", "1024"))

    try:
        # Lease a fractional-GPU slot on the SHARED g6 farm for this whole episode (shared
        # runtimeSessionId => same g6 instance; per-episode sub_session => own actor + workdir).
        sandbox = RayFarmRuntimeSandbox(
            workdir=workdir,
            shared_session_id=_SHARED_RUNTIME_SESSION,
            sub_session=f"ep-{secrets.token_hex(8)}",
        )
        async with sandbox as sb:
            # --- seed the sandbox workspace with the problem + checker -----------------------
            await sb.write_file("PROBLEM.md", task.initial_user_prompt())
            await sb.write_file("run_checks.py", task.runner_source())

            # --- run the mini harness (LLM via the local proxy, tools in the leased g6 slot) --
            harness = MiniClaudeHarness(max_turns=max_turns)
            await harness.run(
                sb, session_id=session_id, adapter_url=adapter_url,
                prompt=_AGENT_PROMPT, max_tokens=max_tokens,
            )

            # --- score by running the KernelBench checker on the shared GPU ------------------
            code, _out, _err = await sb.exec("python3 run_checks.py", timeout=120)
            reward = task.reward(int(code))

        return await adapter.finish_session(
            session_id, base_sample=base_sample, reward=float(reward),
            extra_metadata={"task_id": task.task_id, "solved": reward == 1.0},
        )
    finally:
        await adapter.drop_session(session_id, wait_timeout=30)
