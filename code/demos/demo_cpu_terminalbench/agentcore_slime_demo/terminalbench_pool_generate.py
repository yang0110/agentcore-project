"""slime custom-generate hook for demo_cpu_terminalbench (warm-shared image pool + heartbeat).

Wire with:
    --custom-generate-function-path agentcore_slime_demo.terminalbench_pool_generate.generate

Each episode is handed to the process-wide TerminalBenchPool: a free warm harness worker pulls it,
routes it to ITS Terminal-Bench task's Runtime, reuses that task's ONE warm shared session (kept
alive by the SessionKeeper heartbeat) with a freshly-reset per-episode workdir, runs the
MiniClaudeHarness agentic loop (LLM via the adapter under training, tools in the sandbox), grades it
with the task's baked pytest oracle, and finish_session drains the trajectory into loss-masked GRPO
samples. See terminalbench_pool.py for the warm-shared + heartbeat design.
"""

from __future__ import annotations

from .terminalbench_pool import get_tb_pool


async def generate(args, sample, sampling_params, evaluation: bool = False):
    from .generate import _task_from_sample

    task = _task_from_sample(sample)
    pool = get_tb_pool(args)
    return await pool.submit(task, sampling_params, sample)
