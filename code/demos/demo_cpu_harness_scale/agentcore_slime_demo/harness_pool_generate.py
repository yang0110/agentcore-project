"""slime custom-generate hook for the harness-scaling demo.

Wire with:
    --custom-generate-function-path agentcore_slime_demo.harness_pool_generate.generate

Unlike agentcore_slime_demo.generate.generate (in-process run_episode driving SGLang directly),
this hook hands each episode to a WARM in-process harness worker pool (harness_pool.HarnessPool):
the pool owns W warm workers + one shared AnthropicAdapter; a free worker pulls the episode,
runs the MiniClaudeHarness loop (LLM via the adapter, tools in a fresh remote sandbox Runtime),
scores it, and finish_session drains the trajectory into loss-masked GRPO samples.

The pool is the throughput ceiling knob (HARNESS_WORKERS=W): slime admits up to
S = CONC * num_engines episodes concurrently, but only W are actively DRIVEN by a harness at once,
so raising W lifts the harness tier of throughput = min(inference, sandbox, harness). See
harness_pool.py for the per-tier routing/load-balancing model and docs/harness-scaling.md.

Uses slime's stock async rollout (train_async.py, NO fully-async override): finish_session returns
a LIST of Samples, which slime's stock path handles as the fan-out shape. (The fully-async path's
sort key chokes on nested lists — that is why this demo stays on the stock driver.)
"""

from __future__ import annotations

from .harness_pool import get_pool


async def generate(args, sample, sampling_params, evaluation: bool = False):
    # Lazy: build task from the slime Sample exactly like the recommended hook does.
    from .generate import _task_from_sample

    task = _task_from_sample(sample)
    pool = get_pool(args)
    # Returns the list of fully-formed Samples produced by the adapter's finish_session.
    return await pool.submit(task, sampling_params, sample)
