"""slime custom-generate entry point.

Wire into slime via:
    --custom-generate-function-path agentcore_slime_demo.generate.generate
(dispatch: slime/rollout/sglang_rollout.py:283-294).

Responsibility (Agent Lightning's "turn interactions into training data"): run one episode
with the real harness in the AgentCore sandbox, then flatten the multi-turn Trajectory into
a single token stream with a loss mask — assistant/action tokens are trainable (mask=1),
prompt + observation/tool tokens are masked (mask=0) — and attach the episode reward.

`build_token_stream` is a pure function (no slime, no network) so it can be unit-tested.
The `generate` coroutine is the slime-facing adapter; slime imports are lazy so this module
loads without slime installed.
"""

from __future__ import annotations

import os

from .app_snapshot import maybe_restore, maybe_snapshot
from .harness import Trajectory
from . import harness_pool
from .sandbox import make_sandbox
from .session_keeper import maybe_keep_warm
from .task import CodingTask


def build_token_stream(
    traj: Trajectory, tokenizer
) -> tuple[list[int], int, list[int], list[float]]:
    """Flatten a trajectory into (tokens, response_length, loss_mask, rollout_log_probs).

    slime's Sample contract (verified against slime/utils/types.py + megatron_utils/data.py):
      * `tokens`          = the FULL sequence (initial prompt + everything after).
      * `response_length` = number of tokens AFTER the initial prompt; slime derives
                            `prompt_length = len(tokens) - response_length` and treats
                            `tokens[prompt_length:]` as the response region.
      * `loss_mask`       = a mask over ONLY the response region, so
                            `len(loss_mask) == response_length` (asserted in rollout.py
                            and types.py). It is left-padded by prompt_length-1 in data.py.
      * `rollout_log_probs` = per-response-token logprob from the rollout engine, also
                            length == response_length (asserted in types.py:422). Masked
                            (observation) tokens carry 0.0, exactly as slime's native
                            multi-turn path pads them (types.py:295-302). slime needs this
                            list present whenever it reuses rollout logprobs in the loss
                            (can_reuse_log_probs_in_loss), and uses it for the
                            train/rollout_logprob_abs_diff weight-sync metric.

    So the initial prompt is the non-trainable prefix and is EXCLUDED from loss_mask.
    Within the response region: the assistant's own generated tokens get mask 1; injected
    context between turns (tool output / observations / follow-up user prompts) gets mask 0.
    Uses the recorded completion.token_ids so there's no retokenization drift on the
    trainable span (per the vLLM/AL "return token ids" note).
    """
    tokens: list[int] = []
    loss_mask: list[int] = []       # response region only (len must == response_length)
    rollout_log_probs: list[float] = []  # response region only (same length)
    prompt_len = 0                  # length of the initial, non-trainable prompt prefix
    prev_len = 0

    for i, turn in enumerate(traj.turns):
        prompt_ids = tokenizer.apply_chat_template(
            turn.prompt_messages, add_generation_prompt=True, tokenize=True
        )
        # transformers 5.x may return a BatchEncoding; coerce to a flat list[int] so the
        # slicing / len() below (and the emitted token stream) stay plain Python ints.
        if hasattr(prompt_ids, "input_ids"):
            prompt_ids = prompt_ids["input_ids"]
        if prompt_ids and isinstance(prompt_ids[0], list):
            prompt_ids = prompt_ids[0]
        prompt_ids = [int(t) for t in prompt_ids]

        if i == 0:
            # The very first prompt is the non-trainable prefix: goes into `tokens` but
            # NOT into loss_mask (loss_mask covers only the response region after it).
            prompt_len = len(prompt_ids)
            tokens.extend(prompt_ids)
        else:
            # Context added since the previous turn (observations/tool output): trained-on
            # region but masked 0, and padded with 0.0 rollout logprob (policy didn't emit it).
            new_context = prompt_ids[prev_len:]
            tokens.extend(new_context)
            loss_mask.extend([0] * len(new_context))
            rollout_log_probs.extend([0.0] * len(new_context))

        # The action: the policy's own tokens (trained on).
        resp = turn.completion.token_ids
        tokens.extend(resp)
        loss_mask.extend([1] * len(resp))
        # Per-token rollout logprobs from the policy; fall back to 0.0 if unavailable
        # (e.g. MockPolicy) or length-mismatched, so the response-region invariant holds.
        lps = turn.completion.logprobs
        if len(lps) == len(resp):
            rollout_log_probs.extend(float(x) for x in lps)
        else:
            rollout_log_probs.extend([0.0] * len(resp))

        # Next turn's prompt includes this assistant message; advance the cursor past it.
        prev_len = len(prompt_ids) + len(resp)

    response_length = len(tokens) - prompt_len
    assert len(loss_mask) == response_length, (
        f"internal: loss_mask {len(loss_mask)} != response_length {response_length}"
    )
    assert len(rollout_log_probs) == response_length, (
        f"internal: rollout_log_probs {len(rollout_log_probs)} != response_length {response_length}"
    )
    return tokens, response_length, loss_mask, rollout_log_probs


async def generate(args, sample, sampling_params, evaluation: bool = False):
    """slime per-sample rollout. Returns a single Sample.

    mini_claude_episode.run flattens the episode (the adapter's per-session record) into ONE
    trainable Sample, so we return a bare Sample — not a 1-element list. This matters:
    slime's dispatch (generate_and_rm) treats a LIST return as the "fan-out" case and hands
    back list[list[Sample]] from generate_and_rm_group. The stock rollout path tolerates that
    nesting everywhere via `isinstance(group[0], list)` guards, but the fully-async path
    (slime.rollout.fully_async_rollout, every demo's rollout driver) does NOT guard its
    sort key and crashes on the nested shape. Returning a bare Sample keeps us on slime's
    primary (non-fan-out) shape, which BOTH the stock and fully-async paths handle. Behavior
    is identical for the stock demos (one flattened sample per episode either way)."""
    from . import mini_claude_episode, router_trace

    if os.environ.get("HARNESS", "mini_claude") != "mini_claude":
        raise RuntimeError(f"HARNESS={os.environ['HARNESS']!r} is not supported: the only harness is "
                           "mini_claude, run inside the sandbox session (HARNESS=direct was removed)")
    task = _task_from_sample(sample)
    # THE HARNESS TIER (harness_pool.py, every demo's default): a free dispatch worker (W =
    # HARNESS_WORKERS, default RB x NS) pulls this episode, takes its sandbox FIRST, then the SGLang
    # engine with the fewest in-flight episodes (sticky for all its turns). The agent loop itself runs
    # INSIDE the sandbox session, so the harness scales with the sandbox tier, not the trainer. Here
    # the sandbox tier is the GPU farm, which bounds concurrency (CONC), so W rarely binds.
    kind = os.environ.get("SANDBOX", "agentcore")
    pool = harness_pool.get_pool(args)
    # App-level snapshot/resume (SNAPSHOT_EVERY=1): every episode opens a FRESH sandbox session, so
    # each boundary is effectively a GPU-instance recycle. Restore this task's workspace from the
    # store we own at ENTER (resume), snapshot it back at EXIT (quiescent, before the weight-sync
    # barrier). SNAPSHOT_EVERY=0 (default) disables both. See app_snapshot.py.
    snap_key = f"ep-{task.task_id}"

    async def episode(worker):
        # HEARTBEAT=1 (default on our own remote Runtimes, every demo): wrap the sandbox in a KeptSandbox so a process-wide
        # SessionKeeper heartbeats it if it goes idle across the weight-sync barrier. Off => unchanged.
        # PREOPEN=N (opt-in) keeps N fresh sessions booted ahead; ignored on the warm-shared ray farm.
        async with pool.sandbox(kind, lambda: maybe_keep_warm(
                make_sandbox(kind, region=os.environ.get("AGENTCORE_REGION")))) as (sb, wait_ms, slot):
            with pool.engine() as (e, engine):
                # ONE [router] record per episode: the join that ties this episode's sandbox session (also
                # stamped on every [sbx-op]) to the worker that drove it, the engine its turns ran on and
                # the Runtime it used. metrics.py parses these into <log>.router.csv and check_stickiness
                # asserts each episode is sticky to one engine / sandbox session / runtime ARN.
                router_trace.emit(
                    task_id=task.task_id,
                    rollout_id=getattr(sample, "index", "?"),
                    harness_worker=worker.wid,
                    engine=engine,
                    engine_pinned=int(len(pool._engine_urls) > 1),
                    sandbox_id=router_trace.sandbox_id_of(sb),
                    sandbox_session=router_trace.session_of(sb),
                    runtime_arn=router_trace.arn_of(sb),
                    sandbox_image=getattr(task, "image", "cpu"),
                    family=getattr(task, "family", "?"),
                )
                await maybe_restore(sb, snap_key)
                # The agent loop runs INSIDE this sandbox session (mini_claude_episode.py → in_session_harness):
                # tool-calling agent via slime's AnthropicAdapter, hidden grader after the episode. The trainer
                # only hosts the adapter (token record); there is no in-trainer harness.
                out = await mini_claude_episode.run(args, sample, task, sampling_params, sb, engine,
                                                    fields=pool.tier_fields(worker, e, wait_ms, slot))
                await maybe_snapshot(sb, snap_key)
                return out

    return await pool.run(episode)


def _task_from_sample(sample):
    """Extract the episode's Task from a slime Sample's prompt payload.

    Dispatches on the `family` discriminator so any of the six task families
    (coding/SWE/tool-use/SQL/math/CTF/RAG) rebuilds as its own class. Legacy
    MBPP/KernelBench rows carry no `family` key and map to CodingTask."""
    from .task import task_from_dict

    meta = getattr(sample, "metadata", None) or {}
    if "task" in meta:
        td = meta["task"]
        if "family" in td:
            return task_from_dict(td)
        return CodingTask(**td)  # legacy coding/kernel rows
    # Fallback: prompt string is a JSON-encoded task.
    import json
    return CodingTask(**json.loads(sample.prompt))
