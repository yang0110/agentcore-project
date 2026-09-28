"""Shared episode helpers: `build_token_stream` and `_task_from_sample`.

This module used to hold an in-trainer `generate` (the agent loop run in the trainer process via
SGLangPolicy + run_episode). That path is gone: the agent loop runs in the episode's own sandbox
session, outside the trainer. This demo's hook is

    --custom-generate-function-path agentcore_slime_demo.terminalbench_pool_generate.generate

`build_token_stream` is a pure function (no slime, no network) so it can be unit-tested.
"""

from __future__ import annotations

from .harness import Trajectory
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
    """Removed: the agent loop no longer runs in the trainer."""
    raise RuntimeError("agentcore_slime_demo.generate.generate (in-trainer harness) was removed; "
                       "use --custom-generate-function-path agentcore_slime_demo.terminalbench_pool_generate.generate")


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
