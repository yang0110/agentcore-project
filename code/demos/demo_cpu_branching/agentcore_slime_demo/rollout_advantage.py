"""Rollout-level GRPO advantage — the reconciliation for one-episode-becomes-many-samples.

Wire into slime:
    --custom-advantage-function-path agentcore_slime_demo.rollout_advantage.rollout_level_grpo_advantage

WHY A CUSTOM FUNCTION IS NEEDED (the built-in path is wrong under fan-out):
  The stock GRPO path computes its baseline over SAMPLES. That is correct only when each prompt
  produces exactly `n_samples_per_prompt` samples. In demo_cpu_branching one episode fans out into a
  VARIABLE number of leaf Samples (parent chain + sub-agent chains + post-compaction chains — see
  claude_code_generate.py). Two things then break if we let the built-in path run:
    * `_post_process_rewards` (slime/ray/rollout.py) guards on
      `rewards.shape[-1] == n_samples_per_prompt * rollout_batch_size`. Fan-out inflates the sample
      count, the guard fails, and it falls to the else branch `view(-1, rewards.shape[-1])` — i.e.
      ONE giant group — so every sample is baselined against the whole batch. Wrong baseline.
    * Even if the reshape happened to fit, baselining over SAMPLES double-counts prolific episodes:
      an episode that forked into 5 leaves would contribute 5 votes to its own baseline, an episode
      that stayed linear contributes 1. The unit of credit assignment in RL is the EPISODE, not the
      leaf.
  So we run with `--disable-rewards-normalization` (raw rewards reach the actor) and compute the
  baseline ourselves, over DISTINCT ROLLOUTS.

WHAT THIS FUNCTION DOES:
  1. A "rollout" = one episode = all Samples sharing a `rollout_ids` value. Every sibling carries the
     SAME outcome reward, so the rollout's reward is just any sibling's reward.
  2. A "prompt group" = the `n_samples_per_prompt` distinct rollouts of one prompt. slime gives every
     rollout its OWN `index` (data_source.py: `sample.index = self.sample_index; self.sample_index += 1`
     for each of the NS copies), so `sample_indices` is NOT shared within a group; the group is
     `index // n_samples_per_prompt` (== slime's `group_index`, which rollout_data does not carry).
     Grouping by the raw index made every group a single rollout, baseline == its own reward, and
     every advantage 0 (seen in the 2026-09-25 smoke: pg_loss = grad_norm = 0 on every step).
  3. GRPO baseline for a prompt group = mean reward over the DISTINCT ROLLOUTS in that group (each
     rollout counts once, regardless of how many leaves it fanned into). Advantage of a rollout =
     reward - baseline, optionally / (std_over_distinct_rollouts + eps) when
     `args.grpo_std_normalization` is set.
  4. Broadcast: every Sample of a rollout gets that rollout's scalar advantage, as a per-token
     tensor `torch.ones_like(kl[i]) * adv_scalar`. Returns == advantages (GRPO has no learned value
     baseline; mirrors get_grpo_returns which sets returns = ones_like(kl) * reward).

  Combined with rollout-level token-mean loss (`--calculate-per-token-loss` OFF + the shared
  rollout_id => `rollout_mask_sums` denominator in rollout.py), each EPISODE contributes equally to
  the loss no matter how many leaves/tokens it produced. Advantage is per-episode; the loss
  normalizer is per-episode; the two agree.

CONTRACT (slime/backends/megatron_utils/loss.py:758-761): called AFTER KL is computed, on the last
pipeline stage, in the Megatron actor on this rank's data partition. Must set
`rollout_data["advantages"]` and `rollout_data["returns"]` in place as lists of per-sample tensors.
It assumes each prompt group is co-resident on the rank (true at dp_size=1, which the PP=2
arrangement in run.sh enforces)."""

from __future__ import annotations

from collections import defaultdict

import torch


def rollout_level_grpo_advantage(args, rollout_data) -> None:
    """Populate rollout_data["advantages"] and ["returns"] with per-episode GRPO advantages.

    See the module docstring for the full rationale. Grouping is by rollout_ids (one episode) then
    by prompt group (index // n_samples_per_prompt); the baseline is taken over DISTINCT rollouts, never leaves."""
    kl: list[torch.Tensor] = rollout_data["kl"]
    rewards: list[float] = rollout_data["rewards"]
    rollout_ids: list = rollout_data["rollout_ids"]
    sample_indices: list = rollout_data["sample_indices"]
    n = len(kl)
    n_per_prompt = max(1, int(getattr(args, "n_samples_per_prompt", 1) or 1))

    # --- 1. Collapse leaves → distinct rollouts. Every sibling shares the reward; keep one. ---
    rollout_reward: dict = {}
    rollout_of_prompt: dict = defaultdict(list)  # sample_index -> [distinct rollout_id, ...]
    for i in range(n):
        rid = rollout_ids[i]
        if rid not in rollout_reward:
            rollout_reward[rid] = float(rewards[i])
            rollout_of_prompt[int(sample_indices[i]) // n_per_prompt].append(rid)

    # --- 2. GRPO baseline per prompt group, over DISTINCT rollouts. ---
    std_norm = getattr(args, "grpo_std_normalization", True)
    rollout_adv: dict = {}
    for _prompt_idx, rids in rollout_of_prompt.items():
        group = torch.tensor([rollout_reward[rid] for rid in rids], dtype=torch.float32)
        baseline = group.mean()
        denom = (group.std(unbiased=False) + 1e-6) if (std_norm and group.numel() > 1) else 1.0
        for rid in rids:
            rollout_adv[rid] = float((rollout_reward[rid] - baseline) / denom)

    # --- 3. Broadcast each rollout's scalar advantage across all its leaves' tokens. ---
    advantages: list[torch.Tensor] = []
    returns: list[torch.Tensor] = []
    for i in range(n):
        adv_scalar = rollout_adv[rollout_ids[i]]
        adv_tensor = torch.ones_like(kl[i]) * adv_scalar
        advantages.append(adv_tensor)
        returns.append(adv_tensor)  # GRPO: returns == advantages (no learned value baseline)

    rollout_data["advantages"] = advantages
    rollout_data["returns"] = returns
