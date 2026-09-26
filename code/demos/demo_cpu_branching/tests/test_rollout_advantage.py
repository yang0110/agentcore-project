"""Unit test for the rollout-level GRPO advantage function.

Run:  python3 -m pytest code/demos/demo_cpu_branching/tests/test_rollout_advantage.py
  or: python3 code/demos/demo_cpu_branching/tests/test_rollout_advantage.py   (has a __main__)

No slime / no GPU needed — the function only touches rollout_data dict + torch. The test builds a
FAN-OUT rollout_data by hand and pins down the one property that matters: the baseline is taken over
DISTINCT ROLLOUTS, not over fanned-out leaves."""

from __future__ import annotations

import os
import sys
import types

import torch

# Import the module directly by path so the test needs neither the package installed nor slime.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.join(_HERE, os.pardir, "agentcore_slime_demo")
sys.path.insert(0, os.path.abspath(_PKG))
import rollout_advantage  # noqa: E402


def _args(grpo_std_normalization: bool, n_samples_per_prompt: int = 2):
    return types.SimpleNamespace(grpo_std_normalization=grpo_std_normalization,
                                 n_samples_per_prompt=n_samples_per_prompt)


def _leaf(reward, rollout_id, sample_index, n_tokens):
    """One leaf Sample's contribution to rollout_data, as (reward, rid, sidx, kl_tensor)."""
    return reward, rollout_id, sample_index, torch.zeros(n_tokens, dtype=torch.float32)


def _build(leaves):
    rewards, rollout_ids, sample_indices, kl = [], [], [], []
    for reward, rid, sidx, kl_t in leaves:
        rewards.append(reward)
        rollout_ids.append(rid)
        sample_indices.append(sidx)
        kl.append(kl_t)
    return {"rewards": rewards, "rollout_ids": rollout_ids, "sample_indices": sample_indices, "kl": kl}


def test_baseline_over_distinct_rollouts_not_leaves():
    # Prompt 0: rollout A reward 1.0 fanned into 3 leaves; rollout B reward 0.0, 1 leaf.
    # Baseline over DISTINCT rollouts = (1.0 + 0.0)/2 = 0.5  => A adv +0.5, B adv -0.5.
    # (If it wrongly baselined over LEAVES: mean = 3/4 = 0.75 => A +0.25, B -0.75. Guard against it.)
    rd = _build([
        _leaf(1.0, "A", 0, 4),
        _leaf(1.0, "A", 0, 2),
        _leaf(1.0, "A", 0, 7),
        _leaf(0.0, "B", 1, 3),   # slime: each rollout has its own index; A,B = group 0 at NS=2
    ])
    rollout_advantage.rollout_level_grpo_advantage(_args(False), rd)

    adv = rd["advantages"]
    # Each leaf's advantage tensor is constant across its tokens and matches its length.
    for i, kl_t in enumerate(rd["kl"]):
        assert adv[i].shape == kl_t.shape
        assert torch.allclose(adv[i], adv[i][0].expand_as(adv[i]))
    # All three A leaves share one scalar; B its own.
    a_vals = {round(adv[i][0].item(), 6) for i in range(3)}
    assert a_vals == {0.5}, a_vals
    assert round(adv[3][0].item(), 6) == -0.5
    # returns == advantages for GRPO.
    for i in range(4):
        assert torch.equal(rd["returns"][i], adv[i])


def test_std_normalization():
    rd = _build([
        _leaf(1.0, "A", 0, 2),
        _leaf(0.0, "B", 1, 2),
    ])
    rollout_advantage.rollout_level_grpo_advantage(_args(True), rd)
    # baseline 0.5, population std over {1,0} = 0.5 => adv = ±0.5/(0.5+1e-6) ≈ ±1.0
    assert abs(rd["advantages"][0][0].item() - 1.0) < 1e-3
    assert abs(rd["advantages"][1][0].item() + 1.0) < 1e-3


def test_multiple_prompt_groups_are_independent():
    # Prompt 0: A=1.0 (2 leaves), B=0.0 (1 leaf) -> ±0.5.  Prompt 1: C=1.0, D=1.0 -> both 0.0.
    rd = _build([
        _leaf(1.0, "A", 0, 2),
        _leaf(1.0, "A", 0, 3),
        _leaf(0.0, "B", 1, 2),
        _leaf(1.0, "C", 2, 2),
        _leaf(1.0, "D", 3, 2),
    ])
    rollout_advantage.rollout_level_grpo_advantage(_args(False), rd)
    adv = rd["advantages"]
    assert round(adv[0][0].item(), 6) == 0.5
    assert round(adv[1][0].item(), 6) == 0.5
    assert round(adv[2][0].item(), 6) == -0.5
    assert round(adv[3][0].item(), 6) == 0.0  # C vs baseline 1.0
    assert round(adv[4][0].item(), 6) == 0.0  # D vs baseline 1.0


def test_groups_follow_slime_index_layout():
    # Regression (2026-09-25 smoke: every advantage 0). slime numbers rollouts 0,1,2,3 with NS=2, so
    # {0,1} and {2,3} are the prompt groups. Grouping by the raw index would make 4 singleton groups.
    rd = _build([
        _leaf(1.0, "r0", 0, 2), _leaf(1.0, "r0", 0, 2),   # r0 fanned into 2 leaves
        _leaf(0.0, "r1", 1, 2),
        _leaf(0.0, "r2", 2, 2),
        _leaf(0.0, "r3", 3, 2),                           # group 1 all-zero -> adv 0
    ])
    rollout_advantage.rollout_level_grpo_advantage(_args(False, 2), rd)
    got = [round(a[0].item(), 6) for a in rd["advantages"]]
    assert got == [0.5, 0.5, -0.5, 0.0, 0.0], got


if __name__ == "__main__":
    test_baseline_over_distinct_rollouts_not_leaves()
    test_std_normalization()
    test_multiple_prompt_groups_are_independent()
    test_groups_follow_slime_index_layout()
    print("OK: all rollout-advantage tests passed")
