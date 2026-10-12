"""Per-step TASK MIX of the batch the trainer actually consumes (slime --custom-rollout-log-function-path).

Wire with:
    --custom-rollout-log-function-path agentcore_slime_demo.batch_mix_log.log_rollout_data
    --custom-eval-rollout-log-function-path agentcore_slime_demo.batch_mix_log.log_eval_rollout_data

slime calls this once per training step with the flat list of samples it is about to train on. Under the
fully-async driver that batch is the first RB groups to FINISH, not the RB drawn next, so a slow task can
lag a step behind its draw; this records what was really trained on. It adds to the step's metrics:

    mix/n_tasks            distinct task families in the batch
    mix/episodes_<family>  episodes (samples) of that family in the batch
    mix/reward_<family>    their mean reward

and per held-out eval `eval/<name>_<family>` (that task's held-out reward) next to the pooled
`eval/<name>`. Both hooks then run slime's stock logging with those keys merged in (so they land in the `perf N:` line, wandb and
metrics.py's per-step CSV), and prints one `[batch]` line per step for a quick read of the log.
"""

from __future__ import annotations

import collections
import copy


def _family(sample) -> str:
    from .generate import _task_from_sample

    try:
        return getattr(_task_from_sample(sample), "family", "coding") or "coding"
    except Exception:  # noqa: BLE001 — a logging hook must never break a training step
        return "?"


def mix_metrics(args, samples) -> dict:
    by_fam: dict[str, list[float]] = collections.defaultdict(list)
    for s in samples:
        by_fam[_family(s)].append(float(s.get_reward_value(args) or 0.0))
    out = {"mix/n_tasks": len(by_fam)}
    for fam, rs in sorted(by_fam.items()):
        out[f"mix/episodes_{fam}"] = len(rs)
        out[f"mix/reward_{fam}"] = sum(rs) / len(rs)
    return out


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    from slime.observability import rollout_metrics

    mix = mix_metrics(args, samples)
    fams = " ".join(f"{k[len('mix/episodes_'):]}={v}" for k, v in mix.items() if k.startswith("mix/episodes_"))
    print(f"[batch] rollout={rollout_id} n_tasks={mix['mix/n_tasks']} {fams}", flush=True)
    stock = copy.copy(args)
    stock.custom_rollout_log_function_path = None     # run slime's own logger, not this hook again
    rollout_metrics.log_rollout_data(rollout_id, stock, samples, {**(rollout_extra_metrics or {}), **mix},
                                     rollout_time)
    return True                                        # slime's default logging already ran above


def log_eval_rollout_data(rollout_id, args, data, extra_metrics=None) -> bool:
    from slime.observability import rollout_metrics

    extra = dict(extra_metrics or {})
    for name, d in data.items():
        samples, rewards = d.get("samples"), d.get("rewards")
        if not samples or not rewards or len(samples) != len(rewards):
            continue
        by_fam: dict[str, list[float]] = collections.defaultdict(list)
        for s, r in zip(samples, rewards):
            by_fam[_family(s)].append(float(r or 0.0))
        if len(by_fam) > 1:
            for fam, rs in sorted(by_fam.items()):
                extra[f"eval/{name}_{fam}"] = sum(rs) / len(rs)
    stock = copy.copy(args)
    stock.custom_eval_rollout_log_function_path = None
    rollout_metrics.log_eval_rollout_data(rollout_id, stock, data, extra)
    return True
