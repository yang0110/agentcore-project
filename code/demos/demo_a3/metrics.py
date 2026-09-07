#!/usr/bin/env python3
"""Parse a slime training log into a per-step CSV and (optionally) plot the key curves.

slime logs metrics to stdout (and to wandb/tensorboard if configured), but for the demos we
want a self-contained, dependency-light record: a CSV you can open anywhere + a PNG of the
curves that matter. This script reads the raw ray-job log that `run_async.sh` /
`run_agentcore_mbpp_qwen3_4b.sh` tee to a file and extracts the three metric families slime
prints, each tagged with the rollout/step id:

    model.py:918            - step N:    {'train/pg_loss': ..., 'train/grad_norm': ..., ...}
    train_metric_utils.py   - rollout N: {'rollout/raw_reward': ..., 'rollout/advantages': ...}
    *                       - perf N:    {'perf/step_time': ..., 'perf/wait_time_ratio': ...}
    rollout_metrics.py:247  - eval N:    {'eval/<name>': <solve rate>, ..., 'eval/step': S}

The train families are merged by id (rollout N is trained as step N) and written to a per-step
CSV. The `eval` family is a DIFFERENT cadence (we run held-out eval every 10 steps, no gradient)
so it is written to a SEPARATE eval CSV keyed by `eval/step`. Both CSVs, and a PNG of the key
curves, come from the one raw log. Pure stdlib for parsing; matplotlib only for --plot.

    python3 metrics.py train_async_demoB.log --csv metrics.csv --eval-csv eval.csv --plot fig.png
"""
from __future__ import annotations

import argparse
import ast
import csv
import re
from pathlib import Path

# " - step 5: {…}" / " - rollout 5: {…}" / " - perf 5: {…}" / " - eval 5: {…}"  (dict to EOL)
_LINE = re.compile(r"-\s+(step|rollout|perf|eval)\s+(\d+):\s+(\{.*\})\s*$")

# The columns we care about most, in display order; any other parsed keys are appended after.
KEY_COLUMNS = [
    "rollout_id",
    "rollout/raw_reward",          # solve rate this step (the RL signal)
    "train/grad_norm",             # is the policy actually moving?
    "train/pg_loss",
    "train/kl_loss",               # drift from the frozen reference
    "train/train_rollout_logprob_abs_diff",  # weight-sync correctness (want ~small/stable)
    "train/global_batch_size",
    "rollout/response_lengths",
    "perf/step_time",
    "perf/actor_train_tok_per_s",
    "perf/wait_time_ratio",        # pipeline balance (train vs rollout idle)
]

# Eval CSV columns: the held-out solve rate(s) keyed by the training step eval ran at.
EVAL_KEY_COLUMNS = ["eval/step"]

# Which columns to plot, and a human title for each panel. `eval/*` reward columns are discovered
# at runtime (their name depends on the eval dataset) and appended.
PLOT_COLUMNS = [
    ("rollout/raw_reward", "solve rate (raw_reward)"),
    ("train/grad_norm", "grad norm"),
    ("train/kl_loss", "KL(policy‖ref)"),
    ("train/train_rollout_logprob_abs_diff", "logprob abs diff (weight-sync)"),
    ("perf/wait_time_ratio", "wait-time ratio"),
    ("perf/actor_train_tok_per_s", "actor train tok/s"),
]


def parse_log(path: Path) -> tuple[dict[int, dict], dict[int, dict]]:
    """Parse the log into (train_rows, eval_rows).

    train_rows: {rollout_id: {metric: value}} merged across step/rollout/perf.
    eval_rows:  {eval_step: {eval/*: value}} — a separate cadence (eval every N steps), keyed by
                the `eval/step` value inside each eval dict (fallback: the logged id).
    """
    train_rows: dict[int, dict] = {}
    eval_rows: dict[int, dict] = {}
    for line in path.read_text(errors="replace").splitlines():
        m = _LINE.search(line)
        if not m:
            continue
        kind, sid, payload = m.group(1), int(m.group(2)), m.group(3)
        try:
            d = ast.literal_eval(payload)
        except (ValueError, SyntaxError):
            continue
        if not isinstance(d, dict):
            continue
        nums = {k: v for k, v in d.items() if isinstance(v, (int, float))}
        if kind == "eval":
            step = int(d.get("eval/step", sid))
            row = eval_rows.setdefault(step, {"eval/step": step})
            row.update(nums)
        else:
            row = train_rows.setdefault(sid, {"rollout_id": sid})
            row.update(nums)
    return train_rows, eval_rows


def write_csv(rows: dict[int, dict], out: Path, key_columns: list[str]) -> list[str]:
    ids = sorted(rows)
    extra = sorted({k for r in rows.values() for k in r} - set(key_columns))
    columns = key_columns + extra
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for i in ids:
            w.writerow(rows[i])
    return columns


def plot(rows: dict[int, dict], eval_rows: dict[int, dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt

    ids = sorted(rows)
    panels = [(c, t) for c, t in PLOT_COLUMNS if any(c in rows[i] for i in ids)]
    # add one panel per eval/* reward column (skip the eval/step key itself)
    eval_cols = sorted({k for r in eval_rows.values() for k in r} - {"eval/step"})
    for c in eval_cols:
        panels.append((c, f"held-out {c}"))
    if not panels:
        print("no plottable columns found; skipping figure")
        return
    ncol = 2
    nrow = -(-len(panels) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 3 * nrow), squeeze=False)
    for ax, (col, title) in zip(axes.flat, panels):
        src = eval_rows if col.startswith("eval/") else rows
        xs = [i for i in sorted(src) if col in src[i]]
        ys = [src[i][col] for i in xs]
        ax.plot(xs, ys, marker="o")
        ax.set_title(title)
        ax.set_xlabel("training step")
        ax.grid(True, alpha=0.3)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"wrote figure -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", help="path to the training log (ray job output tee'd by run_*.sh)")
    ap.add_argument("--csv", default=None, help="per-step train CSV path (default: <log>.csv)")
    ap.add_argument("--eval-csv", default=None,
                    help="per-eval CSV path (default: <log>.eval.csv); written only if eval ran")
    ap.add_argument("--plot", nargs="?", const="", default=None,
                    help="also write a PNG of the key curves (default: <log>.png)")
    args = ap.parse_args()

    log = Path(args.log)
    rows, eval_rows = parse_log(log)
    if not rows and not eval_rows:
        print(f"no metric rows found in {log} — is the run past step 1?")
        return 1

    if rows:
        csv_out = Path(args.csv) if args.csv else log.with_suffix(".csv")
        cols = write_csv(rows, csv_out, KEY_COLUMNS)
        print(f"wrote {len(rows)} step rows x {len(cols)} cols -> {csv_out}")

    if eval_rows:
        eval_out = Path(args.eval_csv) if args.eval_csv else log.with_suffix(".eval.csv")
        ecols = write_csv(eval_rows, eval_out, EVAL_KEY_COLUMNS)
        print(f"wrote {len(eval_rows)} eval rows x {len(ecols)} cols -> {eval_out}")

    if args.plot is not None:
        png = Path(args.plot) if args.plot else log.with_suffix(".png")
        plot(rows, eval_rows, png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
