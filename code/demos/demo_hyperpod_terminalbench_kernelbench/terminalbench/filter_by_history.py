#!/usr/bin/env python3
"""Drop train rows whose task never gave GRPO a gradient in earlier runs (step 1, after the build).

GRPO learns only from a group whose rewards differ. A task whose past episodes ALL scored the same
(never solved, no test ever passed; or KernelBench always correct-but-not-faster = 0.5) only adds
zero-gradient groups. In train48 run 2 that was most of the batch: 32 of 48 steps had grad_norm 0.

Reads every logs/*.demo.csv newer than TB_FILTER_SINCE (the run's timestamp in its file name;
default 20261004_143000 = smoke 5, the first run with the KernelBench speedup gate). Re-scores each
Terminal-Bench episode with the CURRENT training reward (1 if solved, else TB_PARTIAL_CREDIT x tests
passed / total) and keeps KernelBench's logged reward. A task is dropped when it has at least
TB_FILTER_MIN_EPISODES (default 8) valid episodes (no infra error) and their reward never varied.
Tasks with less history are kept, so a fresh checkout (no logs) trains on everything.

    python3 terminalbench/filter_by_history.py --train data/terminalbench_train.jsonl --logs logs

Rewrites --train in place (the unfiltered rows stay in <train>.all.jsonl) and writes
data/task_filter.csv (task, episodes, reward mean / min / max, kept). Opt-in: prepare_data.sh runs it only with
TB_FILTER_HISTORY=1 (off by default since 2026-10-06; the demo trains on the full task set).
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import re
from pathlib import Path


def _f(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def episode_rewards(log_dir: str, since: str, partial: float) -> dict[str, list[float]]:
    out: dict[str, list[float]] = collections.defaultdict(list)
    for path in sorted(glob.glob(os.path.join(log_dir, "*.demo.csv"))):
        m = re.search(r"_(\d{8}_\d{6})\.demo\.csv$", path)
        if not m or m.group(1) < since:
            continue
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                solved = _f(r.get("solved"))
                if solved is None or r.get("infra_error") not in ("", "none", None) or r.get("cancelled"):
                    continue          # not an episode-end record, or the sandbox failed it
                if r.get("substrate") == "gpu":
                    rew = _f(r.get("reward"))
                else:
                    total, passed = _f(r.get("tests_total")) or 0.0, _f(r.get("tests_passed")) or 0.0
                    rew = 1.0 if solved >= 1 else (partial * min(1.0, passed / total) if total > 0 else 0.0)
                if rew is not None:
                    out[r["task"]].append(rew)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--logs", required=True)
    ap.add_argument("--report", default="")
    a = ap.parse_args()
    since = os.environ.get("TB_FILTER_SINCE", "20261004_143000")
    min_eps = int(os.environ.get("TB_FILTER_MIN_EPISODES", "8"))
    partial = float(os.environ.get("TB_PARTIAL_CREDIT", "0.5"))
    train = Path(a.train)
    full = train.with_suffix(".all.jsonl")
    rows = [json.loads(line) for line in train.read_text().splitlines() if line.strip()]
    full.write_text("".join(json.dumps(r) + "\n" for r in rows))
    hist = episode_rewards(a.logs, since, partial)

    kept, report = [], []
    for r in rows:
        tid = r["metadata"]["task"]["task_id"]
        rs = hist.get(tid, [])
        keep = len(rs) < min_eps or max(rs) > min(rs)
        report.append({"task": tid, "episodes": len(rs), "reward_mean": round(sum(rs) / len(rs), 4) if rs else "",
                       "reward_min": min(rs) if rs else "", "reward_max": max(rs) if rs else "", "kept": int(keep)})
        if keep:
            kept.append(r)
    train.write_text("".join(json.dumps(r) + "\n" for r in kept))
    rep = Path(a.report or train.parent / "task_filter.csv")
    with rep.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(report[0]))
        w.writeheader()
        w.writerows(report)
    no_hist = sum(1 for x in report if x["episodes"] < min_eps)
    print(f"# filter_by_history: kept {len(kept)} of {len(rows)} train rows ({no_hist} with < {min_eps} "
          f"episodes of history, kept); dropped {len(rows) - len(kept)} whose reward never varied "
          f"(runs since {since}) -> {train} (all rows: {full.name}; per task: {rep.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
