#!/usr/bin/env python3
"""Build demo_cpu_terminalbench's Terminal-Bench train/eval dataset from tasks/manifest.json (offline, stdlib-only).

Each Terminal-Bench task becomes ONE self-describing slime row:

    {"prompt": <instruction>, "label": <task>, "metadata": {"task": {family:"terminalbench",
        task_id, prompt, max_turns, spec:{image, difficulty, category, parser}}}}

`agentcore_slime_demo.generate._task_from_sample` reads `metadata.task.family` and rebuilds the
TerminalBenchTask, whose `spec.image` (the task name) routes the episode to ITS task's Runtime ARN
(AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE) — the large image pool. The sandbox is task-agnostic; all
task specificity is the baked container image + its hidden pytest oracle (grade.sh).

    python3 build_terminalbench_dataset.py                 # all tasks in the manifest
    TB_MAX_TURNS=8 python3 build_terminalbench_dataset.py    # cap agentic turns/episode (default 8)
    python3 build_terminalbench_dataset.py --limit 20        # first 20 tasks (cheap smoke)
    python3 build_terminalbench_dataset.py --repeat 4        # repeat each task 4x in train
    python3 build_terminalbench_dataset.py --eval-frac 0.2   # held-out eval share (disjoint tasks)

Only tasks that are in the deployed pool map (tasks/arn_by_image.json) are emitted — an unmapped task
would be refused at routing time. Train and eval are DISJOINT tasks (a deterministic split, stratified
by difficulty), so eval/terminalbench is a held-out solve rate, not a replay of train.

Reads tasks/manifest.json (from fetch_terminalbench.py); writes ../data/terminalbench_{train,eval}.jsonl.
Idempotent (overwrites). No network, no GPU, no AWS.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEMO_DIR = HERE.parent
sys.path.insert(0, str(DEMO_DIR))  # the demo's vendored agentcore_slime_demo/

from agentcore_slime_demo.task_families import TerminalBenchTask  # noqa: E402


def _test_timeout(task: str) -> float:
    """max_test_timeout_sec from the task's task.yaml (Terminal-Bench default 180s)."""
    import re

    ty = HERE / "tasks" / task / "task.yaml"
    m = re.search(r"(?m)^max_test_timeout_sec:\s*([0-9.]+)", ty.read_text()) if ty.exists() else None
    return float(m.group(1)) if m else 180.0


def _task(entry: dict, max_turns: int) -> TerminalBenchTask:
    task = TerminalBenchTask(
        task_id=entry["task"],
        prompt=entry.get("instruction", ""),
        max_turns=max_turns,
        spec={
            "image": entry["image"],          # task name -> Runtime ARN routing tag
            "difficulty": entry.get("difficulty", "unknown"),
            "category": entry.get("category", "unknown"),
            "parser": entry.get("parser", "pytest"),
            "max_test_timeout_sec": _test_timeout(entry["task"]),
        },
    )
    task.image = entry["image"]               # the row's image: this task's own Runtime
    return task


def _row(task: TerminalBenchTask, max_turns: int) -> dict:
    return {"prompt": task.initial_user_prompt(), "label": task.task_id,
            "metadata": {"task": task.to_row_kwargs(max_turns=max_turns)}}


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(HERE / "tasks" / "manifest.json"))
    ap.add_argument("--out-dir", default=str(DEMO_DIR / "data"))
    ap.add_argument("--limit", type=int, default=0, help="first N tasks only (0 = all)")
    ap.add_argument("--repeat", type=int, default=1, help="repeat each task N times in train")
    ap.add_argument("--eval-frac", type=float, default=0.2, help="held-out eval share of tasks")
    ap.add_argument("--pool", default=str(HERE / "tasks" / "arn_by_image.json"),
                    help="deployed pool map; only its tasks are emitted ('' = no filter)")
    ap.add_argument("--oracle", default=str(HERE / "tasks" / "oracle.csv"),
                    help="validate_oracle.py result; only usable tasks (reference solution passes, "
                         "untouched env fails) are emitted ('' or missing file = no filter)")
    args = ap.parse_args()

    max_turns = int(os.environ.get("TB_MAX_TURNS", "8"))
    mpath = Path(args.manifest)
    if not mpath.exists():
        sys.exit(f"{mpath} not found — run `python3 fetch_terminalbench.py` first")
    entries = json.loads(mpath.read_text())["tasks"]
    if args.pool:
        pool = json.loads(Path(args.pool).read_text())
        dropped = [e["task"] for e in entries if e["task"] not in pool]
        entries = [e for e in entries if e["task"] in pool]
        print(f"pool filter: {len(entries)} deployed tasks kept, {len(dropped)} not in {args.pool}")
    if args.oracle and Path(args.oracle).exists():
        import csv

        with open(args.oracle) as f:
            usable = {r["task"] for r in csv.DictReader(f) if r.get("usable") == "1"}
        dropped = [e["task"] for e in entries if e["task"] not in usable]
        entries = [e for e in entries if e["task"] in usable]
        print(f"oracle filter: {len(entries)} usable tasks kept, {len(dropped)} dropped "
              f"(unvalidated, oracle fails here, or no signal): {dropped[:20]}")
    if args.limit:
        entries = entries[: args.limit]
    if not entries:
        sys.exit("no tasks in manifest")

    tasks = [_task(e, max_turns) for e in entries]
    # Deterministic, difficulty-stratified held-out split: within each difficulty (tasks sorted by
    # name), every k-th task goes to eval.
    k = max(2, round(1 / args.eval_frac)) if args.eval_frac > 0 else 0
    eval_ids = set()
    if k:
        by_diff: dict[str, list[str]] = {}
        for t in tasks:
            by_diff.setdefault(t.spec.get("difficulty", "?"), []).append(t.task_id)
        for ids in by_diff.values():
            eval_ids.update(sorted(ids)[k - 1::k])
    train_tasks = [t for t in tasks if t.task_id not in eval_ids]
    eval_tasks = [t for t in tasks if t.task_id in eval_ids]
    train = [_row(t, max_turns) for _ in range(max(1, args.repeat)) for t in train_tasks]
    eval_ = [_row(t, max_turns) for t in eval_tasks]

    out = Path(args.out_dir)
    _write(out / "terminalbench_train.jsonl", train)
    _write(out / "terminalbench_eval.jsonl", eval_)
    print(f"wrote {len(train)} train rows ({len(train_tasks)} tasks x{args.repeat}) + {len(eval_)} "
          f"held-out eval rows ({len(eval_tasks)} tasks), max_turns={max_turns} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
