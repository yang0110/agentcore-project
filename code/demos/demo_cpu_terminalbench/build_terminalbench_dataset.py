#!/usr/bin/env python3
"""Build demo_cpu_terminalbench's train/eval dataset from tasks/manifest.json (offline, stdlib-only).

Each Terminal-Bench task becomes ONE self-describing slime row:

    {"prompt": <instruction>, "label": <task>, "metadata": {"task": {family:"terminalbench",
        task_id, prompt, max_turns, spec:{image, difficulty, category, parser}}}}

`agentcore_slime_demo.generate._task_from_sample` reads `metadata.task.family` and rebuilds the
TerminalBenchTask, whose `spec.image` (the task name) routes the episode to ITS task's Runtime ARN
(AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE) — the large image pool. The sandbox is task-agnostic; all
task specificity is the baked container image + its hidden pytest oracle (grade.sh).

    python3 build_terminalbench_dataset.py                 # all tasks in the manifest
    TB_MAX_TURNS=8 python3 build_terminalbench_dataset.py    # cap agentic turns/episode (default 6)
    python3 build_terminalbench_dataset.py --limit 20        # first 20 tasks (cheap smoke)
    python3 build_terminalbench_dataset.py --repeat 4        # repeat each task 4x in train

Reads tasks/manifest.json (from fetch_terminalbench.py); writes data/tb_train.jsonl + data/tb_eval.jsonl.
Idempotent (overwrites). No network, no GPU, no AWS.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # this demo's vendored agentcore_slime_demo/

from agentcore_slime_demo.task_families import TerminalBenchTask  # noqa: E402


def _task(entry: dict, max_turns: int) -> TerminalBenchTask:
    return TerminalBenchTask(
        task_id=entry["task"],
        prompt=entry.get("instruction", ""),
        max_turns=max_turns,
        spec={
            "image": entry["image"],          # task name -> Runtime ARN routing tag
            "difficulty": entry.get("difficulty", "unknown"),
            "category": entry.get("category", "unknown"),
            "parser": entry.get("parser", "pytest"),
        },
    )


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
    ap.add_argument("--out-dir", default=str(HERE / "data"))
    ap.add_argument("--limit", type=int, default=0, help="first N tasks only (0 = all)")
    ap.add_argument("--repeat", type=int, default=1, help="repeat each task N times in train")
    args = ap.parse_args()

    max_turns = int(os.environ.get("TB_MAX_TURNS", "6"))
    mpath = Path(args.manifest)
    if not mpath.exists():
        sys.exit(f"{mpath} not found — run `python3 fetch_terminalbench.py` first")
    entries = json.loads(mpath.read_text())["tasks"]
    if args.limit:
        entries = entries[: args.limit]
    if not entries:
        sys.exit("no tasks in manifest")

    tasks = [_task(e, max_turns) for e in entries]
    train = [_row(t, max_turns) for _ in range(max(1, args.repeat)) for t in tasks]
    eval_ = [_row(t, max_turns) for t in tasks]

    out = Path(args.out_dir)
    _write(out / "tb_train.jsonl", train)
    _write(out / "tb_eval.jsonl", eval_)
    print(f"wrote {len(train)} train + {len(eval_)} eval rows "
          f"({len(tasks)} tasks x{args.repeat}, max_turns={max_turns}) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
