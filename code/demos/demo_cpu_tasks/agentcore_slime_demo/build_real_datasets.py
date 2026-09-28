#!/usr/bin/env python3
"""Build slime-format JSONL from a REAL HuggingFace agentic-RL dataset.

Emits the same row schema the MBPP / task-family builders use —
`{"prompt", "label", "metadata": {"task": {...}}}` — so the unchanged in-process harness
(`agentcore_slime_demo.generate`) rebuilds the right Task class and runs its checker. Coding
tasks (CodingTask) serialize the CodingTask kwargs directly; family tasks (MathTask, an
EpisodicTask) serialize via `to_row_kwargs` (carrying `family` + `spec`).

Run inside the container (network needed once to pull HF rows):
    python build_real_datasets.py --dataset gsm8k     --split train --limit 200 --out data/gsm8k_train.jsonl
    python build_real_datasets.py --dataset gsm8k     --split test  --limit 40  --out data/gsm8k_eval.jsonl
    python build_real_datasets.py --dataset humaneval --split test  --limit 120 --out data/humaneval_train.jsonl
    python build_real_datasets.py --dataset math500   --split test  --limit 200 --out data/math500_train.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `agentcore_slime_demo` imports

from agentcore_slime_demo.real_datasets import LOADERS


def _row(task, max_turns: int | None) -> dict:
    prompt = getattr(task, "initial_user_prompt", lambda: task.prompt)()
    if hasattr(task, "to_row_kwargs"):        # EpisodicTask families (math): {family, ..., spec}
        meta_task = task.to_row_kwargs(max_turns=max_turns)
    else:                                     # CodingTask (coding/humaneval): flat kwargs
        meta_task = {
            "task_id": task.task_id,
            "image": task.image,              # the row's sandbox image (= its Runtime)
            "prompt": task.prompt,
            "checks": task.checks,
            "check_mode": task.check_mode,
            "test_setup": task.test_setup,
            "max_turns": max_turns if max_turns is not None else task.max_turns,
        }
    return {"prompt": prompt, "label": task.task_id, "metadata": {"task": meta_task}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(LOADERS),
                    help="real HF dataset -> task family")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--max-turns", type=int, default=None,
                    help="override each task's max_turns (fewer => more GRPO reward variance)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tasks = LOADERS[args.dataset](split=args.split, limit=args.limit, offset=args.offset)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for t in tasks:
            f.write(json.dumps(_row(t, args.max_turns)) + "\n")
    print(f"wrote {len(tasks)} rows ({args.dataset}/{args.split}) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
