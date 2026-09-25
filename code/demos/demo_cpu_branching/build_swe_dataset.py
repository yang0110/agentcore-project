#!/usr/bin/env python3
"""Build the single-family SWE dataset for demo_cpu_branching.

Unlike demo_cpu_tasks:mixed (which INTERLEAVES families), this demo pins ONE family — SWE bug-fix — so
every rollout is the same kind of multi-turn task and the interesting variable is the FAN-OUT (how
many trainable leaves one episode produces), not the task type. Each row is self-describing:

    {"prompt": <task text>, "label": <task_id>, "metadata": {"task": {family:"swe", ...}}}

`agentcore_slime_demo.generate._task_from_sample` reads `metadata.task.family` and rebuilds the
SweTask (prompt + hidden checker + reward) per row — the same dispatch every other demo uses. The
real Claude Code CLI harness (claude_code_generate.py) then drives the episode and fans it out.

Offline + stdlib-only (SweTask.seeds() is hand-authored). Idempotent (overwrites). Run in-container:
    python3 /work/code/demos/demo_cpu_branching/build_swe_dataset.py \
        --out-dir /work/code/demos/demo_cpu_branching/data
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Import ONLY the vendored copy in this demo dir (self-contained; no /work/code on the path).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agentcore_slime_demo.task_families import SweTask


def _row(task, max_turns: int | None) -> dict:
    prompt = getattr(task, "initial_user_prompt", lambda: task.prompt)()
    return {"prompt": prompt, "label": task.task_id,
            "metadata": {"task": task.to_row_kwargs(max_turns=max_turns)}}


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the SWE bug-fix dataset for demo_cpu_branching.")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "data"))
    ap.add_argument("--max-turns", type=int, default=None,
                    help="override each task's max_turns (the CLI does its own multi-turn loop)")
    ap.add_argument("--eval-frac", type=float, default=0.25,
                    help="fraction of SWE seeds held out for eval (min 2)")
    ap.add_argument("--repeat", type=int, default=8,
                    help="replicate the TRAIN pool N times so dataset_size >= NR*RB and one epoch "
                         "spans the whole run => the only checkpoint written is the final one")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out_dir)

    seeds = SweTask.seeds()
    n_eval = max(2, int(len(seeds) * args.eval_frac))
    eval_seeds, train_seeds = seeds[:n_eval], seeds[n_eval:]

    train_rows = [_row(t, args.max_turns) for t in train_seeds]
    eval_rows = [_row(t, args.max_turns) for t in eval_seeds]

    train = list(train_rows) * max(1, args.repeat)
    rng.shuffle(train)  # replicas land at different step positions

    _write(out / "swe_train.jsonl", train)
    _write(out / "swe_eval.jsonl", eval_rows)

    print(f"SWE seeds: {len(seeds)} total -> {len(train_rows)} train seeds + {len(eval_rows)} eval")
    print(f"swe_train.jsonl -> {len(train)} prompts ({len(train_rows)} unique x {max(1, args.repeat)})")
    print(f"swe_eval.jsonl  -> {len(eval_rows)} prompts")
    print(f"out -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
