#!/usr/bin/env python3
"""Build slime-format JSONL for the demo_cpu_math task-family zoo (SWE, tool-use, SQL, math, CTF, RAG).

demo_cpu_math's in-process harness is task-agnostic (see agentcore_slime_demo/task_families.py), so
each family is just a different seed set + checker feeding the SAME custom generate. This emits
`<family>_train.jsonl` + `<family>_eval.jsonl` per family, in the exact row schema the MBPP/
KernelBench builder uses:

    {"prompt": <task text>, "label": <task_id>, "metadata": {"task": {<family task kwargs>}}}

`agentcore_slime_demo.generate._task_from_sample` reads `metadata.task.family` and rebuilds the
right Task class. `mock_attempts` (offline MockPolicy only) are intentionally NOT written, so a
reference answer can never leak into a real-RL prompt.

Run inside the container (stdlib-only, no network/GPU needed):
    python /work/code/demos/demo_cpu_math/build_family_datasets.py --out-dir /work/code/demos/demo_cpu_math/data
Then point demo_cpu_math/run.sh at one family, e.g. `FAMILY=sql bash run.sh` (see run.sh).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agentcore_slime_demo.task_families import FAMILY_CLASSES


def _row(task, max_turns: int | None) -> dict:
    prompt = getattr(task, "initial_user_prompt", lambda: task.prompt)()
    return {"prompt": prompt, "label": task.task_id,
            "metadata": {"task": task.to_row_kwargs(max_turns=max_turns)}}


def _write(path: Path, tasks, max_turns) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for t in tasks:
            f.write(json.dumps(_row(t, max_turns)) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "data"))
    ap.add_argument("--max-turns", type=int, default=None,
                    help="override each task's max_turns (fewer => more reward variance in GRPO)")
    ap.add_argument("--eval-frac", type=float, default=0.25,
                    help="fraction of each family's seeds held out for eval (min 2)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    for cls in FAMILY_CLASSES:
        seeds = cls.seeds()
        n_eval = max(2, int(len(seeds) * args.eval_frac))
        eval_tasks, train_tasks = seeds[:n_eval], seeds[n_eval:]
        _write(out / f"{cls.family}_train.jsonl", train_tasks, args.max_turns)
        _write(out / f"{cls.family}_eval.jsonl", eval_tasks, args.max_turns)
        print(f"{cls.family:9s} -> {len(train_tasks):2d} train + {len(eval_tasks)} eval  "
              f"({out}/{cls.family}_train.jsonl)")
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
