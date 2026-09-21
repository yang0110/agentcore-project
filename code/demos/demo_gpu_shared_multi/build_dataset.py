#!/usr/bin/env python3
"""Build slime-format JSONL for the P4 real training run — Demo A (MBPP) or Demo B (KernelBench).

Each row is what slime's dataset loader expects with
    --input-key prompt --label-key label --metadata-key metadata
and it carries the full CodingTask under metadata.task, which
`agentcore_slime_demo.generate._task_from_sample` reconstructs verbatim:

    {"prompt": <task text+tests>, "label": <task_id>,
     "metadata": {"task": {<CodingTask kwargs>}}}

Both demos use the SAME row schema and the SAME custom generate — they differ only in the
task source (MBPP CPU code-repair vs KernelBench CUDA-op repair). Run inside the container
(has `datasets` + HF cache), writing into the calling demo's own data/ dir. E.g.:
    # MBPP (Demo A / A2 / A3): 500-prompt train + 8-prompt held-out eval
    python /work/code/demos/demo_cpu_mbpp/build_dataset.py \
        --dataset mbpp --split "train+test" --limit 500 --out /work/code/demos/demo_cpu_mbpp/data/mbpp_train.jsonl
    # KernelBench (Demo B / B2 / B3): level_1 train (100) + level_2 held-out eval (20)
    python /work/code/demos/demo_gpu_single/build_dataset.py \
        --dataset kernelbench --level level_1 --limit 100 --out /work/code/demos/demo_gpu_single/data/kernelbench_train.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

from agentcore_slime_demo.mbpp import load_mbpp
from agentcore_slime_demo.kernelbench import load_kernelbench_tasks, load_kernelbench_hf


def task_to_row(t, max_turns: int | None = None) -> dict:
    task_kwargs = {
        "task_id": t.task_id,
        "prompt": t.prompt,
        "checks": t.checks,
        "check_mode": t.check_mode,
        "test_setup": t.test_setup,
        "max_turns": max_turns if max_turns is not None else t.max_turns,
        # mock_attempts intentionally dropped: real RL never uses them, and keeping the
        # reference solution out of the sample avoids any chance of leakage into the prompt.
    }
    return {"prompt": t.prompt, "label": t.task_id, "metadata": {"task": task_kwargs}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mbpp", "kernelbench", "kernelbench-stub"], default="mbpp",
                    help="mbpp = Demo A (CPU code-repair); kernelbench = Demo B, the REAL "
                         "ScalingIntelligence/KernelBench (270 CUDA problems, pick --level); "
                         "kernelbench-stub = the hand-written 3-op smoke set")
    ap.add_argument("--split", default="train")
    ap.add_argument("--level", default="level_1",
                    help="KernelBench level: level_1 (100 single ops), level_2 (100 fusions), "
                         "level_3 (50 archs), level_4 (20). Used only with --dataset kernelbench.")
    ap.add_argument("--limit", type=int, default=32)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--max-turns", type=int, default=None,
                    help="override each task's max_turns (fewer turns => harder => more "
                         "reward variance within GRPO groups, i.e. a real learning signal)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.dataset == "kernelbench":
        # REAL KernelBench from HF: slice a level with --offset/--limit (disjoint for train/eval).
        tasks = load_kernelbench_hf(level=args.level, limit=args.limit or None,
                                    offset=args.offset,
                                    max_turns=args.max_turns if args.max_turns is not None else 5)
    elif args.dataset == "kernelbench-stub":
        # Hand-written 3-op smoke set (no split/offset); limit just truncates.
        tasks = load_kernelbench_tasks()[args.offset:]
        if args.limit:
            tasks = tasks[: args.limit]
    else:
        tasks = load_mbpp(split=args.split, limit=args.limit, offset=args.offset)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for t in tasks:
            f.write(json.dumps(task_to_row(t, max_turns=args.max_turns)) + "\n")
    print(f"wrote {len(tasks)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
