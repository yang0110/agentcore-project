"""Resolve AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE for a run (called by run.sh; prints the JSON map).

ONE ROW = ONE TASK + ONE IMAGE, and ONE IMAGE = ONE AgentCore Runtime. Every dataset row names its
Terminal-Bench task image in metadata.task.image; the image -> Runtime ARN map is the pool map written by
step 4 (terminalbench/tasks/arn_by_image.json, override with POOL_MAP).

Any image the dataset needs that has no Runtime is an error: an unmapped row would otherwise run on
another image, where its oracle fails and the run trains on a silent 0 reward.

    python3 image_routing.py data/terminalbench_train.jsonl data/terminalbench_eval.jsonl   # -> the JSON map
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

TB_POOL_MAP = os.path.join(HERE, "terminalbench", "tasks", "arn_by_image.json")


def rows_tasks(paths: list[str]):
    from agentcore_slime_demo.generate import _task_from_sample

    class _Row:   # the slice of a slime Sample that _task_from_sample reads
        def __init__(self, row: dict) -> None:
            self.metadata, self.prompt = row.get("metadata") or {}, row.get("prompt")

    for p in paths:
        if not os.path.exists(p):
            continue
        for line in open(p):
            if line.strip():
                yield _task_from_sample(_Row(json.loads(line)))


def main(argv: list[str]) -> int:
    tasks = list(rows_tasks(argv))
    need = {getattr(t, "image", "") or "" for t in tasks}
    pool = os.environ.get("POOL_MAP") or TB_POOL_MAP
    if not os.path.exists(pool):
        print(f"!! the pool map {pool} is missing — deploy the per-task Runtimes first: "
              "bash 2_build_image.sh && bash 4_deploy_runtime.sh", file=sys.stderr)
        return 2
    mapping = json.load(open(pool))
    missing = sorted(i or "<no image>" for i in need if not mapping.get(i))
    if missing:
        head = ", ".join(missing[:8]) + (f" … (+{len(missing) - 8})" if len(missing) > 8 else "")
        print(f"!! the dataset needs task image(s) [{head}] but no Runtime is mapped for them. Rebuild the "
              "dataset from the deployed pool (bash 1_prepare_data.sh) or deploy the missing tasks.",
              file=sys.stderr)
        return 2
    print(f"[image_routing] {len(tasks)} rows -> {len(need)} image(s)/Runtime(s)", file=sys.stderr)
    print(json.dumps(mapping))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
