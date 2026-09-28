"""Resolve AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE for a run (called by run.sh; prints the JSON map).

ONE ROW = ONE TASK + ONE IMAGE, and ONE IMAGE = ONE AgentCore Runtime. Every dataset row names its
sandbox image in metadata.task.image (older rows fall back to the family's tag, task.py /
task_families.py); rows may share an image. The images and their Runtimes:

    cpu                         deploy/Dockerfile.sandbox.cpu  (stdlib; AGENTCORE_SANDBOX_RUNTIME_ARNS, K=2)
    datasci / scipy / sklearn   deploy/Dockerfile.sandbox.libs (aliased to the "libs" Runtime)
    <terminal-bench task>       one image + Runtime per task   (terminalbench/tasks/arn_by_image.json,
                                                                VARIANT=terminalbench steps 2-4)

Any image the dataset needs that has no Runtime is an error: an unmapped row would otherwise run on
another image, where its checker fails and the run trains on a silent 0 reward.

    python3 image_routing.py data/libs_train.jsonl data/libs_eval.jsonl        # -> the JSON map
    python3 image_routing.py --families data/x_train.jsonl                     # -> "coding,terminalbench"
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

LIBS_TAGS = ("datasci", "scipy", "sklearn")   # all served by deploy/Dockerfile.sandbox.libs
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
    if argv and argv[0] == "--families":
        print(",".join(sorted({getattr(t, "family", "coding") for t in rows_tasks(argv[1:])})))
        return 0
    tasks = list(rows_tasks(argv))
    need = {getattr(t, "image", "cpu") or "cpu" for t in tasks}
    mapping = json.loads(os.environ.get("AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE") or "{}")
    if mapping.get("libs"):
        for tag in LIBS_TAGS:
            mapping.setdefault(tag, mapping["libs"])
    if any(getattr(t, "family", "") == "terminalbench" for t in tasks):
        pool = os.environ.get("POOL_MAP") or TB_POOL_MAP
        if not os.path.exists(pool):
            print(f"!! the dataset has Terminal-Bench rows but the pool map {pool} is missing — deploy the "
                  "per-task Runtimes first: VARIANT=terminalbench bash 2_build_image.sh && "
                  "VARIANT=terminalbench bash 4_deploy_runtime.sh", file=sys.stderr)
            return 2
        mapping.update(json.load(open(pool)))
    # "cpu" is served by AGENTCORE_SANDBOX_RUNTIME_ARNS (round-robin over K Runtimes), not the map.
    missing = sorted(i for i in need - {"cpu"} if not mapping.get(i))
    if missing:
        head = ", ".join(missing[:8]) + (f" … (+{len(missing) - 8})" if len(missing) > 8 else "")
        print(f"!! the dataset needs sandbox image(s) [{head}] but no Runtime is mapped for them. "
              "libs: deploy/demo.env IMAGES includes libs:Dockerfile.sandbox.libs (steps 2-4), so "
              "deploy/runtime.env carries AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE with a \"libs\" entry. "
              "Terminal-Bench: rebuild the dataset from the deployed pool (DATASET=terminalbench bash "
              "1_prepare_data.sh) or deploy the missing tasks.", file=sys.stderr)
        return 2
    print(f"[image_routing] {len(tasks)} rows -> {len(need)} image(s)/Runtime(s)", file=sys.stderr)
    print(json.dumps(mapping) if mapping else "")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
