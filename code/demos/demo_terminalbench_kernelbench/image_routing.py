"""Resolve TB_ROUTE_BY_IMAGE for a run (called by run.sh; prints the JSON map).

ONE ROW = ONE TASK + ONE IMAGE + ONE ROUTE. Every dataset row names its Terminal-Bench task image in
metadata.task.image; the route map (terminalbench/tasks/route_by_image.json, override with POOL_MAP)
sends it to one of two substrates:
    "arn:aws:bedrock-agentcore:..."   its own AgentCore microVM Runtime (step 4)
    "docker://<m5 ip>:<port>/<task>"  the m5 Docker host, which built its compose project (step 2)
    "gpu:arn:aws:bedrock-agentcore:…" its own Runtime on the g6 capacity provider (TB_GPU_TASKS, step 4)
    "gpuec2:kernelbench-gpu"          OPT-IN, TB_KB_SUBSTRATE=ec2: KernelBench on the warm plain-EC2 g6 pool
                                      (deploy/g6_host.py) instead of the AgentCore g6 Runtime (the default)

Any image the dataset needs that has no route is an error: an unmapped row would otherwise run on
another image, where its oracle fails and the run trains on a silent 0 reward.

    python3 image_routing.py data/terminalbench_train.jsonl data/terminalbench_eval.jsonl   # -> the JSON map
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

TB_POOL_MAP = os.path.join(HERE, "terminalbench", "tasks", "route_by_image.json")
KB_IMAGE = "kernelbench-gpu"


def apply_kb_substrate(mapping: dict) -> dict:
    """TB_KB_SUBSTRATE=agentcore (default): KernelBench keeps its gpu:<AgentCore g6 Runtime> route.
    =ec2: it goes to the plain-EC2 g6 warm pool instead (gpuec2:<image>, needs deploy/g6_host.json)."""
    sub = os.environ.get("TB_KB_SUBSTRATE", "agentcore").strip().lower()
    if sub not in ("agentcore", "ec2"):
        raise SystemExit(f"TB_KB_SUBSTRATE={sub!r}: agentcore | ec2")
    if sub == "ec2":
        mapping = {**mapping, KB_IMAGE: f"gpuec2:{KB_IMAGE}"}
    return mapping


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
    mapping = apply_kb_substrate(json.load(open(pool)))
    missing = sorted(i or "<no image>" for i in need if not mapping.get(i))
    if missing:
        head = ", ".join(missing[:8]) + (f" … (+{len(missing) - 8})" if len(missing) > 8 else "")
        print(f"!! the dataset needs task image(s) [{head}] but no route is mapped for them. Rebuild the "
              "dataset from the deployed pool (bash 1_prepare_data.sh) or deploy the missing tasks.",
              file=sys.stderr)
        return 2
    docker = sum(1 for i in need if str(mapping[i]).startswith("docker://"))
    gpu = sum(1 for i in need if str(mapping[i]).startswith("gpu:"))
    g6ec2 = sum(1 for i in need if str(mapping[i]).startswith("gpuec2:"))
    print(f"[image_routing] {len(tasks)} rows -> {len(need)} image(s): {len(need) - docker - gpu - g6ec2} AgentCore "
          f"microVM Runtime(s) + {docker} m5 docker task(s) + {gpu} g6 GPU Runtime(s) + {g6ec2} plain-EC2 g6 task(s)",
          file=sys.stderr)
    print(json.dumps(mapping))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
