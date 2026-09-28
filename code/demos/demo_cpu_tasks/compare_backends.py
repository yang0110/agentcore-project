#!/usr/bin/env python3
"""demo_cpu_tasks compare_backends.py (was demo_cpu_customruntime) — the SAME episode, two AgentCore backends, side by side.

The teaching point: an AgentCore sandbox has two shapes on the CPU (serverless microVM) substrate,
and they expose the *identical* task-agnostic contract (`write_file` / `exec` / `read_file`), so the
SAME `run_episode` harness runs unchanged against either — only the operational trade-offs differ.

  * SANDBOX=agentcore         -> AgentCoreSandbox      = the MANAGED Code Interpreter.
                                 AWS hosts the runtime; you build/deploy NO image. Zero ops, but you
                                 take the environment AWS gives you (stdlib Python, fixed user/CWD).
  * SANDBOX=agentcore-runtime -> AgentCoreRuntimeSandbox = YOUR OWN custom Runtime.
                                 Your OCI image (sandbox_server_agent.py in a BedrockAgentCoreApp)
                                 deployed via CreateAgentRuntime, reached over invoke_agent_runtime.
                                 You control deps/tools/user/filesystem/VPC — at the cost of an
                                 image build + deploy + a Runtime to manage.
  * SANDBOX=local             -> LocalSandbox (offline subprocess). Same contract, no AWS. Default
                                 here so the driver runs as an offline smoke with no credentials.

For each requested backend we run ONE identical episode (a tiny `ctf` reversing task by default,
driven by MockPolicy — no GPU/model) and print the reward. Same task + same harness ⇒ same reward
on every backend; the difference is purely operational, summarized in the table at the end.

Usage:
    # offline smoke (no AWS) — proves the contract with LocalSandbox:
    python3 compare_backends.py

    # live: compare the managed Code Interpreter against YOUR custom Runtime:
    AWS_REGION=us-west-2 AGENTCORE_SANDBOX_RUNTIME_ARN=<cpu-arn> \
        python3 compare_backends.py --backends agentcore agentcore-runtime

    # pick a different family / seed index:
    python3 compare_backends.py --family coding --seed 0
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agentcore_slime_demo.harness import run_episode
from agentcore_slime_demo.policy import MockPolicy
from agentcore_slime_demo.sandbox import make_sandbox
from agentcore_slime_demo.task import task_from_dict
from agentcore_slime_demo.task_families import FAMILY_CLASSES

# One-line operational contrast per AgentCore backend — printed after the runs so the reader sees
# WHY you'd pick one over the other, given they return the same reward.
_OPS = {
    "agentcore": {
        "kind": "Managed Code Interpreter",
        "image": "none — AWS-hosted",
        "deploy": "nothing to deploy (SDK start/stop a session)",
        "control": "fixed env: stdlib Python, unprivileged user, AWS-chosen CWD",
        "gpu": "no (CPU microVM only)",
    },
    "agentcore-runtime": {
        "kind": "Custom Runtime (your image)",
        "image": "acs-sandbox-cpu (deploy/Dockerfile.sandbox.cpu) -> ECR",
        "deploy": "build+push image, CreateAgentRuntime, invoke_agent_runtime",
        "control": "you own deps/tools/user/filesystem, /invocations handlers, VPC/IAM",
        "gpu": "CPU here; the SAME custom-Runtime path adds a GPU on Instances (see demo_gpu_farm LOCAL_GPU=0 + gpu_torch_smoke.py)",
    },
    "local": {
        "kind": "LocalSandbox (offline)",
        "image": "none — local subprocess",
        "deploy": "none (dev/CI only, NOT isolated)",
        "control": "the host env; no isolation",
        "gpu": "n/a",
    },
}


async def _run_one(backend: str, task) -> tuple[bool, float, int]:
    # Fresh sandbox per episode — exactly what the harness does in a real rollout.
    async with make_sandbox(backend) as sb:
        traj = await run_episode(MockPolicy(task.mock_attempts), sb, task)
    return traj.solved, traj.reward, traj.num_turns


def _make_task(family: str, seed_idx: int):
    cls = next((c for c in FAMILY_CLASSES if c.family == family), None)
    if cls is None:
        raise SystemExit(
            f"unknown family {family!r}; choices: {', '.join(sorted(c.family for c in FAMILY_CLASSES))}"
        )
    seeds = cls.seeds()
    seed = seeds[seed_idx % len(seeds)]
    # Round-trip through dataset serialization exactly like the trainer does.
    return task_from_dict({**seed.to_row_kwargs(), "mock_attempts": seed.mock_attempts})


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backends", nargs="+", default=["local"],
                    help="which sandbox backends to compare (default: local). "
                         "e.g. --backends agentcore agentcore-runtime")
    ap.add_argument("--family", default="ctf", help="task family (default: ctf)")
    ap.add_argument("--seed", type=int, default=0, help="seed index within the family (default: 0)")
    args = ap.parse_args(argv)

    task = _make_task(args.family, args.seed)
    print(f"Task: family={args.family} id={task.task_id}  (same task + harness for every backend)\n")

    results: dict[str, tuple[bool, float, int] | str] = {}
    for backend in args.backends:
        try:
            solved, reward, turns = asyncio.run(_run_one(backend, task))
            results[backend] = (solved, reward, turns)
            print(f"[{backend:18s}] solved={solved} reward={reward} turns={turns}")
        except Exception as e:  # noqa: BLE001
            results[backend] = f"{type(e).__name__}: {e}"
            print(f"[{backend:18s}] ERROR: {type(e).__name__}: {e}")

    # Same-reward assertion across the backends that actually ran.
    rewards = {b: r[1] for b, r in results.items() if isinstance(r, tuple)}
    if len(set(rewards.values())) <= 1:
        print(f"\n==> Identical reward across {len(rewards)} backend(s): the contract is the same.")
    else:
        print(f"\n!! rewards differ across backends: {rewards} — investigate (a real backend should match).")

    print("\nOperational contrast (what actually differs — the reward does not):")
    cols = ["kind", "image", "deploy", "control", "gpu"]
    for backend in args.backends:
        ops = _OPS.get(backend)
        if not ops:
            continue
        print(f"\n  {backend}:")
        for c in cols:
            print(f"    {c:8s}: {ops[c]}")

    bad = [b for b, r in results.items() if not isinstance(r, tuple) or not (r[0] and r[1] == 1.0)]
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
