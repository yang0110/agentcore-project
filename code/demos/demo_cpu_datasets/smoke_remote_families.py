#!/usr/bin/env python3
"""Remote-sandbox smoke for the demo_cpu_datasets task-family zoo.

Runs ONE episode per family through the REAL in-process `run_episode` harness against the REMOTE
AgentCore Runtime CPU sandbox (SANDBOX=agentcore-runtime), driven by MockPolicy (no GPU/model).
This proves the remote runtime executes each family's setup_files + artifact + checker and returns
the discriminating RLVR reward — the same path a real slime rollout takes, minus the policy.

    AGENTCORE_SANDBOX_RUNTIME_ARN=<cpu arn> AWS_REGION=us-west-2 SANDBOX=agentcore-runtime \
        python3 demos/demo_cpu_datasets/smoke_remote_families.py [family ...]
"""
from __future__ import annotations

import asyncio
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

from agentcore_slime_demo.harness import run_episode
from agentcore_slime_demo.policy import MockPolicy
from agentcore_slime_demo.sandbox import make_sandbox
from agentcore_slime_demo.task import task_from_dict
from agentcore_slime_demo.task_families import FAMILY_CLASSES


async def _one(task) -> tuple[bool, float, int]:
    # Fresh remote sandbox session per episode (as the harness does in a real rollout).
    async with make_sandbox("agentcore-runtime") as sb:
        traj = await run_episode(MockPolicy(task.mock_attempts), sb, task)
    return traj.solved, traj.reward, traj.num_turns


def main(argv: list[str]) -> int:
    want = set(argv) or None
    classes = [c for c in FAMILY_CLASSES if want is None or c.family in want]
    ok = True
    for cls in classes:
        seed = cls.seeds()[0]
        # Round-trip through dataset serialization + mock_attempts, exactly like the trainer.
        task = task_from_dict({**seed.to_row_kwargs(), "mock_attempts": seed.mock_attempts})
        try:
            solved, reward, turns = asyncio.run(_one(task))
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {cls.family:9s} {seed.task_id}: {type(e).__name__}: {e}")
            ok = False
            continue
        tag = "OK " if (solved and reward == 1.0 and turns >= 2) else "BAD"
        if tag == "BAD":
            ok = False
        print(f"[{tag}] {cls.family:9s} {seed.task_id}: solved={solved} reward={reward} turns={turns}")
    print("ALL REMOTE FAMILIES OK" if ok else "SOME REMOTE FAMILIES FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
