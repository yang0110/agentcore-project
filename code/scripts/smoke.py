#!/usr/bin/env python3
"""Offline end-to-end smoke test — NO AWS, NO GPU, NO slime.

Runs the full multi-turn loop with MockPolicy (scripted buggy->fixed) + LocalSandbox over
every task in data/tasks.jsonl, prints the transcript, and exits non-zero if any task was
not solved. This is the "it runs" proof for the demo; the real path swaps MockPolicy ->
SGLangPolicy and LocalSandbox -> AgentCoreSandbox with no loop changes.

Usage:
    python code/scripts/smoke.py
    SANDBOX=agentcore AGENTCORE_REGION=us-west-2 python code/scripts/smoke.py   # real sandbox
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentcore_slime_demo.harness import run_episode
from agentcore_slime_demo.policy import MockPolicy
from agentcore_slime_demo.sandbox import make_sandbox
from agentcore_slime_demo.task import load_tasks

TASKS = Path(__file__).resolve().parents[1] / "agentcore_slime_demo" / "data" / "tasks.jsonl"


async def run_one(task, sandbox_kind: str) -> bool:
    policy = MockPolicy(task.mock_attempts)
    sandbox = make_sandbox(sandbox_kind)
    async with sandbox as sb:
        traj = await run_episode(policy, sb, task)

    print(f"\n=== task: {task.task_id}  (sandbox: {sb.sandbox_id}) ===")
    for i, turn in enumerate(traj.turns):
        code = "yes" if turn.submitted_code else "no"
        first_line = turn.observation.splitlines()[0] if turn.observation else ""
        print(f"  turn {i}: submitted_code={code} exit={turn.exit_code} | {first_line}")
    print(f"  --> solved={traj.solved} reward={traj.reward} turns={traj.num_turns}")
    return traj.solved


async def main() -> int:
    kind = os.environ.get("SANDBOX", "local")
    tasks = load_tasks(TASKS)
    results = [await run_one(t, kind) for t in tasks]
    solved, total = sum(results), len(results)
    print(f"\nSUMMARY: solved {solved}/{total} using sandbox={kind!r}")
    return 0 if solved == total else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
