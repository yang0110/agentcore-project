#!/usr/bin/env python3
"""Offline fault-recovery demo — proves rollouts survive sandbox failures. No AWS/GPU/slime.

Injects three failure modes with FlakySandbox and shows resilient_episode recovers (or aborts
cleanly, never crashing):

  1. transient boot failures  -> sandbox recreated, episode still solves
  2. mid-episode session death -> fresh session, episode restarted, still solves
  3. fatal / unrecoverable      -> clean ABORTED trajectory (reward 0), no exception raised

This mirrors the recovery machinery in slime's shipped stack (boot_agent_sandbox retries +
_rpc_retry + _abort_result) so training keeps going when the environment flakes.

Usage: python3 code/scripts/recovery_smoke.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentcore_slime_demo.policy import MockPolicy
from agentcore_slime_demo.recovery import FlakySandbox, resilient_episode
from agentcore_slime_demo.sandbox import LocalSandbox
from agentcore_slime_demo.task import load_tasks

TASKS = Path(__file__).resolve().parents[1] / "agentcore_slime_demo" / "data" / "tasks.jsonl"


async def scenario(name: str, task, **flaky_kw) -> None:
    FlakySandbox.reset_counters()
    traj = await resilient_episode(
        task,
        make_policy=lambda: MockPolicy(task.mock_attempts),
        make_sandbox=lambda: FlakySandbox(LocalSandbox(), **flaky_kw),
        max_boot_retries=4,
        backoff_base_sec=0.01,  # keep the demo fast
    )
    print(f"[{name:24}] status={traj.status:9} solved={traj.solved} "
          f"reward={traj.reward} boots={traj.boot_attempts}"
          + (f" reason={traj.abort_reason}" if traj.status == 'aborted' else ""))


async def main() -> int:
    task = load_tasks(TASKS)[0]  # two_sum
    print(f"task = {task.task_id}\n")
    await scenario("healthy (baseline)", task)
    await scenario("2 transient boot fails", task, fail_boots=2)
    await scenario("mid-episode death", task, fail_exec_after=1)
    await scenario("fatal (unrecoverable)", task, fail_boots=1, fatal=True)
    print("\n(expected: first 3 solved via recovery; last cleanly aborted, no crash)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
