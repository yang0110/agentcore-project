#!/usr/bin/env python3
"""Two demo cases: CPU-based vs GPU-supported AgentCore Runtime sandbox.

Runs both cases end-to-end through the SAME multi-turn harness (write -> run checker ->
read failure -> fix), differing only in the sandbox *profile* and the *task*:

  Case 1  CPU_SANDBOX  — code-repair tasks (two_sum, factorial). Pure CPU. The RL default:
                         tool execution is cheap shell/IO; the policy's GPUs live on slime.
  Case 2  GPU_SANDBOX  — a task whose checker only passes if a real CUDA GPU is visible
                         INSIDE the sandbox. Proves an AgentCore Runtime instance backed by
                         a GPU capacity provider (customer EC2, e.g. g6.xlarge).

Offline (default) both run on LocalSandbox: case 1 needs nothing; case 2 uses whatever torch
+ CUDA is on this box. On AWS, swap in AgentCoreSandbox with the matching profile (SANDBOX=
agentcore). The harness never knows which backend/profile it holds — that is the point.

Usage:
    python3 code/scripts/demo_cases.py                 # both cases, offline
    python3 code/scripts/demo_cases.py --case cpu      # just case 1
    python3 code/scripts/demo_cases.py --case gpu      # just case 2
    SANDBOX=agentcore python3 code/scripts/demo_cases.py   # real AgentCore backend
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentcore_slime_demo.harness import run_episode
from agentcore_slime_demo.policy import MockPolicy
from agentcore_slime_demo.profiles import CPU_SANDBOX, GPU_SANDBOX, SandboxProfile
from agentcore_slime_demo.sandbox import make_sandbox
from agentcore_slime_demo.task import load_tasks

_DATA = Path(__file__).resolve().parents[1] / "agentcore_slime_demo" / "data"
CPU_TASKS = _DATA / "tasks.jsonl"
GPU_TASKS = _DATA / "gpu_tasks.jsonl"


def probe_local_gpu() -> str:
    """Best-effort report of the GPU the offline LocalSandbox would actually use."""
    names = []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            names = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    if not names:
        return "no GPU detected on this host"
    return f"{len(names)}x {names[0]}" + (" (+more)" if len(set(names)) > 1 else "")


async def run_case(profile: SandboxProfile, tasks_path: Path, sandbox_kind: str) -> tuple[int, int]:
    print(f"\n{'=' * 78}\nCASE: {profile.name}\n{'=' * 78}")
    print(profile.summary())
    print("    provision      :")
    for ln in profile.provision_cli:
        print(f"        {ln}")

    if profile.needs_gpu and sandbox_kind == "local":
        print(f"\n    [offline] LocalSandbox GPU visibility: {probe_local_gpu()}")

    tasks = load_tasks(tasks_path)
    solved = 0
    for task in tasks:
        policy = MockPolicy(task.mock_attempts)
        async with make_sandbox(sandbox_kind) as sb:
            traj = await run_episode(policy, sb, task)
        print(f"\n    task {task.task_id}  (sandbox {sb.sandbox_id})")
        for i, turn in enumerate(traj.turns):
            first = turn.observation.splitlines()[0] if turn.observation else ""
            print(f"      turn {i}: submitted={turn.submitted_code} exit={turn.exit_code} | {first}")
        print(f"      --> solved={traj.solved} reward={traj.reward} turns={traj.num_turns}")
        solved += int(traj.solved)
    return solved, len(tasks)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=["cpu", "gpu", "both"], default="both")
    args = ap.parse_args()
    kind = os.environ.get("SANDBOX", "local")

    plan = []
    if args.case in ("cpu", "both"):
        plan.append((CPU_SANDBOX, CPU_TASKS))
    if args.case in ("gpu", "both"):
        plan.append((GPU_SANDBOX, GPU_TASKS))

    totals = []
    for profile, path in plan:
        totals.append(await run_case(profile, path, kind))

    print(f"\n{'=' * 78}\nSUMMARY (sandbox backend = {kind!r})")
    all_ok = True
    for (profile, _), (solved, total) in zip(plan, totals):
        ok = solved == total
        all_ok &= ok
        print(f"  {profile.name:<12} solved {solved}/{total}  {'OK' if ok else 'FAILED'}")

    # A GPU case that solves 0/1 offline usually means no CUDA on this host — call it out
    # rather than failing silently, since it is expected on a CPU-only dev box.
    if args.case in ("gpu", "both") and kind == "local" and not shutil.which("nvidia-smi"):
        print("  note: GPU case needs a CUDA GPU + torch in the offline sandbox; none found here.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
