#!/usr/bin/env python3
"""Validate every deployed Terminal-Bench task ON THIS SUBSTRATE before training on it.

A task is only a useful RL task here if, on its AgentCore Runtime (arm64 microVM, root with an empty
capability set, fresh session):

  * null   — grading the UNTOUCHED environment FAILS   (else reward is 1 for doing nothing: no signal)
  * oracle — running the task's reference solution, then grading, PASSES
             (else reward is structurally 0 on this substrate: the task cannot be learned here)

Each check runs on its own fresh session, through the exact grader the training pool uses
(terminalbench_pool.grade_episode: tests injected at /tests after the fact, TEST_DIR=/tests,
pytest -rA summary parsed). Writes tasks/oracle.csv; build_terminalbench_dataset.py then trains only
on tasks with oracle=1 and null=0 (`usable`).

    export AGENTCORE_REGION=us-west-2
    python3 validate_oracle.py                   # every task in tasks/arn_by_image.json
    ORACLE_CONC=32 python3 validate_oracle.py
    TB_TASKS=a,b python3 validate_oracle.py       # subset
"""
from __future__ import annotations

import asyncio
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from agentcore_slime_demo.agentcore_runtime_sandbox import AgentCoreRuntimeSandbox  # noqa: E402
from agentcore_slime_demo.terminalbench_pool import grade_episode  # noqa: E402

TASKS_DIR = HERE / "tasks"
OUT_CSV = TASKS_DIR / "oracle.csv"
CONC = int(os.environ.get("ORACLE_CONC", "24"))
MAX_EXEC_S = 840        # stay under AgentCore's 15-min request timeout


def _yaml_float(task: str, key: str, default: float) -> float:
    ty = TASKS_DIR / task / "task.yaml"
    m = re.search(rf"(?m)^{key}:\s*([0-9.]+)", ty.read_text()) if ty.exists() else None
    return float(m.group(1)) if m else default


def _solution_script(task: str) -> str | None:
    """The reference solution as one bash script (solution.sh, or solution.yaml's command list)."""
    tdir = TASKS_DIR / task
    if (tdir / "solution.sh").is_file():
        return (tdir / "solution.sh").read_text()
    if (tdir / "solution.yaml").is_file():
        import yaml

        steps = yaml.safe_load((tdir / "solution.yaml").read_text()) or []
        cmds = [s["command"] if isinstance(s, dict) else str(s) for s in steps]
        return "set -e\n" + "\n".join(cmds) + "\n"
    return None


async def _session(arn: str, region: str):
    sb = AgentCoreRuntimeSandbox(runtime_arn=arn, region=region, workdir="")
    await sb.__aenter__()
    boot = (sb.open_reply or {}).get("boot") or {}
    if boot.get("workdir"):
        sb._workdir = boot["workdir"]
    return sb


async def _check(task: str, arn: str, region: str, sem: asyncio.Semaphore) -> dict:
    row = {"task": task, "null": "", "oracle": "", "usable": 0, "oracle_s": "", "grade_s": "",
           "tests": "", "note": ""}
    test_t = _yaml_float(task, "max_test_timeout_sec", 180.0)
    agent_t = min(_yaml_float(task, "max_agent_timeout_sec", 900.0), MAX_EXEC_S)
    async with sem:
        # --- null: grade the untouched environment
        sb = None
        try:
            sb = await _session(arn, region)
            g = await grade_episode(sb, task, str(TASKS_DIR), test_t)
            row["null"] = int(g["solved"])
        except Exception as e:  # noqa: BLE001
            row["note"] += f"null:{type(e).__name__}: {e}"[:150] + " "
        finally:
            if sb:
                await sb.__aexit__(None, None, None)
        # --- oracle: reference solution, then grade
        script = _solution_script(task)
        if script is None:
            row["note"] += "no reference solution"
            return row
        sb = None
        try:
            sb = await _session(arn, region)
            await sb.write_file("/tmp/.acs_solution.sh", script)
            t0 = time.perf_counter()
            code, out, err = await sb.exec("bash /tmp/.acs_solution.sh", timeout=int(agent_t))
            row["oracle_s"] = round(time.perf_counter() - t0, 1)
            if code == 124:
                row["note"] += f"solution timed out at {agent_t:.0f}s "
            t0 = time.perf_counter()
            g = await grade_episode(sb, task, str(TASKS_DIR), test_t)
            row["grade_s"] = round(time.perf_counter() - t0, 1)
            row["oracle"] = int(g["solved"])
            row["tests"] = f"{g['tests_passed']}/{g['tests_total']}"
            if not g["solved"]:
                row["note"] += "grade_tail: " + g["grade_tail"].replace("\n", " | ")[-240:]
        except Exception as e:  # noqa: BLE001
            row["note"] += f"oracle:{type(e).__name__}: {e}"[:200]
        finally:
            if sb:
                await sb.__aexit__(None, None, None)
    row["usable"] = int(row["oracle"] == 1 and row["null"] == 0)
    print(f"[oracle] {task}: null={row['null']} oracle={row['oracle']} usable={row['usable']} "
          f"tests={row['tests']} {row['note'][:100]}", flush=True)
    return row


async def _main() -> int:
    region = os.environ.get("AGENTCORE_REGION", "us-west-2")
    pool = json.loads((TASKS_DIR / "arn_by_image.json").read_text())
    only = {t for t in os.environ.get("TB_TASKS", "").split(",") if t}
    if only:
        pool = {k: v for k, v in pool.items() if k in only}
    sem = asyncio.Semaphore(CONC)
    t0 = time.perf_counter()
    rows = await asyncio.gather(*[_check(t, a, region, sem) for t, a in pool.items()])

    # Merge with earlier results for tasks not re-checked this time.
    prev = {}
    if OUT_CSV.exists() and only:
        with OUT_CSV.open() as f:
            prev = {r["task"]: r for r in csv.DictReader(f)}
    prev.update({r["task"]: r for r in rows})
    fields = ["task", "usable", "null", "oracle", "tests", "oracle_s", "grade_s", "note"]
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(prev.values(), key=lambda r: r["task"]))

    n = len(rows)
    print(f"\n=== oracle validation ({n} tasks, {time.perf_counter() - t0:.0f}s) ===")
    print(f"  usable (oracle passes, null fails): {sum(r['usable'] for r in rows)}")
    print(f"  oracle fails on this substrate:     {sum(1 for r in rows if r['oracle'] == 0)}")
    print(f"  null passes (no signal):            {sum(1 for r in rows if r['null'] == 1)}")
    print(f"  errored / no solution:              {sum(1 for r in rows if r['oracle'] == '')}")
    print(f"  -> {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
