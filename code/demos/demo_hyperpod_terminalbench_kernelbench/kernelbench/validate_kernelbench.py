#!/usr/bin/env python3
"""Oracle-check every KernelBench row on the plain-EC2 GPU pool -> kernelbench/oracle.csv (HOST, real AWS $).

For each row the REFERENCE itself is submitted as the solution (solution.py = the reference code +
`ModelNew = Model`) and graded by the SAME hidden grader a training episode uses
(mini_claude_episode._hidden_grade: setup re-seeded, checker under a random name, nonce'd PASSED line).
A row whose oracle fails can never be solved (OOM on a 24 GB GPU, a missing package, a grade timeout),
so build_kernelbench.py drops it and prints why. The Terminal-Bench analogue is tasks/oracle.csv.

    python3 kernelbench/validate_kernelbench.py                  # all rows of kernelbench_{train,eval}.jsonl
    KB_ONLY=kb_level_4_1_,kb_level_4_3_ python3 kernelbench/validate_kernelbench.py   # id prefixes
    KB_CONC=16 TB_G6_MAX_INSTANCES=16 ...                         # parallel episodes / instances

Then: python3 kernelbench/build_kernelbench.py (in slime_trainer) and bash 1_prepare_data.sh.
Needs deploy/g6_host.json (g6_host.py image + bake-ami). Every instance is terminated at the end.
"""
from __future__ import annotations

import asyncio
import csv
import json
import os
import re
import secrets
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEMO = HERE.parent
sys.path.insert(0, str(DEMO))
OUT = HERE / "oracle.csv"
STATE = DEMO / "deploy" / "g6_host.json"
FIELDS = ["task", "level", "status", "reason", "grade_s", "speedup", "instance_type", "ts"]


def _rows() -> list[dict]:
    out = []
    for name in ("kernelbench_train.jsonl", "kernelbench_eval.jsonl"):
        out += [json.loads(line) for line in open(HERE / name) if line.strip()]
    only = [p for p in os.environ.get("KB_ONLY", "").split(",") if p]
    return [r for r in out if not only or r["label"].startswith(tuple(only))]


def _reference(task) -> str:
    """The reference code from the prompt's ```python block, submitted as ModelNew."""
    m = re.search(r"```python\n(.*?)\n```", task.prompt, re.S)
    return m.group(1) + "\n\nModelNew = Model\n"


def _reason(out: str) -> str:
    lines = [ln.strip() for ln in (out or "").splitlines() if ln.strip()]
    for pat in ("OutOfMemoryError", "out of memory", "Error", "assert", "Timeout", "timed out"):
        hit = [ln for ln in lines if pat.lower() in ln.lower()]
        if hit:
            return hit[-1][:200]
    return (lines[-1] if lines else "no output")[:200]


async def _one(row: dict, region: str, sem: asyncio.Semaphore) -> dict:
    from agentcore_slime_demo.generate import _task_from_sample
    from agentcore_slime_demo.mini_claude_episode import _hidden_grade
    from agentcore_slime_demo.terminalbench_pool import new_sandbox

    class _S:
        metadata, prompt = row["metadata"], row["prompt"]

    task = _task_from_sample(_S)
    rec = {"task": row["label"], "level": row["label"].split("_")[2], "status": "fail", "reason": "",
           "grade_s": "", "speedup": "", "instance_type": "", "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    async with sem:
        sb = new_sandbox(f"gpuec2:{task.image}", region=region, workdir="",
                         session_id=f"kbo-{secrets.token_hex(16)}")
        try:
            await sb.__aenter__()
            rec["instance_type"] = (sb.open_reply.get("boot") or {}).get("instance_type", "")
            await sb.write_file("solution.py", _reference(task))
            t0 = time.perf_counter()
            code, out = await _hidden_grade(sb, task)
            rec["grade_s"] = round(time.perf_counter() - t0, 1)
            sp = re.search(r"SPEEDUP vs reference: ([0-9.]+)x", out or "")
            rec["speedup"] = sp.group(1) if sp else ""
            if code == 0:
                rec["status"] = "ok"
            else:
                rec["reason"] = f"exit {code}: {_reason(out)}"
                if code == 124 or "timed out" in (out or "").lower():
                    rec["status"] = "timeout"
        except Exception as e:  # noqa: BLE001 — an infra failure is recorded, not a verdict on the task
            rec.update(status="infra", reason=f"{type(e).__name__}: {e}"[:200])
        finally:
            try:
                await sb.__aexit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass
    print(f"{rec['status']:<8} {rec['task'][:60]:<60} {rec['grade_s']!s:>6}s {rec['instance_type']} {rec['reason'][:90]}",
          flush=True)
    return rec


async def _main() -> int:
    if not STATE.exists():
        sys.exit(f"{STATE} missing: python3 deploy/g6_host.py image && python3 deploy/g6_host.py bake-ami")
    os.environ["TB_G6_EC2"] = STATE.read_text()
    os.environ.setdefault("MINI_GRADE_TIMEOUT", "600")          # = run.sh
    os.environ.setdefault("TB_G6_MAX_INSTANCES", os.environ.get("KB_CONC", "16"))
    region = os.environ.get("AWS_REGION", "us-west-2")
    rows = _rows()
    sem = asyncio.Semaphore(int(os.environ.get("KB_CONC", "16")))
    print(f"# oracle-checking {len(rows)} KernelBench rows on the EC2 GPU pool", flush=True)
    recs = await asyncio.gather(*[_one(r, region, sem) for r in rows])
    old = {}
    if OUT.exists():               # a subset run (KB_ONLY) updates its rows, keeps the rest
        with OUT.open() as f:
            old = {r["task"]: r for r in csv.DictReader(f)}
    old.update({r["task"]: r for r in recs})
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(sorted(old.values(), key=lambda r: r["task"]))
    by = {}
    for r in recs:
        by.setdefault((r["level"], r["status"]), 0)
        by[(r["level"], r["status"])] += 1
    print("# " + ", ".join(f"level {lv} {st}={n}" for (lv, st), n in sorted(by.items())) + f" -> {OUT}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(_main()))
    finally:
        from agentcore_slime_demo.ec2_gpu_sandbox import _POOL, _terminate_ids

        if _POOL.all_ids():
            _terminate_ids(os.environ.get("AWS_REGION", "us-west-2"), _POOL.all_ids())
