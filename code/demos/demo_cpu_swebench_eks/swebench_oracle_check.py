#!/usr/bin/env python3
"""Oracle check for demo_cpu_swebench_eks: every pushed instance must be learnable ON THIS SUBSTRATE.

For each instance in swebench/image_by_task.json, two fresh gVisor pods from its image (EksPodSandbox —
gVisor is a different kernel than the one SWE-bench validated on, so this is the check that matters),
graded by the exact grader the training pool uses
(agentcore_slime_demo.swebench.grade_swebench):

  * gold — `git apply` the reference fix in /testbed, then grade  -> must be solved (reward reachable)
  * null — grade the untouched repo                             -> must NOT be solved (no free reward)

Writes swebench/oracle.csv (usable = gold_solved=1 and null_solved=0); `python3 fetch_swebench.py
--data-only` then keeps only usable instances in data/swebench_*.jsonl. Pods are deleted on exit.

    export EKS_CLUSTER=acs-swe-eks
    python3 swebench_oracle_check.py                  # all mapped instances
    EKS_RUNTIME_CLASS= python3 swebench_oracle_check.py   # same, under runc (A/B the gVisor kernel)
    SWE_TASKS=django__django-15127 python3 swebench_oracle_check.py
    ORACLE_CONC=8 python3 swebench_oracle_check.py
    python3 swebench_oracle_check.py --local          # docker-run the locally built wrapper images
                                                      # (SWE_NO_PUSH=1 deploy) via AGENTCORE_LOCAL_ENDPOINT
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from agentcore_slime_demo.agentcore_runtime_sandbox import AgentCoreRuntimeSandbox  # noqa: E402
from agentcore_slime_demo.eks_pod_sandbox import EksPodSandbox  # noqa: E402
from agentcore_slime_demo.swebench import REPO_DIR, grade_swebench, load_instance  # noqa: E402

OUT = HERE / "swebench"
OUT_CSV = OUT / "oracle.csv"
CONC = int(os.environ.get("ORACLE_CONC", "8"))
TIMEOUT = float(os.environ.get("SWE_TEST_TIMEOUT", "600"))
FIELDS = ["task", "gold_solved", "null_solved", "usable", "gold_f2p", "gold_p2p", "null_f2p", "null_p2p",
          "gold_apply_exit", "open_s", "gold_grade_s", "null_grade_s", "error"]


async def _one_mode(image: str | None, inst: dict, gold: bool) -> dict:
    """image = ECR URI -> a fresh gVisor pod; None -> the local docker container (--local)."""
    t0 = time.perf_counter()
    sb = EksPodSandbox(image=image, workdir=REPO_DIR) if image else AgentCoreRuntimeSandbox(workdir=REPO_DIR)
    async with sb:
        t_open = time.perf_counter() - t0
        apply_rc = ""
        if gold:
            await sb.write_file("/tmp/.acs_swe_gold.patch", inst["gold_patch"])
            apply_rc, _, e = await sb.exec(f"cd {REPO_DIR} && git apply -v /tmp/.acs_swe_gold.patch", timeout=120)
            if apply_rc:
                print(f"  [{inst['task']}] gold apply rc={apply_rc}: {e[-300:]}", flush=True)
        t1 = time.perf_counter()
        g = await grade_swebench(sb, inst, TIMEOUT)
        g.update(open_s=round(t_open, 1), grade_s=round(time.perf_counter() - t1, 1), apply_rc=apply_rc)
        return g


async def check(task: str, arn: str | None, sem: asyncio.Semaphore, run_mode=None) -> dict:
    """run_mode(inst, gold) -> grade dict; default = a fresh gVisor pod per mode (in parallel)."""
    inst = load_instance(task)
    row = {"task": task, "error": ""}
    async with sem:
        try:
            if run_mode is None:
                gold, null = await asyncio.gather(_one_mode(arn, inst, True), _one_mode(arn, inst, False))
            else:
                gold, null = await run_mode(inst, True), await run_mode(inst, False)
            row.update(gold_solved=int(gold["solved"]), null_solved=int(null["solved"]),
                       gold_f2p=f"{gold['f2p_passed']}/{len(inst['FAIL_TO_PASS'])}",
                       gold_p2p=f"{gold['p2p_passed']}/{len(inst['PASS_TO_PASS'])}",
                       null_f2p=f"{null['f2p_passed']}/{len(inst['FAIL_TO_PASS'])}",
                       null_p2p=f"{null['p2p_passed']}/{len(inst['PASS_TO_PASS'])}",
                       gold_apply_exit=gold["apply_rc"], open_s=max(gold["open_s"], null["open_s"]),
                       gold_grade_s=gold["grade_s"], null_grade_s=null["grade_s"])
            if not gold["solved"]:
                row["error"] = f"gold tail: {gold['grade_tail'][-240:]}".replace("\n", " | ")
        except Exception as e:  # noqa: BLE001 — record and move on
            row.update(gold_solved=0, null_solved=0, error=f"{type(e).__name__}: {e}"[:300])
    row["usable"] = int(row.get("gold_solved") == 1 and row.get("null_solved") == 0)
    print(f"[oracle] {task}: gold={row.get('gold_solved')} ({row.get('gold_f2p')} f2p, {row.get('gold_p2p')} p2p)"
          f" null={row.get('null_solved')} ({row.get('null_f2p')} f2p, {row.get('null_p2p')} p2p)"
          f" usable={row['usable']} open_s={row.get('open_s')} grade_s={row.get('gold_grade_s')} {row['error'][:200]}",
          flush=True)
    return row


def _local_check(task: str) -> dict:
    """One instance against `docker run` of its locally built wrapper image: a NEW container per mode
    (gold / null), mirroring one-pod-per-episode isolation on EKS."""
    image, port = f"acs-swe-local:{task.lower()}", 18480
    name = f"acs-swe-oracle-{os.getpid()}"

    async def run_mode(inst: dict, gold: bool) -> dict:
        subprocess.run(["docker", "run", "-d", "--rm", "--name", name, "--cap-drop=ALL", "-p", f"{port}:8080",
                        image], check=True, stdout=subprocess.DEVNULL)
        try:
            os.environ["AGENTCORE_LOCAL_ENDPOINT"] = f"http://127.0.0.1:{port}"
            for _ in range(60):
                if subprocess.run(["curl", "-sf", f"http://127.0.0.1:{port}/ping"],
                                  capture_output=True).returncode == 0:
                    break
                await asyncio.sleep(1)
            return await _one_mode(None, inst, gold)
        finally:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)

    return asyncio.run(check(task, None, asyncio.Semaphore(1), run_mode))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", action="store_true")
    a = ap.parse_args()
    only = [t for t in os.environ.get("SWE_TASKS", "").split(",") if t]
    if a.local:
        tasks = only or [e["task"] for e in json.loads((OUT / "manifest.json").read_text())["instances"]]
        rows = [_local_check(t) for t in tasks]
        out = OUT / "oracle_local.csv"
    else:
        pool = json.loads((OUT / "image_by_task.json").read_text())
        tasks = [t for t in (only or sorted(pool)) if t in pool]

        async def run_all():
            sem = asyncio.Semaphore(CONC)
            return await asyncio.gather(*(check(t, pool[t], sem) for t in tasks))

        rows = list(asyncio.run(run_all()))
        out = OUT_CSV
    old = {}
    if out.exists():
        with out.open() as f:
            old = {r["task"]: r for r in csv.DictReader(f)}
    old.update({r["task"]: r for r in rows})
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(old.values(), key=lambda r: r["task"]))
    n_ok = sum(r["usable"] for r in rows)
    print(f"\n# {n_ok}/{len(rows)} usable -> {out}")
    return 0 if n_ok == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
