#!/usr/bin/env python3
"""Offline checker proof for THIS demo's dataset: the reference solution PASSES the hidden checker,
a wrong one FAILS, and the training-row round-trip is lossless. Stdlib-only (no GPU; one-time HF pull
to load the rows). Run inside the agentcore_p4 container:
    python3 verify_data.py
"""
import os, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agentcore_slime_demo.real_datasets import load_humaneval
from agentcore_slime_demo.build_real_datasets import _row
from agentcore_slime_demo.task import task_from_dict, CodingTask


def run_checker(task, submission):
    with tempfile.TemporaryDirectory() as d:
        for name, txt in task.setup_files().items():
            fp = os.path.join(d, name)
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            open(fp, "w").write(txt)
        open(os.path.join(d, "run_checks.py"), "w").write(task.runner_source())
        open(os.path.join(d, task.artifact_filename), "w").write(submission)
        p = subprocess.run(["python3", "run_checks.py"], cwd=d, capture_output=True, text=True)
        return p.returncode


tasks = load_humaneval(split="test", limit=15)
good = bad = 0
for t in tasks:
    mt = _row(t, None)["metadata"]["task"]                      # round-trip through the training row
    rt = task_from_dict({**mt}) if "family" in mt else CodingTask(**mt)
    ok = run_checker(rt, t.mock_attempts[-1]) == 0 and run_checker(rt, t.mock_attempts[0]) != 0
    good += ok
    bad += (not ok)
print(f"{good}/{good + bad} tasks: reference PASSES and wrong FAILS (checker is correct)")
sys.exit(0 if bad == 0 else 1)
