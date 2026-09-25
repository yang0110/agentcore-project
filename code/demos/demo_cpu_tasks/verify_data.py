#!/usr/bin/env python3
"""Offline checker proof for one (or every) real dataset of this demo: the reference solution PASSES
the hidden checker, a wrong one FAILS, and the training-row round-trip is lossless. Stdlib-only (no
GPU; one-time HF pull to load the rows). Run inside the agentcore_p4 container:
    python3 verify_data.py                 # every real dataset
    python3 verify_data.py gsm8k xlam      # just these
"""
import os, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agentcore_slime_demo import real_datasets as rd
from agentcore_slime_demo.build_real_datasets import _row
from agentcore_slime_demo.task import task_from_dict, CodingTask

# DATASET label -> 15 rows from the same split prepare_data.sh trains on.
LOADERS = {
    "gsm8k":      lambda: rd.load_gsm8k(split="train", limit=15),
    "math500":    lambda: rd.load_math500(split="test", limit=15),
    "deepscaler": lambda: rd.load_deepscaler(limit=15),
    "humaneval":  lambda: rd.load_humaneval(split="test", limit=15),
    "gretelsql":  lambda: rd.load_sql(split="train", limit=15),
    "hotpotqa":   lambda: rd.load_hotpotqa(split="validation", limit=15),
    "xlam":       lambda: rd.load_xlam(limit=15),
    "mcp":        lambda: rd.load_mcp(limit=15),
}


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


want = sys.argv[1:] or list(LOADERS)
unknown = [w for w in want if w not in LOADERS]
if unknown:
    sys.exit(f"unknown dataset(s) {unknown}; one of {list(LOADERS)}")
failed = 0
for name in want:
    good = bad = 0
    for t in LOADERS[name]():
        mt = _row(t, None)["metadata"]["task"]                  # round-trip through the training row
        rt = task_from_dict({**mt}) if "family" in mt else CodingTask(**mt)
        ok = run_checker(rt, t.mock_attempts[-1]) == 0 and run_checker(rt, t.mock_attempts[0]) != 0
        good += ok
        bad += (not ok)
    print(f"{name}: {good}/{good + bad} tasks: reference PASSES and wrong FAILS (checker is correct)")
    failed += bad
sys.exit(0 if failed == 0 else 1)
