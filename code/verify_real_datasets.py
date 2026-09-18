import json, subprocess, sys, tempfile, os
sys.path.insert(0, "/work/code")
from agentcore_slime_demo.real_datasets import (
    load_gsm8k, load_math500, load_humaneval, load_deepscaler, load_hotpotqa, load_xlam,
    load_mcp, load_sql)
from agentcore_slime_demo.build_real_datasets import _row
from agentcore_slime_demo.task import task_from_dict

def run_checker(task, submission):
    with tempfile.TemporaryDirectory() as d:
        # write extra setup files (none for these), the checker, and the submission
        for name, txt in task.setup_files().items():
            fp = os.path.join(d, name)
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            open(fp, "w").write(txt)
        open(os.path.join(d, "run_checks.py"), "w").write(task.runner_source())
        open(os.path.join(d, task.artifact_filename), "w").write(submission)
        p = subprocess.run(["python3", "run_checks.py"], cwd=d, capture_output=True, text=True)
        return p.returncode, (p.stdout + p.stderr).strip().splitlines()[-1:] 

def check_family(name, tasks):
    print(f"\n===== {name}: {len(tasks)} tasks =====")
    good = bad = 0
    for t in tasks:
        # round-trip through the training row to prove serialization is lossless
        row = _row(t, None)
        t2 = task_from_dict({**row["metadata"]["task"]}) if "family" in row["metadata"]["task"] else None
        rt = t2 if t2 is not None else t  # coding rebuilds via CodingTask below
        if t2 is None:
            from agentcore_slime_demo.task import CodingTask
            rt = CodingTask(**row["metadata"]["task"])
        wrong, ref = t.mock_attempts[0], t.mock_attempts[-1]
        rc_ref, tail_ref = run_checker(rt, ref)
        rc_wrong, tail_wrong = run_checker(rt, wrong)
        ok = (rc_ref == 0 and rc_wrong != 0)
        good += ok; bad += (not ok)
        if not ok:
            print(f"  BAD {t.task_id}: ref_rc={rc_ref}{tail_ref} wrong_rc={rc_wrong}{tail_wrong}")
    print(f"  -> {good}/{good+bad} tasks: reference PASSES and wrong FAILS (checker is correct)")

check_family("gsm8k",      load_gsm8k(split="train", limit=15))
check_family("math500",    load_math500(split="test", limit=15))
check_family("humaneval",  load_humaneval(split="test", limit=15))
check_family("deepscaler", load_deepscaler(limit=15))
check_family("hotpotqa",   load_hotpotqa(split="validation", limit=15))
check_family("xlam",       load_xlam(limit=15))
check_family("mcp",        load_mcp(limit=15))
check_family("sql",        load_sql(split="train", limit=15))
