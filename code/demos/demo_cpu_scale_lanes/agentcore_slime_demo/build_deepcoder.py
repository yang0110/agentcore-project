"""Build the DeepCoder (primeintellect) dataset for demo_cpu_scale_lanes' 256-sandbox plan, with an oracle pass.

Source: agentica-org/DeepCoder-Preview-Dataset, subset `primeintellect` (stdin/stdout competitive
programming problems, each with gold solutions and hidden tests). The oracle pass runs one gold
Python solution over EVERY hidden test, exactly as the sandbox checker will, and keeps a problem
only if the gold solution passes all of them. It also records how long the full suite takes — the
sandbox CPU one final check costs — which is the number this demo is sized by.

Run inside agentcore_p4 (network needed once to pull HF rows):
    python3 build_deepcoder.py --probe 400                    # measure only -> data/deepcoder_probe.csv
    python3 build_deepcoder.py --train 512 --eval 64          # build data/deepcoder_{train,eval}.jsonl

Row schema matches every other demo: {"prompt", "label", "metadata": {"task": {family, task_id,
prompt, max_turns, spec}}}; `spec` carries the tests, so task.task_from_dict rebuilds a DeepCoderTask.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

DEMO = Path(__file__).resolve().parents[1]
PER_TEST_TIMEOUT_S = 6.0

_FENCE = re.compile(r"```(?:python|py|python3)?[^\n]*\n(.*?)```", re.DOTALL)


def _gold_python(solutions) -> str | None:
    """First gold solution that is plain Python (some rows mix in C++ or fenced blocks)."""
    if isinstance(solutions, str):
        solutions = json.loads(solutions)
    for s in solutions or []:
        m = _FENCE.findall(s)
        code = m[-1] if m else s
        if "#include" in code or "public static void" in code:
            continue
        return code
    return None


def _tests(raw) -> list[dict]:
    tests = json.loads(raw) if isinstance(raw, str) else raw
    return [t for t in tests if t.get("type", "stdin_stdout") == "stdin_stdout"
            and isinstance(t.get("input"), str) and isinstance(t.get("output"), str)]


def outputs_match(got: str, want: str) -> bool:
    """Token-wise compare, the usual competitive-programming judge (whitespace-insensitive)."""
    return got.split() == want.split()


def run_suite(code: str, tests: list[dict], timeout: float = PER_TEST_TIMEOUT_S) -> dict:
    """Run `code` on every test; the same loop the sandbox checker runs (see DeepCoderTask)."""
    with tempfile.TemporaryDirectory() as d:
        sol = Path(d, "solution.py")
        sol.write_text(code)
        t0, passed, worst = time.perf_counter(), 0, 0.0
        for t in tests:
            s = time.perf_counter()
            try:
                p = subprocess.run([sys.executable, str(sol)], input=t["input"], capture_output=True,
                                   text=True, timeout=timeout, cwd=d)
                ok = p.returncode == 0 and outputs_match(p.stdout, t["output"])
            except subprocess.TimeoutExpired:
                ok = False
            worst = max(worst, time.perf_counter() - s)
            passed += ok
        return {"passed": passed, "n": len(tests), "suite_s": time.perf_counter() - t0, "max_test_s": worst}


def _probe_one(args):
    idx, problem, solutions, raw_tests = args
    code, tests = _gold_python(solutions), _tests(raw_tests)
    if not code or not tests:
        return {"idx": idx, "ok": 0, "n": len(tests), "passed": 0, "suite_s": 0.0, "max_test_s": 0.0,
                "reason": "no_python_gold" if not code else "no_stdin_tests"}
    r = run_suite(code, tests)
    return {"idx": idx, "ok": int(r["passed"] == r["n"]), **r, "reason": ""}


def load(split="train"):
    os.environ.setdefault("HF_HOME", str(DEMO / ".hf_cache"))
    from datasets import load_dataset
    return load_dataset("agentica-org/DeepCoder-Preview-Dataset", "primeintellect", split=split)


def probe(ds, indices, workers):
    jobs = [(i, None, ds[i]["solutions"], ds[i]["tests"]) for i in indices]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_probe_one, jobs, chunksize=1))


PROMPT_SUFFIX = (
    "\n\nWrite a complete Python 3 program in solution.py that reads from standard input and writes "
    "to standard output. `python3 run_checks.py` runs it on the sample tests; the final grade runs "
    "it on the full hidden test suite."
)


def _row(i, ex, n_public, max_turns):
    tests = _tests(ex["tests"])
    problem = ex["problem"].replace("Solve the following coding problem using the programming "
                                    "language python:\n\n", "").strip()
    task_id = f"deepcoder_pi_{i}"
    spec = {"public_tests": tests[:n_public], "hidden_tests": tests, "timeout_s": PER_TEST_TIMEOUT_S}
    task = {"family": "deepcoder", "task_id": task_id, "prompt": problem + PROMPT_SUFFIX,
            "max_turns": max_turns, "spec": spec}
    return {"prompt": task["prompt"], "label": task_id, "metadata": {"task": task}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=int, default=0, help="measure N random problems, write the CSV, exit")
    ap.add_argument("--train", type=int, default=512)
    ap.add_argument("--eval", type=int, default=64)
    ap.add_argument("--min-suite-s", type=float, default=0.0,
                    help="keep only problems whose gold suite takes at least this long")
    ap.add_argument("--max-suite-s", type=float, default=90.0)
    ap.add_argument("--public", type=int, default=2, help="tests the agent may run itself")
    ap.add_argument("--max-turns", type=int, default=4)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds = load()
    order = list(range(len(ds)))
    random.Random(args.seed).shuffle(order)
    out = DEMO / "data"
    out.mkdir(exist_ok=True)

    if args.probe:
        res = probe(ds, order[: args.probe], args.workers)
        with (out / "deepcoder_probe.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(res[0]))
            w.writeheader()
            w.writerows(res)
        ok = sorted(r["suite_s"] for r in res if r["ok"])
        q = lambda p: ok[min(len(ok) - 1, int(p * len(ok)))] if ok else 0.0  # noqa: E731
        print(f"probed {len(res)}: oracle-ok {len(ok)} ({100 * len(ok) / len(res):.0f}%)  "
              f"suite_s p10 {q(.1):.2f} p50 {q(.5):.2f} p90 {q(.9):.2f} max {q(1):.2f}  "
              f"tests p50 {sorted(r['n'] for r in res)[len(res) // 2]}")
        return 0

    need = args.train + args.eval
    keep, pos = [], 0
    while len(keep) < need and pos < len(order):
        batch = order[pos: pos + 4 * args.workers]
        pos += len(batch)
        for r in probe(ds, batch, args.workers):
            if r["ok"] and args.min_suite_s <= r["suite_s"] <= args.max_suite_s:
                keep.append(r)
        print(f"  scanned {pos}, kept {len(keep)}/{need}", flush=True)
    keep = keep[:need]
    splits = {"train": keep[: args.train], "eval": keep[args.train:]}
    for name, rows in splits.items():
        with (out / f"deepcoder_{name}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(_row(r["idx"], ds[r["idx"]], args.public, args.max_turns)) + "\n")
        s = sorted(r["suite_s"] for r in rows)
        print(f"wrote {len(rows)} -> data/deepcoder_{name}.jsonl  (gold suite_s p50 {s[len(s) // 2]:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
