"""MBPP as the CPU-microVM (Demo A) RL dataset.

MBPP (Mostly Basic Python Problems) is the standard code-generation benchmark: each item
is a natural-language task + a reference solution + a `test_list` of `assert` statements.
It maps exactly onto our write -> run -> reward loop, and it is pure CPU — the right fit
for the serverless Code Interpreter microVM (the RL default; the policy's GPUs stay on
slime's cluster). GPU-required tasks live in kernel_task.py for the Demo B instance path.

Each MBPP example becomes a CodingTask with:
  * check_mode="assert" — the tests are bare-name asserts (e.g. `assert f(3) == 4`), run
    in a namespace seeded with the solution's globals + test_setup_code (see task.py).
  * prompt = the task text PLUS the tests, the standard MBPP prompting trick that pins the
    exact function name/signature the tests call (otherwise it is ambiguous).
  * mock_attempts (offline MockPolicy only) = [failing stub, reference solution], so the
    offline demo/tests still exercise a genuine multi-turn fail->fix loop without a policy.

Live RL ignores mock_attempts: the slime policy generates solution.py each turn.

Loading uses HF `datasets`. We default the HF cache to a writable project dir because the
shared ~/.cache/huggingface here is root-owned (verified). Network is required on first
load; use `dump_jsonl` to snapshot a subset for fully offline runs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .task import CodingTask

_DEFAULT_CACHE = Path(__file__).resolve().parents[1] / ".hf_cache"

# A stub that defines nothing -> every bare-name assert raises NameError -> turn 0 fails,
# forcing the fix on turn 1. Keeps the offline demo honestly multi-turn.
_FAILING_STUB = "# TODO: implement the function required by the tests\n"


def _ensure_cache() -> None:
    os.environ.setdefault("HF_HOME", str(_DEFAULT_CACHE))
    os.environ.setdefault("HF_DATASETS_CACHE", str(_DEFAULT_CACHE / "datasets"))
    Path(os.environ["HF_DATASETS_CACHE"]).mkdir(parents=True, exist_ok=True)


def _to_task(ex: dict) -> CodingTask:
    tests = list(ex.get("test_list") or [])
    setup = ex.get("test_setup_code") or ""
    text = (ex.get("text") or "").strip()
    # Standard MBPP prompting: show the tests so the agent knows the exact entry point.
    prompt = text + "\n\nYour solution.py must satisfy these tests:\n" + "\n".join(tests)
    reference = ex.get("code") or ""
    return CodingTask(
        task_id=f"mbpp_{ex.get('task_id')}",
        prompt=prompt,
        checks=tests,
        check_mode="assert",
        test_setup=setup,
        max_turns=4,
        mock_attempts=[_FAILING_STUB, reference],
    )


def load_mbpp(split: str = "train", limit: int | None = None, offset: int = 0) -> list[CodingTask]:
    """Load MBPP (full config) into CodingTasks. split ∈ {train,test,validation,prompt}."""
    _ensure_cache()
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/mbpp", "full", split=split)
    n = len(ds)
    end = n if limit is None else min(n, offset + limit)
    return [_to_task(ds[i]) for i in range(offset, end)]


def dump_jsonl(path: str | Path, split: str = "train", limit: int | None = None, offset: int = 0) -> int:
    """Snapshot a subset to JSONL (loadable offline via task.load_tasks). Returns count."""
    tasks = load_mbpp(split=split, limit=limit, offset=offset)
    lines = []
    for t in tasks:
        lines.append(json.dumps({
            "task_id": t.task_id, "prompt": t.prompt, "checks": t.checks,
            "check_mode": t.check_mode, "test_setup": t.test_setup,
            "max_turns": t.max_turns, "mock_attempts": t.mock_attempts,
        }))
    Path(path).write_text("\n".join(lines) + "\n")
    return len(tasks)
