"""The multi-turn task: iterative code-repair.

One task = a problem statement + a hidden checker. The agent must write `solution.py`
so the checker passes. Each turn it may resubmit; the checker's failure output is fed
back as the next observation. This is genuinely multi-turn: the reward signal (did the
checker pass) only exists *after running code in the sandbox*, and the agent typically
needs several turns (write → run → read error → fix).

Reward is deterministic and comes straight from the sandbox exit code — exactly the RL
reward primitive described in docs/sandbox-comparison.md §5 and used by slime's coding
example (git_diff → run_evaluation). No LLM judge, no heuristics.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

_CODE_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

# Written into the sandbox next to solution.py. Imports the agent's module, runs the
# task's checks, prints a machine-readable summary, and sets the process exit code.
#
# Two check modes:
#   * "expr"   — each check is a truthy Python expression over `solution.<name>` (our
#                synthetic tasks). Evaluated with {"solution": module}.
#   * "assert" — each check is an assert statement referencing BARE names (MBPP style,
#                e.g. `assert f(3) == 4`). Executed in a namespace seeded with the
#                solution module's globals + the task's setup code, so the function names
#                the tests call resolve. A raised AssertionError/Exception = fail.
_RUNNER_TEMPLATE = """\
import importlib.util, sys, traceback

# Never cache bytecode: across turns the agent rewrites solution.py, and an edit of the
# SAME byte-length within the same wall-clock second would otherwise let Python reuse the
# STALE __pycache__/*.pyc (its validity check is only (mtime, size)) — silently re-running
# the previous turn's code and reporting a wrong reward. Setting this before any import of
# solution means no .pyc is ever written, so no turn can read a stale one.
sys.dont_write_bytecode = True

def _load():
    spec = importlib.util.spec_from_file_location("solution", "solution.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_mode = {mode!r}
_checks = {checks!r}
_setup = {setup!r}

try:
    solution = _load()
except Exception:
    print("IMPORT_ERROR"); traceback.print_exc(); sys.exit(1)

_passed = 0
if _mode == "assert":
    _ns = dict(vars(solution))
    try:
        if _setup:
            exec(_setup, _ns)
    except Exception as e:
        print(f"SETUP_ERROR: {{type(e).__name__}}: {{e}}"); sys.exit(1)
    for i, stmt in enumerate(_checks):
        try:
            exec(stmt, _ns)
            _passed += 1
        except Exception as e:
            print(f"FAIL[{{i}}]: {{stmt}}  -> {{type(e).__name__}}: {{e}}")
else:
    for i, expr in enumerate(_checks):
        try:
            ok = eval(expr, {{"solution": solution}})
            if ok:
                _passed += 1
            else:
                print(f"FAIL[{{i}}]: {{expr}}  -> falsy result")
        except Exception as e:
            print(f"ERROR[{{i}}]: {{expr}}  -> {{type(e).__name__}}: {{e}}")

print(f"PASSED {{_passed}}/{{len(_checks)}}")
sys.exit(0 if _passed == len(_checks) else 1)
"""


@dataclass
class CodingTask:
    task_id: str
    prompt: str                       # human-readable problem statement
    checks: list[str]                 # expr (truthy) or assert stmts, per check_mode; all must pass
    starter_code: str = ""            # optional scaffold shown to the agent
    max_turns: int = 4
    check_mode: str = "expr"          # "expr" | "assert" (MBPP-style bare-name asserts)
    test_setup: str = ""              # setup code run before assert checks (MBPP test_setup_code)
    # Only used by MockPolicy for deterministic offline runs; ignored by real policies.
    mock_attempts: list[str] = field(default_factory=list)

    # ---- task-family interface (see task_families.py) ------------------------
    # The harness is family-agnostic: it writes `setup_files()` + the checker, asks the
    # policy for a submission, extracts it with `extract_submission`, writes it to
    # `artifact_filename`, runs `check_command`, and turns the exit code into a reward.
    # CodingTask is the "coding" family (MBPP/HumanEval-style); the other families in
    # task_families.py subclass EpisodicTask and override these same hooks.
    family: ClassVar[str] = "coding"
    artifact_filename: ClassVar[str] = "solution.py"
    check_command: ClassVar[str] = "python3 run_checks.py"
    fence_lang: ClassVar[str] = "python"
    submission_hint: ClassVar[str] = (
        "Reply with the full solution.py in one ```python code block."
    )

    def setup_files(self) -> dict[str, str]:
        """Extra files written into the sandbox once, before the loop (none for coding)."""
        return {}

    def extract_submission(self, text: str) -> str | None:
        """Family-agnostic hook the harness calls to pull the turn's artifact."""
        return self.extract_code(text)

    @classmethod
    def from_dict(cls, d: dict) -> "CodingTask":
        return cls(**{k: v for k, v in d.items() if k != "family"})

    # ---- prompts -------------------------------------------------------------
    def system_prompt(self) -> str:
        return (
            "You are a coding agent working inside a sandbox. Write a complete Python "
            "file named solution.py that satisfies the requirements. Reply with ONE fenced "
            "```python code block containing the full file — no prose outside it. If the "
            "checker reports failures, fix the code and resubmit the full file."
        )

    def initial_user_prompt(self) -> str:
        parts = [self.prompt]
        if self.starter_code:
            parts.append(f"\nStarter code:\n```python\n{self.starter_code}\n```")
        return "\n".join(parts)

    # ---- turn helpers --------------------------------------------------------
    @staticmethod
    def extract_code(text: str) -> str | None:
        """Pull the last fenced python block from a model message."""
        matches = _CODE_FENCE.findall(text or "")
        if matches:
            return matches[-1].strip()
        return None  # no code submitted this turn

    def runner_source(self) -> str:
        return _RUNNER_TEMPLATE.format(
            mode=self.check_mode, checks=self.checks, setup=self.test_setup
        )

    def observation(self, exit_code: int, stdout: str, stderr: str) -> str:
        if exit_code == 0:
            return "Checker: PASSED — all checks green. Task solved."
        tail = (stdout + stderr).strip()
        tail = tail[-1500:] if len(tail) > 1500 else tail
        return f"Checker: FAILED (exit {exit_code}). Output:\n{tail}\nFix solution.py and resubmit."

    @staticmethod
    def reward(exit_code: int) -> float:
        """Binary task reward: 1.0 iff every check passed."""
        return 1.0 if exit_code == 0 else 0.0


def load_tasks(path: str | Path) -> list[CodingTask]:
    tasks: list[CodingTask] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tasks.append(CodingTask(**json.loads(line)))
    return tasks


# --- Shared checker builder + task-family registry --------------------------------------------

def assert_runner(checks: list[str], setup: str = "") -> str:
    """The MBPP-style assert checker as standalone source, reusable by other families.

    Imports the agent's `solution.py`, seeds a namespace with its globals + `setup`, then
    runs each bare-name assert statement; exit 0 iff all pass. Five of the six task families
    (coding/SWE/tool-use/math/CTF/RAG) express their reward this way — they differ only in
    the prompt, the seed files, and which asserts they hand in. SQL needs a custom runner
    (it runs a query against a DB), so it overrides `runner_source` instead.
    """
    return _RUNNER_TEMPLATE.format(mode="assert", checks=checks, setup=setup)


# family name -> Task class. Populated by CodingTask below and by task_families on import.
FAMILY_REGISTRY: dict[str, type] = {}


def register_family(cls: type) -> type:
    """Class decorator / helper: register a task class under its ClassVar `family`."""
    FAMILY_REGISTRY[cls.family] = cls
    return cls


def task_from_dict(d: dict):
    """Reconstruct the right family Task from a serialized dict (dataset `metadata.task`).

    Dispatches on the `family` key (default "coding" for legacy MBPP/KernelBench rows that
    predate the task-family scaffold and carry no discriminator)."""
    from . import task_families  # noqa: F401 — ensures every family self-registers

    fam = d.get("family", "coding")
    cls = FAMILY_REGISTRY.get(fam)
    if cls is None:
        raise ValueError(f"unknown task family: {fam!r} (known: {sorted(FAMILY_REGISTRY)})")
    return cls.from_dict(d)


register_family(CodingTask)
