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
