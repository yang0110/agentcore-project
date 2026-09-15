"""The agentic-RL task-family zoo — one harness, many tasks (Demo A, generalized).

`demo_cpu_single`'s in-process `run_episode` harness is task-agnostic: it seeds a workspace, asks
the policy for a submission, runs a hidden checker in the sandbox, and turns the checker's
exit code into a deterministic reward (RLVR — reward from verification, no LLM judge). MBPP
(coding) is just the first family. This module plugs the rest of the agentic-RL task zoo
into the *same* harness by changing only three things per family:

    (1) the prompt,           (2) the seed workspace (`setup_files`),   (3) the checker.

Six families live here, each a thin `EpisodicTask` subclass:

    coding   -> agentcore_slime_demo.task.CodingTask  (MBPP/HumanEval; lives in task.py)
    swe      -> SweTask       fix a buggy module so its test suite passes
    tool_use -> ToolUseTask   complete a task using ONLY the provided tools.py
    sql      -> SqlTask       write one SQL query whose result set equals the gold answer
    math     -> MathTask      compute a numeric answer (assert within tolerance)
    ctf      -> CtfTask       find the flag a verifier `check(flag)` accepts
    rag      -> RagTask       answer a question grounded in a local ./docs corpus

Five of the six express their reward through the shared MBPP-style assert checker
(`task.assert_runner`): the agent's `solution.py` sets a module-level variable and the
checker asserts over it. SQL is the one exception — it runs the submitted query against a
throwaway SQLite DB and compares the result set — so it overrides `runner_source`.

All checkers are pure-CPU, stdlib-only, and fully offline (no network, no GPU), so every
family runs on the same CPU Code-Interpreter microVM demo_cpu_single uses, and the whole loop is
exercisable with `SANDBOX=local` + MockPolicy (see tests/test_smoke.py). The seed datasets
here are demo-scale (hand-authored / templated, ~10 tasks/family); they are meant to prove
the scaffold and give GRPO real reward variance, not to be benchmark-sized corpora.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import ClassVar

from .task import assert_runner, register_family

# Last fenced block matching the family's language; falls back to any fenced block (so a
# policy — or MockPolicy, which always emits a ```python fence — still parses for SQL etc.).
def _extract_fenced(text: str, lang: str) -> str | None:
    if not text:
        return None
    lang_pat = re.compile(rf"```{lang}[^\n]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
    matches = lang_pat.findall(text)
    if not matches:
        matches = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL).findall(text)
    return matches[-1].strip() if matches else None


@dataclass
class EpisodicTask:
    """Base for the non-coding families: a prompt + a `spec` the checker consumes.

    Uniform serialization (`{family, task_id, prompt, max_turns, spec}`) so the dataset
    builder and `task.task_from_dict` round-trip every family the same way. `mock_attempts`
    is offline-only (MockPolicy) and is intentionally dropped from the dataset row so the
    reference answer can never leak into a real-RL prompt.
    """

    task_id: str
    prompt: str
    spec: dict = field(default_factory=dict)
    max_turns: int = 4
    mock_attempts: list[str] = field(default_factory=list)

    # Subclasses override these.
    family: ClassVar[str] = "base"
    artifact_filename: ClassVar[str] = "solution.py"
    check_command: ClassVar[str] = "python3 run_checks.py"
    fence_lang: ClassVar[str] = "python"
    SYSTEM: ClassVar[str] = "You are an agent working inside a sandbox."
    submission_hint: ClassVar[str] = (
        "Reply with the full solution.py in one ```python code block."
    )

    # ---- harness interface ---------------------------------------------------
    def system_prompt(self) -> str:
        return self.SYSTEM

    def initial_user_prompt(self) -> str:
        return self.prompt

    def setup_files(self) -> dict[str, str]:
        return {}

    def runner_source(self) -> str:
        # Default = the shared MBPP-style assert checker over the agent's solution.py.
        return assert_runner(self.spec.get("checks", []), self.spec.get("setup", ""))

    def extract_submission(self, text: str) -> str | None:
        return _extract_fenced(text, self.fence_lang)

    def observation(self, exit_code: int, stdout: str, stderr: str) -> str:
        if exit_code == 0:
            return "Checker: PASSED — all checks green. Task solved."
        tail = (stdout + stderr).strip()
        tail = tail[-1500:] if len(tail) > 1500 else tail
        return (
            f"Checker: FAILED (exit {exit_code}). Output:\n{tail}\n"
            "Fix your submission and resubmit the full file."
        )

    @staticmethod
    def reward(exit_code: int) -> float:
        return 1.0 if exit_code == 0 else 0.0

    # ---- serialization -------------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict) -> "EpisodicTask":
        return cls(
            task_id=d["task_id"],
            prompt=d["prompt"],
            spec=d.get("spec", {}),
            max_turns=d.get("max_turns", 4),
            mock_attempts=d.get("mock_attempts", []),
        )

    def to_row_kwargs(self, max_turns: int | None = None) -> dict:
        return {
            "family": self.family,
            "task_id": self.task_id,
            "prompt": self.prompt,
            "max_turns": max_turns if max_turns is not None else self.max_turns,
            "spec": self.spec,
        }

    @classmethod
    def seeds(cls) -> list["EpisodicTask"]:  # pragma: no cover - overridden
        raise NotImplementedError


# ==============================================================================
# 1. SWE — fix a buggy module so its tests pass
# ==============================================================================
@register_family
class SweTask(EpisodicTask):
    family = "swe"
    SYSTEM = (
        "You are a software engineer fixing a bug. A module fails its test suite. Rewrite "
        "the whole file as solution.py so every test passes. Reply with ONE fenced ```python "
        "code block containing the full corrected file — no prose outside it. If the checker "
        "reports failures, fix the code and resubmit the full file."
    )
    submission_hint = "Reply with the full corrected solution.py in one ```python block."

    @classmethod
    def seeds(cls) -> list["SweTask"]:
        # (name, buggy source, fixed source, list of test asserts)
        bugs = [
            ("add", "def add(a, b):\n    return a - b\n",
             "def add(a, b):\n    return a + b\n",
             ["assert add(2, 3) == 5", "assert add(-1, 1) == 0", "assert add(0, 0) == 0"]),
            ("is_even", "def is_even(n):\n    return n % 2 == 1\n",
             "def is_even(n):\n    return n % 2 == 0\n",
             ["assert is_even(4) is True", "assert is_even(7) is False"]),
            ("last", "def last(xs):\n    return xs[0]\n",
             "def last(xs):\n    return xs[-1]\n",
             ["assert last([1, 2, 3]) == 3", "assert last(['a']) == 'a'"]),
            ("count_vowels",
             "def count_vowels(s):\n    return sum(1 for c in s if c in 'xyz')\n",
             "def count_vowels(s):\n    return sum(1 for c in s if c in 'aeiou')\n",
             ["assert count_vowels('hello') == 2", "assert count_vowels('sky') == 0"]),
            ("clamp", "def clamp(x, lo, hi):\n    return max(hi, min(x, lo))\n",
             "def clamp(x, lo, hi):\n    return max(lo, min(x, hi))\n",
             ["assert clamp(5, 0, 10) == 5", "assert clamp(-1, 0, 10) == 0",
              "assert clamp(99, 0, 10) == 10"]),
            ("factorial",
             "def factorial(n):\n    r = 1\n    for i in range(1, n):\n        r *= i\n    return r\n",
             "def factorial(n):\n    r = 1\n    for i in range(1, n + 1):\n        r *= i\n    return r\n",
             ["assert factorial(4) == 24", "assert factorial(0) == 1", "assert factorial(1) == 1"]),
            ("dedup", "def dedup(xs):\n    return xs\n",
             "def dedup(xs):\n    out = []\n    for x in xs:\n        if x not in out:\n            out.append(x)\n    return out\n",
             ["assert dedup([1, 1, 2, 3, 3]) == [1, 2, 3]", "assert dedup([]) == []"]),
            ("fib",
             "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a\n    return a\n",
             "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n",
             ["assert fib(0) == 0", "assert fib(1) == 1", "assert fib(7) == 13"]),
            ("title_case", "def title_case(s):\n    return s.upper()\n",
             "def title_case(s):\n    return s.title()\n",
             ["assert title_case('hello world') == 'Hello World'"]),
            ("second_largest",
             "def second_largest(xs):\n    return sorted(xs)[-1]\n",
             "def second_largest(xs):\n    return sorted(set(xs))[-2]\n",
             ["assert second_largest([3, 1, 4, 1, 5]) == 4", "assert second_largest([9, 2]) == 2"]),
        ]
        tasks = []
        for i, (name, buggy, fixed, checks) in enumerate(bugs):
            prompt = (
                f"The function `{name}` below has a bug — it fails its test suite:\n\n"
                f"```python\n{buggy}```\n\nFailing tests:\n" + "\n".join(checks) +
                "\n\nRewrite the file so all tests pass."
            )
            tasks.append(cls(
                task_id=f"swe_{i}_{name}", prompt=prompt, max_turns=3,
                spec={"checks": checks, "setup": ""},
                mock_attempts=[buggy, fixed],
            ))
        return tasks


# ==============================================================================
# 2. Tool-use — complete a task using ONLY the provided tools.py
# ==============================================================================
_TOOLS_SRC = '''\
"""Sandbox-provided tools. The agent must USE these — not reimplement pricing."""
CALLS = []
PRICES = {"widget": 5, "gadget": 8, "gizmo": 12, "bolt": 2, "nut": 1, "cog": 7}


def lookup(name):
    """Return the unit price of an item."""
    CALLS.append(("lookup", name))
    return PRICES[name]


def multiply(a, b):
    CALLS.append(("multiply", a, b))
    return a * b


def add(a, b):
    CALLS.append(("add", a, b))
    return a + b
'''

_PRICES = {"widget": 5, "gadget": 8, "gizmo": 12, "bolt": 2, "nut": 1, "cog": 7}


@register_family
class ToolUseTask(EpisodicTask):
    family = "tool_use"
    SYSTEM = (
        "You are a tool-using agent. A module `tools.py` is already present in the sandbox "
        "with functions lookup(name), multiply(a, b) and add(a, b). Use ONLY those tools — do "
        "not hardcode prices or reimplement them. Write solution.py that imports tools, uses "
        "them to solve the task, and stores the final answer in a module-level variable "
        "`result`. Reply with ONE fenced ```python block."
    )
    submission_hint = "Reply with solution.py (import tools; set `result`) in one ```python block."

    def setup_files(self) -> dict[str, str]:
        return {"tools.py": _TOOLS_SRC}

    @classmethod
    def seeds(cls) -> list["ToolUseTask"]:
        orders = [
            (3, "widget", 2, "gadget"), (5, "gizmo", 4, "bolt"), (10, "nut", 1, "cog"),
            (2, "gadget", 7, "widget"), (6, "cog", 3, "gizmo"), (4, "bolt", 9, "nut"),
            (8, "widget", 8, "gizmo"), (1, "gadget", 1, "cog"), (12, "nut", 5, "widget"),
            (7, "gizmo", 2, "bolt"),
        ]
        tasks = []
        for i, (q1, it1, q2, it2) in enumerate(orders):
            total = q1 * _PRICES[it1] + q2 * _PRICES[it2]
            prompt = (
                f"An order contains {q1} × {it1} and {q2} × {it2}. Using only the tools in "
                f"tools.py, compute the total cost and store it in `result`."
            )
            checks = [
                f"assert result == {total}, f'result={{result}} expected {total}'",
                "assert any(c[0] == 'lookup' for c in __import__('tools').CALLS), 'you must call lookup()'",
                "assert any(c[0] == 'multiply' for c in __import__('tools').CALLS), 'you must call multiply()'",
            ]
            right = (
                "import tools\n"
                f"p1 = tools.lookup({it1!r})\n"
                f"p2 = tools.lookup({it2!r})\n"
                f"result = tools.add(tools.multiply(p1, {q1}), tools.multiply(p2, {q2}))\n"
            )
            wrong = "result = 0  # forgot to use the tools\n"
            tasks.append(cls(
                task_id=f"tool_{i}", prompt=prompt, max_turns=3,
                spec={"checks": checks, "setup": ""},
                mock_attempts=[wrong, right],
            ))
        return tasks


# ==============================================================================
# 3. Data / SQL — one query whose result set equals the gold answer
# ==============================================================================
_SQL_RUNNER = '''\
import sqlite3, sys

SCHEMA = {schema}
INSERTS = {inserts}
EXPECTED = {expected}
ORDER_MATTERS = {order}

try:
    query = open("query.sql").read().strip().rstrip(";").strip()
except Exception as e:
    print("NO_QUERY:", e); sys.exit(1)
if not query:
    print("EMPTY_QUERY — write one SQL statement"); sys.exit(1)
if ";" in query:
    print("ONE_STATEMENT_ONLY — remove extra ';'-separated statements"); sys.exit(1)

con = sqlite3.connect(":memory:")
cur = con.cursor()
try:
    for stmt in SCHEMA:
        cur.execute(stmt)
    for stmt in INSERTS:
        cur.execute(stmt)
except Exception as e:
    print("DB_SETUP_ERROR:", type(e).__name__, e); sys.exit(1)

try:
    cur.execute(query)
    rows = [list(r) for r in cur.fetchall()]
except Exception as e:
    print("QUERY_ERROR:", type(e).__name__, e); sys.exit(1)

expected = [list(r) for r in EXPECTED]
if ORDER_MATTERS:
    ok = rows == expected
else:
    ok = sorted(map(repr, rows)) == sorted(map(repr, expected))
if ok:
    print("PASSED", rows); sys.exit(0)
print("FAIL got", rows, "expected", expected); sys.exit(1)
'''

_EMP_SCHEMA = ["CREATE TABLE emp (id INTEGER, name TEXT, dept TEXT, age INTEGER, salary INTEGER)"]
_EMP_ROWS = [
    (1, "Ann", "eng", 34, 120), (2, "Bo", "eng", 41, 140), (3, "Cy", "sales", 29, 90),
    (4, "Di", "sales", 52, 110), (5, "Ed", "hr", 38, 80), (6, "Fi", "eng", 26, 100),
    (7, "Gu", "sales", 45, 130), (8, "Ha", "hr", 31, 85),
]
_EMP_INSERTS = [
    f"INSERT INTO emp VALUES ({i}, '{n}', '{d}', {a}, {s})" for (i, n, d, a, s) in _EMP_ROWS
]


@register_family
class SqlTask(EpisodicTask):
    family = "sql"
    artifact_filename = "query.sql"
    fence_lang = "sql"
    SYSTEM = (
        "You are a data analyst. Write a SINGLE SQL query (SQLite dialect) that answers the "
        "question against the given schema. Reply with ONE fenced ```sql block containing only "
        "the query (no explanation, no trailing semicolon, one statement)."
    )
    submission_hint = "Reply with a single SQL statement in one ```sql block."

    def initial_user_prompt(self) -> str:
        schema_txt = "\n".join(self.spec["schema"])
        return (
            f"Schema:\n```sql\n{schema_txt}\n```\n\n{self.prompt}\n\n"
            "Return the answer as the result set of one SQL query."
        )

    def runner_source(self) -> str:
        sp = self.spec
        return _SQL_RUNNER.format(
            schema=repr(sp["schema"]), inserts=repr(sp["inserts"]),
            expected=repr(sp["expected"]), order=repr(sp.get("order_matters", False)),
        )

    @classmethod
    def _task(cls, i, question, expected, correct_sql, wrong_sql, order=False):
        return cls(
            task_id=f"sql_{i}", prompt=question, max_turns=3,
            spec={"schema": _EMP_SCHEMA, "inserts": _EMP_INSERTS,
                  "expected": expected, "order_matters": order},
            mock_attempts=[wrong_sql, correct_sql],
        )

    @classmethod
    def seeds(cls) -> list["SqlTask"]:
        q = cls._task
        return [
            q(0, "How many employees are in the 'eng' department?", [[3]],
              "SELECT COUNT(*) FROM emp WHERE dept = 'eng'", "SELECT COUNT(*) FROM emp"),
            q(1, "What is the total salary of the 'sales' department?", [[330]],
              "SELECT SUM(salary) FROM emp WHERE dept = 'sales'",
              "SELECT salary FROM emp WHERE dept = 'sales'"),
            q(2, "List the names of employees older than 40, ordered alphabetically.",
              [["Bo"], ["Di"], ["Gu"]],
              "SELECT name FROM emp WHERE age > 40 ORDER BY name",
              "SELECT name FROM emp ORDER BY name", order=True),
            q(3, "How many employees earn a salary of at least 110?", [[4]],
              "SELECT COUNT(*) FROM emp WHERE salary >= 110",
              "SELECT COUNT(*) FROM emp WHERE salary > 110"),
            q(4, "What is the total salary of the 'eng' department?", [[360]],
              "SELECT SUM(salary) FROM emp WHERE dept = 'eng'",
              "SELECT COUNT(salary) FROM emp WHERE dept = 'eng'"),
            q(5, "How many employees are in the 'hr' department?", [[2]],
              "SELECT COUNT(*) FROM emp WHERE dept = 'hr'", "SELECT COUNT(*) FROM emp"),
            q(6, "What is the highest salary in the company?", [[140]],
              "SELECT MAX(salary) FROM emp", "SELECT MIN(salary) FROM emp"),
            q(7, "What is the smallest age in the company?", [[26]],
              "SELECT MIN(age) FROM emp", "SELECT MAX(age) FROM emp"),
            q(8, "List the names of 'eng' employees, ordered alphabetically.",
              [["Ann"], ["Bo"], ["Fi"]],
              "SELECT name FROM emp WHERE dept = 'eng' ORDER BY name",
              "SELECT name FROM emp ORDER BY name", order=True),
            q(9, "What is the total salary of the 'hr' department?", [[165]],
              "SELECT SUM(salary) FROM emp WHERE dept = 'hr'",
              "SELECT AVG(salary) FROM emp WHERE dept = 'hr'"),
        ]


# ==============================================================================
# 4. Math — compute a numeric answer (assert within tolerance)
# ==============================================================================
@register_family
class MathTask(EpisodicTask):
    family = "math"
    SYSTEM = (
        "You are solving a math word problem. Write solution.py that computes the answer and "
        "stores it in a module-level variable `answer` (a number). Reply with ONE fenced "
        "```python block — no prose outside it."
    )
    submission_hint = "Reply with solution.py that sets `answer` in one ```python block."

    @classmethod
    def seeds(cls) -> list["MathTask"]:
        combos = [
            (3, 4, 5, 6), (10, 2, 7, 3), (8, 5, 2, 9), (12, 10, 6, 4), (7, 7, 3, 3),
            (15, 2, 5, 8), (9, 11, 4, 2), (6, 3, 13, 1), (20, 5, 3, 7), (4, 25, 8, 2),
            (11, 6, 9, 4), (2, 50, 3, 10),
        ]
        tasks = []
        for i, (w, px, g, py) in enumerate(combos):
            total = w * px + g * py
            prompt = (
                f"A shop sold {w} notebooks at ${px} each and {g} pens at ${py} each. "
                "What was the total revenue, in dollars?"
            )
            checks = [f"assert abs(answer - {total}) <= 1e-6, f'answer={{answer}} expected {total}'"]
            wrong = f"answer = {total + px}  # forgot the pens\n"
            right = f"answer = {w} * {px} + {g} * {py}\n"
            tasks.append(cls(
                task_id=f"math_{i}", prompt=prompt, max_turns=3,
                spec={"checks": checks, "setup": ""},
                mock_attempts=[wrong, right],
            ))
        return tasks


# ==============================================================================
# 5. Security / CTF — find the flag a verifier check(flag) accepts
# ==============================================================================
@register_family
class CtfTask(EpisodicTask):
    family = "ctf"
    SYSTEM = (
        "You are solving a security challenge (reversing). Below is a verifier check(flag) "
        "that returns True only for the correct flag. Reverse-engineer it and write solution.py "
        "that stores the winning flag string in a module-level variable `flag`. Reply with ONE "
        "fenced ```python block."
    )
    submission_hint = "Reply with solution.py that sets `flag` in one ```python block."

    @classmethod
    def _reverse(cls, i, secret):
        rev = secret[::-1]
        check = f"def check(flag):\n    return flag[::-1] == {rev!r}\n"
        return cls(
            task_id=f"ctf_rev_{i}",
            prompt=(f"Reverse-engineer this verifier and find the flag:\n\n```python\n{check}```"),
            max_turns=3, spec={"checks": ["assert check(flag) is True"], "setup": check},
            mock_attempts=["flag = 'nope'\n", f"flag = {secret!r}\n"],
        )

    @classmethod
    def _caesar(cls, i, secret, shift):
        shifted = "".join(chr(ord(c) + shift) for c in secret)
        check = (
            f"def check(flag):\n"
            f"    return ''.join(chr(ord(c) + {shift}) for c in flag) == {shifted!r}\n"
        )
        return cls(
            task_id=f"ctf_caesar_{i}",
            prompt=(f"Reverse-engineer this verifier and find the flag:\n\n```python\n{check}```"),
            max_turns=3, spec={"checks": ["assert check(flag) is True"], "setup": check},
            mock_attempts=["flag = 'wrong'\n", f"flag = {secret!r}\n"],
        )

    @classmethod
    def seeds(cls) -> list["CtfTask"]:
        secrets = ["agentcor3", "sl1me_rl", "b3drock", "sandb0x", "rollout9"]
        tasks = [cls._reverse(i, s) for i, s in enumerate(secrets)]
        caesars = [("kernel7", 3), ("qwen_4b", 1), ("gr8po", 5), ("microVM", 2), ("capacity", 4)]
        tasks += [cls._caesar(i, s, sh) for i, (s, sh) in enumerate(caesars)]
        return tasks


# ==============================================================================
# 6. Agentic RAG — answer a question grounded in a local ./docs corpus
# ==============================================================================
@register_family
class RagTask(EpisodicTask):
    family = "rag"
    SYSTEM = (
        "You are a retrieval-augmented agent. Answer the question using ONLY the text files in "
        "the ./docs directory (the answer is NOT in your training data — you must read the "
        "documents). Write solution.py that reads ./docs, finds the answer, and stores it as a "
        "string in a module-level variable `answer`. Reply with ONE fenced ```python block."
    )
    submission_hint = "Reply with solution.py that reads ./docs and sets `answer` in one ```python block."

    def setup_files(self) -> dict[str, str]:
        return {f"docs/{name}": text for name, text in self.spec["corpus"].items()}

    @classmethod
    def _task(cls, i, corpus, question, gold):
        gold_norm = str(gold).strip().lower()
        # a couple of distractor docs + the answer doc make retrieval non-trivial
        return cls(
            task_id=f"rag_{i}", prompt=question, max_turns=3,
            spec={"corpus": corpus,
                  "checks": [f"assert str(answer).strip().lower() == {gold_norm!r}, "
                             f"f'answer={{answer!r}}'"],
                  "setup": ""},
            mock_attempts=["answer = 'unknown'\n", f"answer = {gold!r}\n"],
        )

    @classmethod
    def seeds(cls) -> list["RagTask"]:
        return [
            cls._task(
                0,
                {"a.txt": "The Zorbian Council governs the northern provinces.",
                 "b.txt": "The Meridian Accord was ratified by the Zorbian Council in the year 4471.",
                 "c.txt": "Trade along the Vell River peaked two centuries earlier."},
                "In what year was the Meridian Accord ratified?", "4471"),
            cls._task(
                1,
                {"a.txt": "Captain Voss commanded the survey vessel Halcyon.",
                 "b.txt": "The Halcyon's home port is Brindlemark on the eastern coast.",
                 "c.txt": "Its sister ship, the Ember, was decommissioned."},
                "What is the home port of the survey vessel Halcyon?", "Brindlemark"),
            cls._task(
                2,
                {"a.txt": "Project Larkspur studies deep-sea bioluminescence.",
                 "b.txt": "Project Larkspur is led by Dr. Imani Osei since its founding.",
                 "c.txt": "Funding comes from the Terran Marine Institute."},
                "Who leads Project Larkspur?", "Dr. Imani Osei"),
            cls._task(
                3,
                {"a.txt": "The alloy Duranite is prized for aerospace fittings.",
                 "b.txt": "Duranite melts at 2310 degrees Celsius under standard pressure.",
                 "c.txt": "It was first synthesized on the orbital station Kepler-9."},
                "At how many degrees Celsius does Duranite melt?", "2310"),
            cls._task(
                4,
                {"a.txt": "The festival of Nualan is held in the walled city of Cresthaven.",
                 "b.txt": "Nualan lasts exactly nine days each autumn.",
                 "c.txt": "Visitors arrive by rail from the lowland towns."},
                "How many days does the festival of Nualan last?", "nine"),
            cls._task(
                5,
                {"a.txt": "The compiler Basilisk targets the Aether virtual machine.",
                 "b.txt": "Basilisk version 3 introduced tail-call optimization.",
                 "c.txt": "The Aether VM uses a register-based bytecode."},
                "Which version of the Basilisk compiler introduced tail-call optimization?",
                "3"),
            cls._task(
                6,
                {"a.txt": "The Solveig Prize honors advances in cryptography.",
                 "b.txt": "In 4402 the Solveig Prize went to the Lattice-Sieve team.",
                 "c.txt": "The prize includes a residency at the Halden Institute."},
                "Which team won the Solveig Prize in 4402?", "Lattice-Sieve"),
            cls._task(
                7,
                {"a.txt": "Windmoor is the capital of the province of Aldreth.",
                 "b.txt": "Aldreth borders the Grey Marshes to the south.",
                 "c.txt": "Its chief export is spun glass."},
                "What is the capital of the province of Aldreth?", "Windmoor"),
        ]


# ==============================================================================
# 7. Regex — synthesize a pattern that matches positives and rejects negatives
# ==============================================================================
@register_family
class RegexTask(EpisodicTask):
    family = "regex"
    SYSTEM = (
        "You are synthesizing a regular expression. Write solution.py that sets a module-level "
        "string variable `pattern` to a Python regex which FULL-MATCHES every positive example "
        "and matches NONE of the negatives. Reply with ONE fenced ```python block."
    )
    submission_hint = "Reply with solution.py that sets `pattern` in one ```python block."

    @classmethod
    def _t(cls, i, desc, correct, pos, neg):
        setup = f"import re\nPOS = {pos!r}\nNEG = {neg!r}\n"
        checks = [
            "assert all(re.fullmatch(pattern, s) for s in POS), 'a positive example did not match'",
            "assert all(re.fullmatch(pattern, s) is None for s in NEG), 'a negative example matched'",
        ]
        prompt = (
            f"{desc}\n\nYour regex must full-match every positive and match none of the negatives.\n"
            f"Positives: {pos}\nNegatives: {neg}\n\n"
            "Write solution.py that sets a module-level string variable `pattern`."
        )
        return cls(task_id=f"regex_{i}", prompt=prompt, max_turns=3,
                   spec={"checks": checks, "setup": setup},
                   mock_attempts=["pattern = '.*'\n", f"pattern = {correct!r}\n"])

    @classmethod
    def seeds(cls) -> list["RegexTask"]:
        return [
            cls._t(0, "Match a US 5-digit ZIP code.", r"\d{5}",
                   ["12345", "00042"], ["1234", "123456", "ab123"]),
            cls._t(1, "Match a 6-digit hex color like #a1b2c3.", r"#[0-9a-fA-F]{6}",
                   ["#a1b2c3", "#FFFFFF", "#000000"], ["a1b2c3", "#12345", "#gggggg", "#1234567"]),
            cls._t(2, "Match a Python identifier.", r"[A-Za-z_]\w*",
                   ["foo", "_bar", "x1", "A_1"], ["1abc", "a-b", "", "x y"]),
            cls._t(3, "Match a US phone number ddd-ddd-dddd.", r"\d{3}-\d{3}-\d{4}",
                   ["123-456-7890"], ["1234567890", "12-345-6789", "123-45-6789"]),
            cls._t(4, "Match a 24-hour time HH:MM.", r"([01]\d|2[0-3]):[0-5]\d",
                   ["00:00", "09:30", "23:59"], ["24:00", "9:30", "12:60", "23:5"]),
            cls._t(5, "Match a lowercase word (letters only).", r"[a-z]+",
                   ["hello", "abc"], ["Hello", "abc1", "1"]),
            cls._t(6, "Match a simple email like name@host.tld.", r"[\w.]+@\w+\.[a-z]{2,}",
                   ["a.b@host.com", "x@y.io"], ["a@b", "a@@b.com", "a b@c.com"]),
        ]


# ==============================================================================
# 8. Planning — solve a discrete search / constraint puzzle, output a valid plan
# ==============================================================================
@register_family
class PlanningTask(EpisodicTask):
    family = "planning"
    SYSTEM = (
        "You are a planning agent solving a discrete puzzle. Write solution.py that sets a "
        "module-level variable `plan` to a valid solution (a Python list). A hidden verifier "
        "checks it. Reply with ONE fenced ```python block."
    )
    submission_hint = "Reply with solution.py that sets `plan` in one ```python block."

    @classmethod
    def _queens(cls, n, sol):
        setup = (
            "def verify(plan):\n"
            f"    n = {n}\n"
            "    if sorted(plan) != list(range(n)): return False\n"
            "    for a in range(n):\n"
            "        for b in range(a + 1, n):\n"
            "            if abs(plan[a] - plan[b]) == abs(a - b): return False\n"
            "    return True\n"
        )
        prompt = (
            f"Place {n} non-attacking queens on a {n}x{n} chessboard. Set `plan` to a list of {n} "
            f"integers where plan[r] is the column (0..{n - 1}) of the queen in row r. No two queens "
            "may share a column or a diagonal."
        )
        return cls(task_id=f"plan_queens_{n}", prompt=prompt, max_turns=3,
                   spec={"checks": ["assert verify(plan) is True"], "setup": setup},
                   mock_attempts=[f"plan = list(range({n}))\n", f"plan = {sol}\n"])

    @classmethod
    def _subset(cls, i, nums, target, sol, wrong):
        setup = (
            f"NUMS = {nums!r}\nTARGET = {target}\n"
            "def verify(plan):\n"
            "    if len(set(plan)) != len(plan): return False\n"
            "    if any(k < 0 or k >= len(NUMS) for k in plan): return False\n"
            "    return sum(NUMS[k] for k in plan) == TARGET\n"
        )
        prompt = (
            f"Given NUMS = {nums}, choose a subset whose values sum to exactly {target}. Set `plan` "
            "to the list of chosen INDICES (0-based, no repeats)."
        )
        return cls(task_id=f"plan_subset_{i}", prompt=prompt, max_turns=3,
                   spec={"checks": ["assert verify(plan) is True"], "setup": setup},
                   mock_attempts=[f"plan = {wrong}\n", f"plan = {sol}\n"])

    @classmethod
    def seeds(cls) -> list["PlanningTask"]:
        return [
            cls._queens(4, [1, 3, 0, 2]),
            cls._queens(5, [0, 2, 4, 1, 3]),
            cls._queens(6, [1, 3, 5, 0, 2, 4]),
            cls._subset(0, [3, 7, 1, 8, 4, 6], 15, [0, 3, 4], [0, 1]),
            cls._subset(1, [2, 4, 6, 8, 10], 18, [1, 2, 3], [0, 1]),
            cls._subset(2, [5, 1, 9, 3, 7], 12, [2, 3], [0, 1]),
            cls._subset(3, [10, 20, 30, 40], 60, [1, 3], [0, 1]),
            cls._subset(4, [1, 2, 3, 4, 5, 6], 11, [4, 5], [0, 1]),
        ]


# ==============================================================================
# 9. Config — repair a structured config so it satisfies validation constraints
# ==============================================================================
@register_family
class ConfigTask(EpisodicTask):
    family = "config"
    SYSTEM = (
        "You are fixing a configuration. Write solution.py that sets a module-level dict `config` "
        "satisfying ALL stated requirements. A draft is given but violates some of them. Reply "
        "with ONE fenced ```python block."
    )
    submission_hint = "Reply with solution.py that sets the dict `config` in one ```python block."

    @classmethod
    def _t(cls, i, desc, draft, checks, right):
        prompt = (
            f"{desc}\n\nThis draft violates the requirements — fix it:\n"
            f"```python\nconfig = {draft}\n```\n\nWrite solution.py that sets a module-level dict `config`."
        )
        return cls(task_id=f"config_{i}", prompt=prompt, max_turns=3,
                   spec={"checks": checks, "setup": ""},
                   mock_attempts=[f"config = {draft}\n", f"config = {right!r}\n"])

    @classmethod
    def seeds(cls) -> list["ConfigTask"]:
        return [
            cls._t(0,
                   "Web server config: host (non-empty str), port (int in 1024..65535), "
                   "workers (int >= 1), tls (must be boolean True).",
                   "{'host': '', 'port': 80, 'workers': 0, 'tls': 'yes'}",
                   ["assert isinstance(config['host'], str) and config['host']",
                    "assert isinstance(config['port'], int) and 1024 <= config['port'] <= 65535",
                    "assert isinstance(config['workers'], int) and config['workers'] >= 1",
                    "assert config['tls'] is True"],
                   {'host': 'example.com', 'port': 8080, 'workers': 4, 'tls': True}),
            cls._t(1,
                   "DB pool config: engine (one of 'postgres'/'mysql'), max_conns (int 1..100), "
                   "timeout_s (number > 0).",
                   "{'engine': 'sqlite', 'max_conns': 500, 'timeout_s': 0}",
                   ["assert config['engine'] in ('postgres', 'mysql')",
                    "assert isinstance(config['max_conns'], int) and 1 <= config['max_conns'] <= 100",
                    "assert config['timeout_s'] > 0"],
                   {'engine': 'postgres', 'max_conns': 20, 'timeout_s': 5}),
            cls._t(2,
                   "Cache config: backend (one of 'redis'/'memcached'), ttl_s (int > 0), "
                   "max_mb (int >= 1).",
                   "{'backend': 'disk', 'ttl_s': -1, 'max_mb': 0}",
                   ["assert config['backend'] in ('redis', 'memcached')",
                    "assert isinstance(config['ttl_s'], int) and config['ttl_s'] > 0",
                    "assert isinstance(config['max_mb'], int) and config['max_mb'] >= 1"],
                   {'backend': 'redis', 'ttl_s': 300, 'max_mb': 64}),
            cls._t(3,
                   "Logging config: level (one of DEBUG/INFO/WARN/ERROR), json (bool), "
                   "path (non-empty str).",
                   "{'level': 'TRACE', 'json': 'true', 'path': ''}",
                   ["assert config['level'] in ('DEBUG', 'INFO', 'WARN', 'ERROR')",
                    "assert isinstance(config['json'], bool)",
                    "assert isinstance(config['path'], str) and config['path']"],
                   {'level': 'INFO', 'json': True, 'path': '/var/log/app.log'}),
            cls._t(4,
                   "Retry policy: max_attempts (int >= 1), backoff (one of 'fixed'/'exponential'), "
                   "base_ms (int > 0).",
                   "{'max_attempts': 0, 'backoff': 'random', 'base_ms': 0}",
                   ["assert isinstance(config['max_attempts'], int) and config['max_attempts'] >= 1",
                    "assert config['backoff'] in ('fixed', 'exponential')",
                    "assert isinstance(config['base_ms'], int) and config['base_ms'] > 0"],
                   {'max_attempts': 3, 'backoff': 'exponential', 'base_ms': 100}),
            cls._t(5,
                   "Rate-limiter config: requests_per_min (int in 1..10000), burst (int >= 1 and "
                   "<= requests_per_min), enabled (must be boolean True).",
                   "{'requests_per_min': 0, 'burst': 50, 'enabled': 1}",
                   ["assert isinstance(config['requests_per_min'], int) and 1 <= config['requests_per_min'] <= 10000",
                    "assert isinstance(config['burst'], int) and 1 <= config['burst'] <= config['requests_per_min']",
                    "assert config['enabled'] is True"],
                   {'requests_per_min': 600, 'burst': 60, 'enabled': True}),
        ]


# ==============================================================================
# 10. Data / ETL — read input.json, produce a transformed `output`
# ==============================================================================
@register_family
class DataTask(EpisodicTask):
    family = "data"
    SYSTEM = (
        "You are a data-wrangling agent. The file input.json is present in the sandbox. Write "
        "solution.py that reads ./input.json, computes the requested transform, and stores it in "
        "a module-level variable `output`. Reply with ONE fenced ```python block."
    )
    submission_hint = "Reply with solution.py that reads input.json and sets `output` in one ```python block."

    def setup_files(self) -> dict[str, str]:
        return {"input.json": self.spec["input_json"]}

    @classmethod
    def _t(cls, i, rows, desc, expected, right_body):
        import json as _j
        prompt = (
            f"input.json contains: {_j.dumps(rows)}\n\n{desc}\n\n"
            "Write solution.py that reads ./input.json and sets a module-level variable `output`."
        )
        return cls(task_id=f"data_{i}", prompt=prompt, max_turns=3,
                   spec={"input_json": _j.dumps(rows),
                         "checks": [f"assert output == {expected!r}"], "setup": ""},
                   mock_attempts=["output = None\n", right_body])

    @classmethod
    def seeds(cls) -> list["DataTask"]:
        _read = "import json\ndata = json.load(open('input.json'))\n"
        return [
            cls._t(0,
                   [{"name": "Ann", "age": 34}, {"name": "Bo", "age": 15},
                    {"name": "Cy", "age": 20}, {"name": "Di", "age": 12}],
                   "Set `output` to the alphabetically-sorted list of names of people aged 18 or older.",
                   ["Ann", "Cy"],
                   _read + "output = sorted(p['name'] for p in data if p['age'] >= 18)\n"),
            cls._t(1,
                   [{"item": "a", "amount": 5}, {"item": "b", "amount": 12}, {"item": "c", "amount": 3}],
                   "Set `output` to the total of all 'amount' fields.", 20,
                   _read + "output = sum(r['amount'] for r in data)\n"),
            cls._t(2,
                   [{"name": "Ann", "age": 34}, {"name": "Bo", "age": 41}, {"name": "Cy", "age": 20}],
                   "Set `output` to the name of the oldest person.", "Bo",
                   _read + "output = max(data, key=lambda p: p['age'])['name']\n"),
            cls._t(3,
                   [{"city": "NY"}, {"city": "LA"}, {"city": "NY"}, {"city": "SF"}, {"city": "LA"}],
                   "Set `output` to a dict mapping each city to how many times it appears.",
                   {"NY": 2, "LA": 2, "SF": 1},
                   _read + "from collections import Counter\noutput = dict(Counter(r['city'] for r in data))\n"),
            cls._t(4,
                   [{"name": "Ann", "age": 34}, {"name": "Bo", "age": 41}, {"name": "Cy", "age": 20}],
                   "Set `output` to the list of names ordered by age, oldest first.",
                   ["Bo", "Ann", "Cy"],
                   _read + "output = [p['name'] for p in sorted(data, key=lambda p: -p['age'])]\n"),
            cls._t(5,
                   [{"email": "a@x.com", "active": True}, {"email": "b@x.com", "active": False},
                    {"email": "c@x.com", "active": True}],
                   "Set `output` to the list of emails of active users, in input order.",
                   ["a@x.com", "c@x.com"],
                   _read + "output = [r['email'] for r in data if r['active']]\n"),
        ]


# Every family class self-registers via @register_family on import; nothing else to do.
FAMILY_CLASSES = [
    SweTask, ToolUseTask, SqlTask, MathTask, CtfTask, RagTask,
    RegexTask, PlanningTask, ConfigTask, DataTask,
]
