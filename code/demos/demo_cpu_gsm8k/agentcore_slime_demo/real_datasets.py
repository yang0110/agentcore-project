"""Real agentic-RL datasets from HuggingFace, mapped onto the demo task families.

The hand-authored `seeds()` in `task_families.py` are demo-scale (~10 templated tasks/family)
— enough to prove the scaffold, not to be a benchmark-sized corpus. This module swaps in the
*real* datasets used in agentic RL / RLVR training, downloaded from HuggingFace, WITHOUT any
change to the harness or the checker: every loader returns the SAME `MathTask` / `CodingTask`
objects the seeds produce, so they serialize through the existing builders
(`task_to_row` / `to_row_kwargs`) and run through the unchanged in-process `run_episode`
loop + assert checker.

Verifiable reward only (RLVR — reward from a programmatic checker, no LLM judge):

    gsm8k     -> MathTask   (openai/gsm8k, config "main")      grade-school math, exact int match
    math500   -> MathTask   (HuggingFaceH4/MATH-500)           competition math, numeric-answer subset
    humaneval -> CodingTask (openai/openai_humaneval)          the dataset's own `test` fn, check(entry_point)

All checkers stay pure-CPU, stdlib-only and offline (they run in the same CPU Code-Interpreter
microVM every demo uses). Network is required once, at build time, to pull the HF rows.
"""

from __future__ import annotations

import os
import re
from fractions import Fraction
from pathlib import Path

from .task import CodingTask
from .task_families import MathTask, RagTask, SqlTask, ToolCallTask

# Default the HF cache to a writable project dir (the shared ~/.cache/huggingface is root-owned
# in the container; mirror mbpp.py's behavior so builds work as any user).
_DEFAULT_CACHE = Path(__file__).resolve().parents[1] / ".hf_cache"


def _ensure_cache() -> None:
    os.environ.setdefault("HF_HOME", str(_DEFAULT_CACHE))
    os.environ.setdefault("HF_DATASETS_CACHE", str(_DEFAULT_CACHE / "datasets"))
    Path(os.environ["HF_DATASETS_CACHE"]).mkdir(parents=True, exist_ok=True)


def _slice(ds, limit: int | None, offset: int):
    n = len(ds)
    end = n if limit is None else min(n, offset + limit)
    return (ds[i] for i in range(offset, end))


# ============================================================================
# Math — GSM8K (numeric, exact)
# ============================================================================
_MATH_SYS = MathTask.SYSTEM  # "...store it in a module-level variable `answer` (a number)..."


def _math_task(task_id: str, prompt: str, gold: float, *, tol: float = 1e-6,
               wrong: float | None = None, max_turns: int = 3) -> MathTask:
    # Format gold compactly: ints as ints, else a repr that round-trips.
    g = int(gold) if float(gold).is_integer() else float(gold)
    check = f"assert abs(answer - {g!r}) <= {tol!r}, f'answer={{answer}} expected {g!r}'"
    w = wrong if wrong is not None else (g + 1)
    return MathTask(
        task_id=task_id, prompt=prompt, max_turns=max_turns,
        spec={"checks": [check], "setup": ""},
        mock_attempts=[f"answer = {w!r}\n", f"answer = {g!r}\n"],
    )


def _gsm8k_gold(answer_field: str) -> int:
    # GSM8K answers end with "#### <int>" (may contain thousands separators).
    m = re.search(r"####\s*(-?[\d,]+)", answer_field or "")
    if not m:
        raise ValueError(f"no #### answer in: {answer_field!r}")
    return int(m.group(1).replace(",", ""))


def load_gsm8k(split: str = "train", limit: int | None = None, offset: int = 0) -> list[MathTask]:
    """openai/gsm8k (config 'main'). Grade-school word problems; gold is an integer."""
    _ensure_cache()
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)
    tasks: list[MathTask] = []
    for i, ex in enumerate(_slice(ds, limit, offset)):
        gold = _gsm8k_gold(ex["question"] and ex["answer"])
        tasks.append(_math_task(f"gsm8k_{split}_{offset + i}", ex["question"].strip(), gold))
    return tasks


# ============================================================================
# Math — MATH-500 (competition math; numeric-answer subset)
# ============================================================================
def _parse_numeric(ans: str) -> float | None:
    """Best-effort parse of a MATH boxed answer into a float; None if not numeric.

    Keeps the checker stdlib-only (no sympy in the sandbox): we only admit answers that
    reduce to a rational/decimal — integers, decimals, plain fractions a/b and \\frac{a}{b}.
    Non-numeric answers (surds, symbolic expressions, tuples) are skipped so every emitted
    task has an exact, offline-verifiable numeric reward.
    """
    if ans is None:
        return None
    s = ans.strip().strip("$").replace(" ", "").replace(",", "")
    s = s.replace("\\left", "").replace("\\right", "").replace("\\!", "")
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    if s.endswith("\\%") or s.endswith("%"):
        s = s.rstrip("%").rstrip("\\")
    # \frac{a}{b} or \dfrac{a}{b}
    m = re.fullmatch(r"\\d?frac\{(-?\d+)\}\{(-?\d+)\}", s)
    if m:
        try:
            return float(Fraction(int(m.group(1)), int(m.group(2))))
        except ZeroDivisionError:
            return None
    # plain a/b
    m = re.fullmatch(r"(-?\d+)/(-?\d+)", s)
    if m:
        try:
            return float(Fraction(int(m.group(1)), int(m.group(2))))
        except ZeroDivisionError:
            return None
    try:
        return float(s)
    except ValueError:
        return None


def load_math500(split: str = "test", limit: int | None = None, offset: int = 0) -> list[MathTask]:
    """HuggingFaceH4/MATH-500. Competition math; we keep the numeric-answer subset (tol 1e-4)."""
    _ensure_cache()
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split=split)
    tasks: list[MathTask] = []
    seen = 0
    for ex in ds:
        gold = _parse_numeric(ex.get("answer"))
        if gold is None:
            continue
        if seen < offset:
            seen += 1
            continue
        tasks.append(_math_task(
            f"math500_{ex.get('unique_id', seen)}".replace("/", "_"),
            ex["problem"].strip(), gold, tol=1e-4, max_turns=4,
        ))
        seen += 1
        if limit is not None and len(tasks) >= limit:
            break
    return tasks


# ============================================================================
# Coding — HumanEval (the dataset's own unit-test function)
# ============================================================================
_HUMANEVAL_STUB = "# TODO: implement the function described above\n"


def load_humaneval(split: str = "test", limit: int | None = None, offset: int = 0) -> list[CodingTask]:
    """openai/openai_humaneval. Each row ships a `test` fn defining check(candidate); the
    reward is `check(<entry_point>)` run in the checker (assert mode) against the agent's
    solution.py. The reference completion (prompt + canonical_solution) seeds mock_attempts
    for offline MockPolicy runs only — never written into the training row."""
    _ensure_cache()
    from datasets import load_dataset

    ds = load_dataset("openai/openai_humaneval", split=split)
    tasks: list[CodingTask] = []
    for ex in _slice(ds, limit, offset):
        entry = ex["entry_point"]
        tid = ex["task_id"].replace("/", "_").lower()  # "HumanEval/0" -> "humaneval_0"
        prompt = (
            "Complete the following Python function. Write the FULL solution.py "
            "(including the function definition):\n\n"
            f"```python\n{ex['prompt'].rstrip()}\n```"
        )
        reference = ex["prompt"] + ex["canonical_solution"]
        tasks.append(CodingTask(
            task_id=tid,
            prompt=prompt,
            checks=[f"check({entry})"],          # assert mode: raises on failure
            check_mode="assert",
            test_setup=ex["test"],               # defines check(candidate)
            max_turns=3,
            mock_attempts=[_HUMANEVAL_STUB, reference],
        ))
    return tasks


# ============================================================================
# Math — DeepScaleR (competition/olympiad math; numeric-answer subset)
# ============================================================================
def load_deepscaler(split: str = "train", limit: int | None = None, offset: int = 0) -> list[MathTask]:
    """agentica-org/DeepScaleR-Preview-Dataset. ~40k competition/olympiad problems used for
    RL math training. Only a `train` split exists, so eval is carved by offset/limit at build
    time. We keep the NUMERIC-ANSWER SUBSET (same `_parse_numeric` filter as MATH-500: ints,
    decimals, `\\frac{a}{b}`, `a/b`; surds/symbolic skipped) so every task has an exact,
    offline-verifiable numeric reward on the stdlib CPU sandbox."""
    _ensure_cache()
    from datasets import load_dataset

    ds = load_dataset("agentica-org/DeepScaleR-Preview-Dataset", split="train")
    tasks: list[MathTask] = []
    seen = 0
    for ex in ds:
        gold = _parse_numeric(ex.get("answer"))
        if gold is None:
            continue
        if seen < offset:
            seen += 1
            continue
        tasks.append(_math_task(f"deepscaler_{seen}", ex["problem"].strip(), gold,
                                tol=1e-4, max_turns=4))
        seen += 1
        if limit is not None and len(tasks) >= limit:
            break
    return tasks


# ============================================================================
# RAG — HotpotQA (multi-hop QA; gold + distractor passages as the ./docs corpus)
# ============================================================================
# SQuAD-style answer normalization (lowercase, drop articles + punctuation, collapse spaces),
# embedded in the checker `setup` so the exact-match reward is robust to trivial formatting.
_RAG_NORM_SETUP = (
    "import re as _re, string as _string\n"
    "def _norm(s):\n"
    "    s = str(s).lower().strip()\n"
    "    s = _re.sub(r'\\b(a|an|the)\\b', ' ', s)\n"
    "    s = ''.join(ch for ch in s if ch not in _string.punctuation)\n"
    "    return ' '.join(s.split())\n"
)


def load_hotpotqa(split: str = "validation", limit: int | None = None, offset: int = 0) -> list[RagTask]:
    """hotpotqa/hotpot_qa (config 'distractor'). Each row ships 10 passages (2 supporting +
    8 distractors) → written into ./docs as the retrieval corpus; the agent must read the docs
    to answer. Reward = SQuAD-normalized exact match of `answer` vs the gold answer."""
    _ensure_cache()
    from datasets import load_dataset

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    tasks: list[RagTask] = []
    for i, ex in enumerate(_slice(ds, limit, offset)):
        titles = ex["context"]["title"]
        sents = ex["context"]["sentences"]
        corpus = {
            f"doc{j:02d}.txt": f"{titles[j]}\n" + " ".join(sents[j]).strip()
            for j in range(len(titles))
        }
        gold = str(ex["answer"]).strip()
        # Put the gold answer in a GOLD variable in `setup` (safe repr, no nested-quote hazard),
        # then reference it from the check — embedding {gold!r} directly inside a single-quoted
        # f-string assert message breaks when gold itself contains quotes.
        setup = _RAG_NORM_SETUP + f"GOLD = {gold!r}\n"
        check = "assert _norm(answer) == _norm(GOLD), f'answer={answer!r} expected {GOLD!r}'"
        tasks.append(RagTask(
            task_id=f"hotpotqa_{offset + i}",
            prompt=ex["question"].strip(),
            max_turns=3,
            spec={"corpus": corpus, "checks": [check], "setup": setup},
            mock_attempts=["answer = 'unknown'\n", f"answer = {gold!r}\n"],
        ))
    return tasks


# ============================================================================
# Tool-call — xLAM function-calling (open parsed mirror), verifiable gold calls
# ============================================================================
import json as _json


def _xlam_gold(tool_calls: list) -> list[dict]:
    """messages[assistant].tool_calls -> [{name, arguments: dict}] (arguments parsed from the
    JSON-string the dataset stores)."""
    gold = []
    for tc in tool_calls or []:
        fn = tc.get("function", tc)
        args = fn.get("arguments", {})
        if isinstance(args, str):
            args = _json.loads(args) if args.strip() else {}
        gold.append({"name": fn.get("name"), "arguments": args})
    return gold


def load_xlam(split: str = "train", limit: int | None = None, offset: int = 0) -> list[ToolCallTask]:
    """minpeter/xlam-function-calling-60k-parsed — an openly-loadable parsed mirror of
    Salesforce/xLAM-function-calling-60k (the gated original). Each row: a user request, the
    available tool specs (JSON), and the gold assistant tool_calls. Reward = normalized exact
    match of the agent's emitted calls vs the gold calls. Only a `train` split exists, so eval
    is carved by offset/limit at build time."""
    _ensure_cache()
    from datasets import load_dataset

    ds = load_dataset("minpeter/xlam-function-calling-60k-parsed", split="train")
    tasks: list[ToolCallTask] = []
    seen = 0
    for ex in ds:
        msgs = ex.get("messages") or []
        user = next((m for m in msgs if m.get("role") == "user" and m.get("content")), None)
        asst = next((m for m in msgs if m.get("role") == "assistant" and m.get("tool_calls")), None)
        if not user or not asst:
            continue
        gold = _xlam_gold(asst["tool_calls"])
        if not gold or any(c["name"] is None for c in gold):
            continue
        if seen < offset:
            seen += 1
            continue
        tasks.append(ToolCallTask.build(
            task_id=f"xlam_{seen}",
            query=user["content"].strip(),
            tools_text=ex["tools"] if isinstance(ex["tools"], str) else _json.dumps(ex["tools"]),
            gold_calls=gold,
        ))
        seen += 1
        if limit is not None and len(tasks) >= limit:
            break
    return tasks


# ============================================================================
# MCP / tool-calling — Team-ACE/ToolACE (MCP-style JSON tool schemas + gold calls)
# ============================================================================
# "MCP tools" == JSON-schema tools a server exposes (name/description/inputSchema). ToolACE is a
# large tool-learning corpus whose system prompt embeds exactly that — a JSON array of tool schemas
# (name/description/parameters) — and whose first assistant turn is the gold tool call(s) in BFCL
# bracket syntax `[Func Name(k=v, ...), ...]`. We parse both and reuse the SAME verifiable ToolCallTask
# (agent sets a module-level `calls` list; reward = order-insensitive, arg-normalized match vs gold),
# so no harness/checker change — just a different tool-call source than xLAM. Single `train` split →
# eval carved by offset/limit.
import ast as _ast


def _parse_bracket_calls(text: str) -> list[dict] | None:
    """Parse BFCL '[Func Name(k=v, ...), ...]' into [{name, arguments}]; None if not a call list.

    Function names may contain spaces and arg values are Python literals (str/num/list/dict/bool),
    so we split top-level calls respecting quotes/brackets, then parse each call's args by feeding
    `f(<args>)` to ast and literal_eval-ing the keyword values.
    """
    s = (text or "").strip()
    if not (s.startswith("[") and s.endswith("]")) or "(" not in s:
        return None
    inner = s[1:-1].strip()
    if not inner:
        return None
    calls, depth, buf, quote = [], 0, [], None
    for ch in inner:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
            buf.append(ch)
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            calls.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        calls.append("".join(buf).strip())
    out: list[dict] = []
    for c in calls:
        m = re.fullmatch(r"\s*(.+?)\((.*)\)\s*", c, re.S)
        if not m:
            return None
        name, argstr = m.group(1).strip(), m.group(2).strip()
        args: dict = {}
        if argstr:
            try:
                node = _ast.parse("f(" + argstr + ")", mode="eval").body
                for kw in node.keywords:
                    if kw.arg is None:            # **kwargs splat — can't verify exactly
                        return None
                    args[kw.arg] = _ast.literal_eval(kw.value)
            except Exception:
                return None
        out.append({"name": name, "arguments": args})
    return out or None


def _extract_tools_json(system: str) -> str | None:
    """Pull the embedded JSON array of tool schemas out of the ToolACE system prompt (robust to
    ']' chars inside tool descriptions — uses JSONDecoder.raw_decode, not bracket matching)."""
    dec = _json.JSONDecoder()
    i = system.find("[")
    while i != -1:
        try:
            obj, end = dec.raw_decode(system[i:])
            if isinstance(obj, list) and obj and isinstance(obj[0], dict) and "name" in obj[0]:
                return system[i:i + end]
        except Exception:
            pass
        i = system.find("[", i + 1)
    return None


def load_mcp(split: str = "train", limit: int | None = None, offset: int = 0) -> list[ToolCallTask]:
    """Team-ACE/ToolACE — realistic API/tool corpus, treated as an MCP-style tool-calling task.
    Each row: MCP-style JSON tool schemas (in the system prompt) + a user request + the gold
    assistant tool call(s). Reward = normalized exact match of the agent's `calls` vs gold. Only a
    `train` split exists, so eval is carved by offset/limit at build time (~89% of rows are usable —
    those whose first assistant turn is a parseable bracket call and whose tools JSON extracts)."""
    _ensure_cache()
    from datasets import load_dataset

    ds = load_dataset("Team-ACE/ToolACE", split="train")
    tasks: list[ToolCallTask] = []
    seen = 0
    for ex in ds:
        conv = ex.get("conversations") or []
        if len(conv) < 2 or conv[0].get("from") != "user" or conv[1].get("from") != "assistant":
            continue
        gold = _parse_bracket_calls(conv[1].get("value"))
        if not gold:
            continue
        tools_text = _extract_tools_json(ex.get("system") or "")
        if not tools_text:
            continue
        if seen < offset:
            seen += 1
            continue
        tasks.append(ToolCallTask.build(
            task_id=f"toolace_{seen}",
            query=conv[0]["value"].strip(),
            tools_text=tools_text,
            gold_calls=gold,
        ))
        seen += 1
        if limit is not None and len(tasks) >= limit:
            break
    return tasks


# ============================================================================
# SQL — gretelai/synthetic_text_to_sql (self-contained: schema + rows + gold query)
# ============================================================================
# NOTE on the dataset choice: the user asked for Spider/BIRD, but neither ships its SQLite
# databases on HuggingFace (they live in the external Yale/BIRD zips), so an execution-accuracy
# reward can't be computed offline from the HF rows alone. gretelai/synthetic_text_to_sql is a
# large, widely-used real text-to-SQL corpus whose `sql_context` embeds the CREATE TABLE **and**
# INSERT statements — so we materialize the DB at build time, run the gold query for the expected
# result set, and the existing SqlTask in-memory-sqlite checker verifies the agent's query. Fully
# self-contained, stdlib `sqlite3`, no external DB files or Dockerfile change.
def _split_sql(context: str) -> tuple[list[str], list[str]]:
    """Split a ';'-separated SQL context into (schema CREATE stmts, INSERT stmts)."""
    schema, inserts = [], []
    for raw in (context or "").split(";"):
        s = raw.strip()
        if not s:
            continue
        head = s.lstrip().upper()
        if head.startswith("CREATE"):
            schema.append(s)
        elif head.startswith("INSERT"):
            inserts.append(s)
    return schema, inserts


def load_sql(split: str = "train", limit: int | None = None, offset: int = 0) -> list[SqlTask]:
    """gretelai/synthetic_text_to_sql. Materializes each row's embedded schema+rows in an
    in-memory sqlite, runs the gold query to capture the expected result set, and emits a
    SqlTask whose checker rebuilds the same DB and compares the agent's query's result set."""
    import sqlite3

    _ensure_cache()
    from datasets import load_dataset

    ds = load_dataset("gretelai/synthetic_text_to_sql", split=split)
    tasks: list[SqlTask] = []
    seen = 0
    for ex in ds:
        gold_sql = (ex.get("sql") or "").strip().rstrip(";").strip()
        if not gold_sql or ";" in gold_sql:          # need exactly one gold statement
            continue
        schema, inserts = _split_sql(ex.get("sql_context"))
        if not schema:
            continue
        # Materialize + run the gold query to get the expected result set (skip rows that error).
        try:
            con = sqlite3.connect(":memory:")
            cur = con.cursor()
            for stmt in schema + inserts:
                cur.execute(stmt)
            cur.execute(gold_sql)
            expected = [list(r) for r in cur.fetchall()]
            con.close()
        except Exception:
            continue
        if not expected:                              # empty result -> degenerate, skip
            continue
        if seen < offset:
            seen += 1
            continue
        order = "ORDER BY" in gold_sql.upper()
        tasks.append(SqlTask(
            task_id=f"gretelsql_{seen}",
            prompt=(ex.get("sql_prompt") or "").strip(),
            max_turns=3,
            spec={"schema": schema, "inserts": inserts,
                  "expected": expected, "order_matters": order},
            mock_attempts=["SELECT -987654321", gold_sql],   # wrong (sentinel) / reference
        ))
        seen += 1
        if limit is not None and len(tasks) >= limit:
            break
    return tasks


LOADERS = {
    "gsm8k": load_gsm8k,
    "math500": load_math500,
    "deepscaler": load_deepscaler,
    "humaneval": load_humaneval,
    "hotpotqa": load_hotpotqa,
    "xlam": load_xlam,
    "mcp": load_mcp,
    "sql": load_sql,
}
