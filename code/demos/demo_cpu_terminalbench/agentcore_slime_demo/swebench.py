"""SWE-bench Verified on the Terminal-Bench image pool (BENCH=swebench): task class + hidden grader.

Each SWE-bench instance ships a prebuilt x86_64 image (`swebench/sweb.eval.x86_64.<id>`, repo checked
out at `base_commit` in /testbed, conda env `testbed`). deploy/deploy_swebench.py wraps each one with
the same stdlib AgentCore HTTP server the Terminal-Bench images get and registers ONE Runtime per
instance on an m5 capacity provider (these images are 1-3 GB compressed and amd64-only, so they cannot
run on the arm64 <=2 GB microVM fleet). TerminalBenchPool routes, opens a fresh session per episode
and grades exactly as for Terminal-Bench; only `_grade` dispatches here for a SWEBenchTask.

Grading = the official SWE-bench recipe (swebench 3.0.x `make_eval_script_list_py` +
`get_logs_eval` + `get_eval_tests_report`), run through `sb.exec` AFTER the agent's turns:
  1. conda activate testbed; cd /testbed; the repo's `eval_commands` (e.g. django's locale exports)
  2. reset every file the test_patch touches to base_commit (the agent's edits to them are discarded;
     files the patch ADDS are deleted), then `git apply` the hidden test_patch (never in the image)
  3. run the repo/version `test_cmd` over the test_patch's test files, parse the log per repo
  4. solved iff every FAIL_TO_PASS and every PASS_TO_PASS test is PASSED/XFAIL (a test missing from
     the log or FAILED/ERROR is a failure; SKIPPED counts neither way, like the official grader).
Deviation from the official script (on purpose): the repo `install` step is NOT re-run. Every
supported repo except requests is installed editable (`pip install -e .`) so source edits are live;
re-running pip would need PyPI egress the capacity-provider instance may not have. requests (a
non-editable install) still works because pytest puts /testbed first on sys.path for its root-level
test_requests.py; swebench_oracle_check.py verifies gold->solved per instance, so any instance this
shortcut breaks is caught and dropped.

Supported repos (vendored parsers below, MIT-licensed SWE-bench logic, trimmed): pytest-style
(astropy, requests, pytest, xarray, sphinx, pylint, flask), django's runtests.py, sympy's bin/test.
matplotlib / scikit-learn / seaborn are excluded by fetch_swebench.py (large images, slow suites).
"""

from __future__ import annotations

import json
import os
import re
import shlex
from pathlib import Path
from typing import ClassVar

from .task import register_family
from .task_families import TerminalBenchTask

REPO_DIR = "/testbed"
START, END = ">>>>> Start Test Output", ">>>>> End Test Output"
_DEMO_DIR = Path(__file__).resolve().parent.parent


# ==============================================================================
# The task: a TerminalBenchTask whose "environment" is a SWE-bench repo snapshot
# ==============================================================================
@register_family
class SWEBenchTask(TerminalBenchTask):
    """One SWE-bench Verified instance. `spec.image` (the instance id) routes the episode to the
    instance's own Runtime; the prompt is the issue text; the reward is grade_swebench."""

    family: ClassVar[str] = "swebench"
    GUIDE: ClassVar[str] = (
        "You are fixing a real GitHub issue in the repository checked out at /testbed (your working "
        "directory). The project's Python environment is the conda env `testbed` (run `source "
        "/opt/miniconda3/bin/activate testbed` if `python` is not it). Edit the library SOURCE files so "
        "the issue is resolved; do NOT modify or add test files — hidden tests are applied after you "
        "finish and any edits to test files are discarded. You may write throwaway scripts to reproduce "
        "the bug. Keep the change minimal. When the fix is in place, stop."
    )
    SYSTEM: ClassVar[str] = TerminalBenchTask.SYSTEM + " " + GUIDE

    def initial_user_prompt(self) -> str:
        # TerminalBenchPool keeps ONE harness system prompt (the Terminal-Bench shell persona) for
        # every episode, so the SWE-specific guidance travels in the user prompt.
        return f"{self.GUIDE}\n\n<issue>\n{self.prompt.strip()}\n</issue>"


# ==============================================================================
# Log parsers (trimmed from swebench.harness.log_parsers.python, MIT)
# ==============================================================================
_STATUSES = ("FAILED", "PASSED", "SKIPPED", "ERROR", "XFAIL")


def parse_pytest(log: str) -> dict[str, str]:
    """`pytest -rA` short-summary lines: `PASSED path::test`, `FAILED path::test - msg`."""
    out: dict[str, str] = {}
    for line in log.split("\n"):
        if line.startswith(_STATUSES):
            if line.startswith("FAILED"):
                line = line.replace(" - ", " ")
            parts = line.split()
            if len(parts) > 1:
                out[parts[1]] = parts[0]
    return out


def parse_pytest_options(log: str) -> dict[str, str]:
    """parse_pytest + SWE-bench's normalisation of path-like parametrize ids (requests, pylint)."""
    opt = re.compile(r"(.*?)\[(.*)\]")
    out: dict[str, str] = {}
    for line in log.split("\n"):
        if line.startswith(_STATUSES):
            if line.startswith("FAILED"):
                line = line.replace(" - ", " ")
            parts = line.split()
            if len(parts) <= 1:
                continue
            name, m = parts[1], opt.search(parts[1])
            if m:
                main, option = m.groups()
                if option.startswith("/") and not option.startswith("//") and "*" not in option:
                    option = "/" + option.split("/")[-1]
                name = f"{main}[{option}]"
            out[name] = parts[0]
    return out


def parse_pytest_v2(log: str) -> dict[str, str]:
    """ANSI-stripped summary lines, plus old-pytest `path::test STATUS` lines (astropy, sphinx)."""
    out: dict[str, str] = {}
    esc = str.maketrans("", "", "".join(chr(c) for c in range(1, 32)))
    for line in log.split("\n"):
        line = re.sub(r"\[(\d+)m", "", line).translate(esc)
        if line.startswith(_STATUSES):
            if line.startswith("FAILED"):
                line = line.replace(" - ", " ")
            parts = line.split()
            if len(parts) >= 2:
                out[parts[1]] = parts[0]
        elif line.endswith(_STATUSES):
            parts = line.split()
            if len(parts) >= 2:
                out[parts[0]] = parts[1]
    return out


def parse_django(log: str) -> dict[str, str]:
    """runtests.py --verbosity 2: `name (mod.Class) ... ok|FAIL|ERROR|skipped`, docstring variants."""
    out: dict[str, str] = {}
    prev = None
    for line in log.split("\n"):
        line = line.strip()
        if "--version is equivalent to version" in line:
            out["--version is equivalent to version"] = "PASSED"
        if " ... " in line:
            prev = line.split(" ... ")[0]
        for suf in (" ... ok", " ... OK", " ...  OK"):
            if line.endswith(suf):
                if line.startswith("Applying sites.0002_alter_domain_unique...test_no_migrations"):
                    line = line.split("...", 1)[-1].strip()
                out[line.rsplit(suf, 1)[0]] = "PASSED"
                break
        if " ... skipped" in line:
            out[line.split(" ... skipped")[0]] = "SKIPPED"
        if line.endswith(" ... FAIL"):
            out[line.split(" ... FAIL")[0]] = "FAILED"
        if line.startswith("FAIL:") and len(line.split()) > 1:
            out[line.split()[1].strip()] = "FAILED"
        if line.endswith(" ... ERROR"):
            out[line.split(" ... ERROR")[0]] = "ERROR"
        if line.startswith("ERROR:") and len(line.split()) > 1:
            out[line.split()[1].strip()] = "ERROR"
        if line.lstrip().startswith("ok") and prev is not None:
            out[prev] = "PASSED"
    for pat in (r"^(.*?)\s\.\.\.\sTesting\ against\ Django\ installed\ in\ ((?s:.*?))\ silenced\)\.\nok$",
                r"^(.*?)\s\.\.\.\sInternal\ Server\ Error:\ \/(.*)\/\nok$",
                r"^(.*?)\s\.\.\.\sSystem check identified no issues \(0 silenced\)\nok$"):
        for m in re.finditer(pat, log, re.MULTILINE):
            out[m.group(1)] = "PASSED"
    return out


def parse_sympy(log: str) -> dict[str, str]:
    """bin/test -C --verbose: `test_name ok|F|E`, plus `___ path.py:test ___` failure headers."""
    out: dict[str, str] = {}
    for m in re.findall(r"(_*) (.*)\.py:(.*) (_*)", log):
        out[f"{m[1]}.py:{m[2]}"] = "FAILED"
    for line in log.split("\n"):
        line = line.strip()
        if line.startswith("test_"):
            if line.endswith(" E"):
                out[line.split()[0]] = "ERROR"
            if line.endswith(" F"):
                out[line.split()[0]] = "FAILED"
            if line.endswith(" ok"):
                out[line.split()[0]] = "PASSED"
    return out


# Official repo -> parser map (swebench MAP_REPO_TO_PARSER_PY), restricted to the supported repos.
PARSER_BY_REPO = {
    "astropy/astropy": "pytest_v2",
    "django/django": "django",
    "pallets/flask": "pytest",
    "psf/requests": "pytest_options",
    "pydata/xarray": "pytest",
    "pylint-dev/pylint": "pytest_options",
    "pytest-dev/pytest": "pytest",
    "sphinx-doc/sphinx": "pytest_v2",
    "sympy/sympy": "sympy",
}
PARSERS = {"pytest": parse_pytest, "pytest_options": parse_pytest_options,
           "pytest_v2": parse_pytest_v2, "django": parse_django, "sympy": parse_sympy}


def parse_log(log: str, parser: str) -> dict[str, str] | None:
    """Status map of the section between the START/END markers (None if the tests never ran)."""
    if START not in log or END not in log:
        return None
    return PARSERS[parser](log.split(START, 1)[1].split(END, 1)[0])


def evaluate(status: dict[str, str] | None, f2p: list[str], p2p: list[str]) -> dict:
    """Official resolution rule: F2P and P2P success rates both 1.0 (SKIPPED ignored)."""
    status = status or {}
    ok = lambda t: status.get(t) in ("PASSED", "XFAIL")                       # noqa: E731
    bad = lambda t: t not in status or status[t] in ("FAILED", "ERROR")       # noqa: E731
    f_ok, f_bad = sum(map(ok, f2p)), sum(map(bad, f2p))
    p_ok, p_bad = sum(map(ok, p2p)), sum(map(bad, p2p))
    return {"f2p_passed": f_ok, "f2p_failed": f_bad, "p2p_passed": p_ok, "p2p_failed": p_bad,
            "solved": f_bad == 0 and p_bad == 0 and bool(f2p),
            "tests_passed": f_ok + p_ok, "tests_total": f_ok + f_bad + p_ok + p_bad}


# ==============================================================================
# The grader
# ==============================================================================
def eval_script(inst: dict, patch_path: str) -> str:
    """The bash that resets test files, applies the hidden test_patch and runs the tests."""
    base = inst["base_commit"]
    q = shlex.quote
    lines = ["set +e", "source /opt/miniconda3/bin/activate", "conda activate testbed", f"cd {REPO_DIR}"]
    lines += list(inst.get("eval_commands") or [])
    lines += [f"git config --global --add safe.directory {REPO_DIR}", f"cd {REPO_DIR}",
              "source /opt/miniconda3/bin/activate", "conda activate testbed"]
    # Reset every file the test_patch touches (official: `git checkout <base> <files>`); a file the
    # patch ADDS does not exist at base, so remove it instead of letting the whole checkout fail.
    for f in inst["test_files"]:
        lines.append(f"if git cat-file -e {base}:{q(f)} 2>/dev/null; then git checkout -q {base} -- {q(f)}; "
                     f"else rm -rf -- {q(f)}; fi")
    lines += [f"git apply -v {q(patch_path)} || {{ echo 'ACS_TEST_PATCH_APPLY_FAILED'; exit 3; }}",
              f"echo {q(START)}", inst["test_cmd"], f"echo {q(END)}"]
    return "\n".join(lines) + "\n"


async def grade_swebench(sb, instance: dict, timeout: float = 900.0) -> dict:
    """Apply the hidden tests after the agent, run FAIL_TO_PASS/PASS_TO_PASS, return the
    grade_episode dict shape (solved, grade_exit, tests_passed, tests_total, grade_timeout, grade_tail)
    plus the SWE-bench split (f2p_*/p2p_*)."""
    gdir = "/tmp/.acs_swe_grade"
    await sb.write_file(f"{gdir}/test.patch", instance["test_patch"])
    await sb.write_file(f"{gdir}/eval.sh", eval_script(instance, f"{gdir}/test.patch"))
    # The full log goes to a file and comes back whole when it is small; a huge log (thousands of
    # verbose tests) is cut down to the lines the parsers read, to bound the invoke payload.
    keep = (r"^\s*(PASSED|FAILED|SKIPPED|ERROR|XFAIL|FAIL:|ok|test_|_+ |>>>>> |ACS_)|( \.\.\. )"
            r"|(PASSED|FAILED|SKIPPED|ERROR|XFAIL)\s*$")
    log = f"{gdir}/eval.log"
    cmd = (f"timeout {int(timeout)} bash {gdir}/eval.sh > {log} 2>&1; rc=$?; "
           f"if [ $(stat -c %s {log}) -le 1000000 ]; then cat {log}; "
           f"else grep -aE {shlex.quote(keep)} {log} | head -c 2000000; fi; "
           f"echo; echo '===ACS_TAIL==='; tail -c 1500 {log}; exit $rc")
    code, out, err = await sb.exec(cmd, timeout=int(timeout) + 120)
    body, _, tail = out.partition("===ACS_TAIL===")
    status = parse_log(body, instance.get("parser") or PARSER_BY_REPO[instance["repo"]])
    res = evaluate(status, instance["FAIL_TO_PASS"], instance["PASS_TO_PASS"])
    res.update(grade_exit=int(code), grade_timeout=int(code == 124),
               patch_applied=int("ACS_TEST_PATCH_APPLY_FAILED" not in body),
               grade_tail=(tail + err)[-600:])
    return res


# ==============================================================================
# Trainer-side manifest (the hidden test_patch never enters the image or the dataset row)
# ==============================================================================
_MANIFEST: dict[str, dict] | None = None


def load_instance(instance_id: str) -> dict:
    global _MANIFEST
    if _MANIFEST is None:
        path = os.environ.get("SWE_MANIFEST", str(_DEMO_DIR / "swebench" / "manifest.json"))
        _MANIFEST = {e["task"]: e for e in json.loads(Path(path).read_text())["instances"]}
    return _MANIFEST[instance_id]
