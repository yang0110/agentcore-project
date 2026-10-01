"""The harness: the agent loop that ties policy + sandbox + task together.

This plays the role of Agent Lightning's *Rollout Controller* + the agent harness: it runs
the agent, keeps tools/environment in the loop, and records every interaction as training
data. Crucially, the harness is decoupled from training — it only produces a Trajectory of
transitions; converting those into slime training samples happens in generate.py.

Multi-turn loop (one episode = one task):
    turn 0: policy sees system+task  -> emits solution.py -> sandbox runs checker -> observation
    turn 1: policy sees observation  -> fixes solution.py -> sandbox runs checker -> observation
    ... until checker passes or max_turns reached.

Each turn is one MDP transition: (prompt so far) --action(assistant tokens)--> (observation).
Only the assistant tokens are trainable; observation/tool tokens are masked downstream. This
is the "unified data interface" the AL report builds credit assignment on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .policy import Completion, Message, Policy
from .sandbox import Sandbox
from .task import CodingTask

# Leading-word outcome tokens the hidden checkers print (task.py assert-runner, task_families.py
# SQL runner). Compressed into ONE space-free `checker=` class for the [demo] metric channel so the
# per-demo split — compile-vs-assert (humaneval/mbpp), numeric-extraction-vs-wrong (math families,
# FAIL_AssertionError vs FAIL_NameError), exec-match-vs-syntax-error (sql, QUERY_ERROR vs FAIL),
# valid-call-vs-correct (xlam/mcp) — falls out of one generic parser (see metrics.py [demo] channel).
_CHECK_CLASSES = ("IMPORT_ERROR", "SETUP_ERROR", "DB_SETUP_ERROR", "QUERY_ERROR",
                  "NO_QUERY", "EMPTY_QUERY", "ONE_STATEMENT_ONLY")


def _checker_token(stdout: str) -> tuple[str, dict[str, str]]:
    """Compress a checker's stdout into one space-free outcome token (+ optional pass_n/pass_m)."""
    text = stdout or ""
    if not any(ln.strip() for ln in text.splitlines()):
        return "NO_OUTPUT", {}
    if "PASSED" in text:
        m = re.search(r"PASSED\s+(\d+)\s*/\s*(\d+)", text)
        return "PASSED", ({"pass_n": m.group(1), "pass_m": m.group(2)} if m else {})
    for cls in _CHECK_CLASSES:
        if cls in text:
            return cls, {}
    m = re.search(r"->\s*([A-Za-z_][A-Za-z0-9_]*)", text)   # "FAIL[0]: … -> ExcType"
    if "FAIL" in text and m:
        return "FAIL_" + m.group(1), {}     # e.g. FAIL_AssertionError vs FAIL_NameError
    if "FAIL" in text:
        return "FAIL", {}
    return "OTHER", {}


@dataclass
class Turn:
    """One MDP transition."""
    prompt_messages: list[Message]      # full context the policy saw (state)
    completion: Completion              # the action (assistant tokens — trainable)
    observation: str                    # environment feedback (masked downstream)
    exit_code: int                      # checker exit code this turn
    submitted_code: bool                # did the agent produce a code block?


@dataclass
class Trajectory:
    task_id: str
    turns: list[Turn] = field(default_factory=list)
    reward: float = 0.0
    solved: bool = False
    # Recovery bookkeeping (see recovery.py). status ∈ {"completed", "aborted"}.
    status: str = "completed"
    abort_reason: str = ""
    boot_attempts: int = 1   # how many sandbox boots this episode consumed

    @property
    def num_turns(self) -> int:
        return len(self.turns)


async def run_episode(
    policy: Policy,
    sandbox: Sandbox,
    task: CodingTask,
    sampling_params: dict | None = None,
) -> Trajectory:
    """Run one multi-turn episode inside an already-open sandbox."""
    sampling_params = sampling_params or {"max_new_tokens": 1024, "temperature": 1.0}
    traj = Trajectory(task_id=task.task_id)

    # Family-agnostic hooks (CodingTask supplies the coding defaults; other families in
    # task_families.py override them). getattr keeps this robust to minimal task objects.
    artifact_filename = getattr(task, "artifact_filename", "solution.py")
    check_command = getattr(task, "check_command", "python3 run_checks.py")
    submission_hint = getattr(
        task, "submission_hint",
        "Reply with the full solution.py in one ```python code block.",
    )

    # Seed the workspace once: any family-specific files (tools module, doc corpus, ...),
    # then the checker. All are stable across turns.
    for name, content in getattr(task, "setup_files", lambda: {})().items():
        await sandbox.write_file(name, content)
    await sandbox.write_file("run_checks.py", task.runner_source())

    messages: list[Message] = [
        {"role": "system", "content": task.system_prompt()},
        {"role": "user", "content": task.initial_user_prompt()},
    ]

    last_stdout = ""   # last checker stdout, for the per-episode [demo] outcome token
    for _ in range(task.max_turns):
        completion = await policy.complete(messages, sampling_params)
        messages.append({"role": "assistant", "content": completion.text})

        code = task.extract_submission(completion.text)
        if code is None:
            # No submission this turn — nudge and record a "no submission" transition.
            observation = f"No code block found. {submission_hint}"
            exit_code, submitted = 1, False
        else:
            await sandbox.write_file(artifact_filename, code)
            exit_code, stdout, stderr = await sandbox.exec(check_command, timeout=60)
            observation = task.observation(exit_code, stdout, stderr)
            last_stdout = stdout
            submitted = True

        traj.turns.append(Turn(
            prompt_messages=list(messages[:-1]),   # state before this action
            completion=completion,
            observation=observation,
            exit_code=exit_code,
            submitted_code=submitted,
        ))

        if submitted and exit_code == 0:
            traj.solved = True
            break

        messages.append({"role": "user", "content": observation})

    traj.reward = task.reward(0 if traj.solved else 1)

    # Per-episode DEMO-SPECIFIC metric line (parsed by metrics.py's generic [demo] channel). One
    # emit here instruments EVERY run_episode-based demo: `family` gives mixed-task / libs
    # per-family reward, `image` gives per-image routing+correctness, `checker` gives each
    # dataset demo's outcome split (compile-vs-assert, numeric-vs-wrong, exec-match-vs-syntax-error,
    # valid-call-vs-correct), and boot_attempts/status give custom-Runtime rollout health. Numeric
    # keys are aggregated per step; categorical keys (family/image/checker) get run-level value-counts
    # + mean-reward-per-category. Values are space-free so the `k=v` parser stays simple.
    tok, extras = _checker_token(last_stdout)
    fields = {
        "task": str(task.task_id).replace(" ", "_"),
        "family": str(getattr(task, "family", "?")).replace(" ", "_"),
        "reward": traj.reward,
        "solved": int(traj.solved),
        "turns": traj.num_turns,
        "boot_attempts": traj.boot_attempts,
        "status": str(traj.status).replace(" ", "_"),
        "checker": tok,
    }
    img = getattr(task, "image", "")
    if img:
        fields["image"] = str(img).replace(" ", "_")
    fields.update(extras)
    print("[demo] " + " ".join(f"{k}={v}" for k, v in fields.items()), flush=True)
    return traj
