"""One mini_claude episode (the only harness), graded by a HIDDEN checker.

The agent loop runs INSIDE the sandbox session (in_session_harness.py), dialling this process's
adapter back; the trainer only hosts the adapter and the token record. There is no in-trainer
loop and no HARNESS_WHERE=trainer / HARNESS=direct (removed 2026-09-28).

This is the tool-calling agent every demo uses: the policy talks Anthropic Messages to slime's
AnthropicAdapter (one per SGLang engine, so an episode stays pinned to one engine) and drives
write_file / bash / read_file in the sandbox for as many turns as it wants.

Reward hacking: the checker is NOT in the sandbox while the agent runs (it could otherwise be
read or rewritten). After the agent stops, the task's setup files are re-seeded (undoing any
tampering), the checker is written under a random name, and its exit code is the reward — the
Terminal-Bench grade_episode pattern. If the agent never wrote the artifact file, a fenced block
in its final reply is accepted as the submission (the task's extract_submission hook).

Returns ONE bare Sample: finish_session linearises the adapter's per-session tree into a list,
which is a single sample for this append-only loop; if it ever yields more we keep the longest
(and record fan_out) so the fully-async rollout (MODE=B) keeps slime's non-fan-out shape.
"""

from __future__ import annotations

import os
import secrets

from .harness_pool import adapter_for, one_sample
from .in_session_harness import make_harness

_PROMPT_TAIL = (
    "\n\nYou are in a sandbox working directory with the tools write_file, bash and read_file. "
    "Write your final answer to the file `{artifact}` with write_file; you may run it with bash to "
    "test it. There is no checker in the sandbox: your file is graded by hidden tests after you "
    "stop. When you are done, reply with a one-line summary and no tool call."
)


def _final_text(transcript: list[dict]) -> str:
    for msg in reversed(transcript or []):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        return "\n".join(b.get("text", "") for b in content or [] if isinstance(b, dict))
    return ""


def _nonce_runner(src: str) -> tuple[str, bool]:
    """Make the checker's PASSED line carry a per-grade nonce the solution never sees.

    The checker imports the agent's solution in-process, so a solution can `sys.exit(0)` (or print
    "PASSED") at import time. The nonce arrives in GRADE_NONCE and is popped from the environment
    BEFORE the solution loads; only the checker's own PASSED print emits it. Residual hole: a
    solution that introspects sys.modules["__main__"]. Unknown templates -> exit code only."""
    marked = src
    for old, new in (('print(f"PASSED ', 'print(_GN, f"PASSED '), ('print("PASSED", ', 'print(_GN, "PASSED", ')):
        marked = marked.replace(old, new)
    if marked == src:
        return src, False
    return "import os as _gos\n_GN = _gos.environ.pop('GRADE_NONCE', '')\n" + marked, True


async def _hidden_grade(sb, task) -> tuple[int, str]:
    """Re-seed the task files, drop the checker in under a random name, run it once.

    Returns (0, stdout) only if the checker exited 0 AND printed its nonce'd PASSED line."""
    for name, content in getattr(task, "setup_files", lambda: {})().items():
        await sb.write_file(name, content)
    grader = f".grade_{secrets.token_hex(6)}.py"
    nonce = secrets.token_hex(8)
    src, marked = _nonce_runner(task.runner_source())
    await sb.write_file(grader, src)
    code, out, _err = await sb.exec(f"GRADE_NONCE={nonce} python3 {grader}",
                                    timeout=int(os.environ.get("MINI_GRADE_TIMEOUT", "120")))
    try:
        await sb.exec(f"rm -f {grader}", timeout=30)
    except Exception:  # noqa: BLE001 — cleanup only; the sandbox is closed right after
        pass
    if code == 0 and marked and not any(ln.startswith(nonce + " PASSED") for ln in (out or "").splitlines()):
        code = 1   # exited 0 without the checker's own PASSED line: the solution short-circuited it
    return code, (out or "").replace(nonce + " ", "")


async def run(args, sample, task, sampling_params, sb, engine: str, fields: dict | None = None):
    """Run one mini_claude episode in the already-open sandbox ``sb``; return a bare Sample.

    ``fields``: extra per-episode keys for the [demo] record (the harness pool's tier fields)."""
    from .harness import _checker_token
    adapter, _local_url, session_url = adapter_for(args, engine)
    artifact = getattr(task, "artifact_filename", "solution.py")

    sid = getattr(sample, "session_id", None)
    if not sid or len(sid) < 33:   # the adapter keys trajectories by a >=33-char Bearer id
        sid = f"mc-{secrets.token_hex(16)}-{getattr(sample, 'index', 0)}"
    sample.session_id = sid

    # The task's own files (tools module, corpus, db...) — but NOT the checker.
    for name, content in getattr(task, "setup_files", lambda: {})().items():
        await sb.write_file(name, content)

    harness = make_harness(
        max_turns=int(os.environ.get("MINI_AGENT_MAX_TURNS", str(max(4, 2 * task.max_turns)))),
        exec_timeout_sec=int(os.environ.get("MINI_AGENT_EXEC_TIMEOUT", "120")),
        system_prompt=("You are an agent working in a sandbox. Solve the task using ONLY the "
                       "provided tools; keep replies terse and prefer acting over explaining."),
    )
    adapter.open_session(sid, sampling_defaults=sampling_params)
    try:
        res = await harness.run(
            sb, session_id=sid, adapter_url=session_url,
            prompt=task.initial_user_prompt() + _PROMPT_TAIL.format(artifact=artifact),
            max_tokens=int((sampling_params or {}).get("max_new_tokens", 1024)),
        )
        # Fallback: no artifact file written, but a fenced answer in the final reply.
        wrote = any(tu.get("name") == "write_file" and str((tu.get("input") or {}).get("path", "")).strip().removeprefix("./") == artifact
                    for m in res.transcript if m.get("role") == "assistant" and isinstance(m.get("content"), list)
                    for tu in m["content"] if isinstance(tu, dict) and tu.get("type") == "tool_use")
        if not wrote:
            sub = task.extract_submission(_final_text(res.transcript))
            if sub is not None:
                await sb.write_file(artifact, sub)
        exit_code, stdout = await _hidden_grade(sb, task)
        solved = exit_code == 0
        reward = float(task.reward(0 if solved else 1))

        samples = await adapter.finish_session(
            sid, base_sample=sample, reward=reward,
            extra_metadata={"task_id": task.task_id, "solved": solved, "num_turns": res.turns,
                            "tool_calls": res.tool_calls, "sandbox_id": getattr(sb, "sandbox_id", "?")},
        )
    finally:
        await adapter.drop_session(sid, wait_timeout=30)

    out, fan_out = one_sample(samples, reward)

    tok, extras = _checker_token(stdout)
    fields_in, fields = fields or {}, {
        "task": str(task.task_id).replace(" ", "_"),
        "family": str(getattr(task, "family", "?")).replace(" ", "_"),
        "reward": reward,
        "solved": int(solved),
        "turns": res.turns,
        "boot_attempts": 1,
        "status": "completed",
        "checker": tok,
        "harness": "mini_claude",
        "harness_where": res.where,
        "tool_calls": res.tool_calls,
        "stop": res.stop_reason,
        "wrote_artifact": int(wrote),
        "fan_out": fan_out,
    }
    # Substrate-specific per-episode fields a sandbox may expose (e.g. the warm instance pool's
    # pool_slot / instance / open_ms / cold). Looks through a KeptSandbox (HEARTBEAT=1) wrapper.
    for obj in (sb, getattr(sb, "_inner", None)):
        extra = getattr(obj, "demo_fields", None)
        if isinstance(extra, dict) and extra:
            fields.update({k: str(v).replace(" ", "_") for k, v in extra.items()})
            break
    img = getattr(task, "image", "")
    if img:
        fields["image"] = str(img).replace(" ", "_")
    fields.update(extras)
    fields.update({k: str(v).replace(" ", "_") for k, v in fields_in.items()})
    print("[demo] " + " ".join(f"{k}={v}" for k, v in fields.items()), flush=True)
    return out
