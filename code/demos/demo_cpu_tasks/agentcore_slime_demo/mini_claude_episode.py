"""HARNESS=mini_claude (default): one episode driven by MiniClaudeHarness, graded by a HIDDEN checker.

Where the loop runs is HARNESS_WHERE (in_session_harness.py): `session` (default on own Runtimes,
farms and local) runs it INSIDE the sandbox session, dialling this process's adapter back;
`trainer` (default on Code Interpreter, which has no route back) runs it here.

The direct harness (harness.run_episode, HARNESS=direct) extracts a fenced code block from each
reply and runs the visible run_checks.py for the model. This path is the tool-calling agent the
terminalbench / scale_lanes demos already use: the policy talks Anthropic Messages to slime's
AnthropicAdapter (one per SGLang engine, so an episode stays pinned to one engine) and drives
write_file / bash / read_file in the sandbox for as many turns as it wants.

Reward hacking: the checker is NOT in the sandbox while the agent runs (it could otherwise be
read or rewritten). After the agent stops, the task's setup files are re-seeded (undoing any
tampering), the checker is written under a random name, and its exit code is the reward — the
Terminal-Bench grade_episode pattern. If the agent never wrote the artifact file, a fenced block
in its final reply is accepted as the submission (same extract_submission hook as HARNESS=direct).

Returns ONE bare Sample: finish_session linearises the adapter's per-session tree into a list,
which is a single sample for this append-only loop; if it ever yields more we keep the longest
(and record fan_out) so the fully-async rollout (MODE=B) keeps slime's non-fan-out shape.
"""

from __future__ import annotations

import asyncio
import os
import secrets

from .in_session_harness import adapter_bind, adapter_urls, harness_where, make_harness

_ADAPTERS: dict[str, tuple[object, str, str]] = {}   # engine url -> (adapter, trainer url, session url)
_LOCK: asyncio.Lock | None = None

_PROMPT_TAIL = (
    "\n\nYou are in a sandbox working directory with the tools write_file, bash and read_file. "
    "Write your final answer to the file `{artifact}` with write_file; you may run it with bash to "
    "test it. There is no checker in the sandbox: your file is graded by hidden tests after you "
    "stop. When you are done, reply with a one-line summary and no tool call."
)


async def _adapter_for(args, engine: str) -> tuple[object, str, str]:
    """Lazily start one AnthropicAdapter per engine (shared by every episode pinned to it).

    Returns (adapter, url for the trainer, url the sandbox session dials back to)."""
    global _LOCK
    if _LOCK is None:
        _LOCK = asyncio.Lock()
    async with _LOCK:
        if engine not in _ADAPTERS:
            from slime.agent.adapters import AnthropicAdapter
            from slime.agent.aiohttp_threaded import run_app_in_thread
            from slime.utils.processing_utils import load_tokenizer

            adapter = AnthropicAdapter(
                tokenizer=load_tokenizer(args.hf_checkpoint, trust_remote_code=True),
                sglang_url=engine,
                tool_parser=getattr(args, "sglang_tool_call_parser", None) or None,
                reasoning_parser=getattr(args, "sglang_reasoning_parser", None) or None,
            )
            host, port = adapter_bind(len(_ADAPTERS))
            handle = run_app_in_thread(adapter.app, host=host, port=port,
                                       thread_name=f"anthropic-adapter-{len(_ADAPTERS)}")
            _ADAPTERS[engine] = (adapter, *adapter_urls(handle.port))
        return _ADAPTERS[engine]


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


async def run(args, sample, task, sampling_params, sb, engine: str):
    """Run one mini_claude episode in the already-open sandbox ``sb``; return a bare Sample."""
    from .harness import _checker_token
    adapter, local_url, session_url = await _adapter_for(args, engine)
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
            sb, session_id=sid, adapter_url=session_url if harness_where() == "session" else local_url,
            local_adapter_url=local_url,
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

    samples = samples if isinstance(samples, list) else [samples]
    out = max(samples, key=lambda s: len(getattr(s, "tokens", []) or []))
    out.reward = reward

    tok, extras = _checker_token(stdout)
    fields = {
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
        "fan_out": len(samples),
    }
    img = getattr(task, "image", "")
    if img:
        fields["image"] = str(img).replace(" ", "_")
    fields.update(extras)
    print("[demo] " + " ".join(f"{k}={v}" for k, v in fields.items()), flush=True)
    return out
