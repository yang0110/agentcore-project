"""A minimal, Claude-Code-like coding agent — the harness for Demos A3 / B3.

This is the "build a simple harness that mimics Claude Code" path. Where Demo A2/B2 run the
*real* Claude Code CLI (a Node binary) inside the sandbox, this is a ~1-file pure-Python agent
that reproduces Claude Code's essential loop and nothing else:

    call the model  ->  it picks a tool  ->  run the tool in the sandbox  ->  feed the
    result back  ->  repeat until the model stops (or a turn/time budget is hit).

That loop is exactly the "agent harness" AWS describes in the AgentCore harness docs
(https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/harness.html): an orchestration
layer that calls the model, picks tools, passes results back, and manages the run — here backed
by an AgentCore Runtime microVM as the sandbox (its own shell + filesystem).

Why it plugs into slime RL for free (Agent Lightning decoupling): the agent is just an HTTP
client of slime's `AnthropicAdapter`. It speaks the Anthropic Messages API (`POST /v1/messages`
with `tools`), and the adapter — sitting in front of SGLang — serves every completion from the
policy under training and records each turn into a per-session `TrajectoryManager`. So this file
contains ZERO training code: it only has to (a) route every model call through the adapter and
(b) tag the session so the adapter can key the trajectory. The session id travels in the
`Authorization: Bearer <session_id>` header, which is where the adapter reads it
(`sid_from_bearer`). The generate hook's `finish_session` (agentcore_runtime_generate) then drains
that trajectory into loss-masked training samples.

Deliberately NOT here (that is the point — "simple"): no CLI, no Node, no MCP, no subagents, no
context compaction, no permissions system. Three tools (write_file / bash / read_file) and a
flat message list. Because the agent runs in the trainer process (not in the sandbox), the
sandbox needs no Node/CLI install — the vanilla AgentCore Code Interpreter (A3) or GPU Runtime
(B3) is enough, unlike the custom multi-user container A2/B2 require.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Cap tool-result text fed back to the model so a chatty command can't blow up the context
# (and the trajectory) unboundedly. Head+tail keep the useful ends of a long traceback.
_MAX_TOOL_OUTPUT_CHARS = 4000

# The agent's persona + operating instructions. Kept short on purpose; this is the whole
# "prompt engineering" of the mimic. It mirrors Claude Code's coding-agent framing (work in a
# sandbox, prefer acting over explaining, iterate on test output) without any of its machinery.
SYSTEM_PROMPT = (
    "You are a coding agent working in a sandboxed working directory. You solve the user's task "
    "by editing files and running commands using ONLY the provided tools. Workflow: write your "
    "solution to a file named solution.py, then run `python3 run_checks.py` and read its output. "
    "If any check fails, revise solution.py and run the checks again. Repeat until the checks "
    "pass (the command exits with code 0). Keep replies terse and prefer taking a tool action "
    "over explaining. When the checks pass, stop."
)

# Anthropic-style tool schemas advertised to the model each turn. The adapter converts these to
# the tokenizer's chat-template tool schema, so tool-calling is driven by the served policy.
TOOLS: list[dict[str, Any]] = [
    {
        "name": "write_file",
        "description": "Create or overwrite a file with the given UTF-8 text content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path, relative to the working directory."},
                "content": {"type": "string", "description": "Full file contents to write."},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "bash",
        "description": "Run a bash command in the working directory; returns its stdout, stderr and exit code.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The bash command to run."}},
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read and return the UTF-8 text content of a file.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "File path, relative to the working directory."}},
            "required": ["path"],
        },
    },
]


def _clip(text: str) -> str:
    """Clip tool output to a bounded head+tail so long logs don't balloon the context."""
    if len(text) <= _MAX_TOOL_OUTPUT_CHARS:
        return text
    half = _MAX_TOOL_OUTPUT_CHARS // 2
    return f"{text[:half]}\n... [truncated {len(text) - _MAX_TOOL_OUTPUT_CHARS} chars] ...\n{text[-half:]}"


@dataclass
class MiniAgentResult:
    """Outcome of one agent run (for logging / metadata; the trajectory lives in the adapter)."""

    turns: int = 0
    stop_reason: str = "none"        # last wire stop_reason ("end_turn"/"tool_use"/"max_tokens") or "budget"
    tool_calls: int = 0
    last_exit_code: int | None = None  # exit code of the last bash command, if any
    transcript: list[dict] = field(default_factory=list)  # the raw message list, for debugging


class MiniClaudeHarness:
    """Drives the write->run->fix loop by talking Anthropic Messages to slime's adapter.

    Stateless across runs; hold one instance and call ``run`` per episode. Only depends on
    aiohttp (imported lazily) and the sandbox ``Sandbox`` contract — no slime import, so it is
    unit-testable against any HTTP endpoint that speaks the Anthropic Messages API.
    """

    name = "mini_claude"

    def __init__(self, *, max_turns: int = 12, request_timeout_sec: int = 300,
                 exec_timeout_sec: int = 120) -> None:
        self.max_turns = max_turns
        self.request_timeout_sec = request_timeout_sec
        self.exec_timeout_sec = exec_timeout_sec

    async def _call_tool(self, sb, name: str, tool_input: dict, result: MiniAgentResult) -> str:
        """Execute one tool call in the sandbox and return its text result for the model."""
        try:
            if name == "write_file":
                path = str(tool_input.get("path", "")).strip()
                if not path:
                    return "error: write_file requires a non-empty 'path'"
                await sb.write_file(path, str(tool_input.get("content", "")))
                return f"wrote {path}"
            if name == "read_file":
                path = str(tool_input.get("path", "")).strip()
                if not path:
                    return "error: read_file requires a non-empty 'path'"
                return _clip(await sb.read_file(path))
            if name == "bash":
                command = str(tool_input.get("command", "")).strip()
                if not command:
                    return "error: bash requires a non-empty 'command'"
                code, out, err = await sb.exec(command, timeout=self.exec_timeout_sec)
                result.last_exit_code = code
                body = f"exit_code: {code}\n--- stdout ---\n{out}\n--- stderr ---\n{err}"
                return _clip(body)
            return f"error: unknown tool {name!r}"
        except Exception as e:  # noqa: BLE001 — surface tool failures to the model, don't crash the rollout
            logger.warning("[mini_agent] tool %s failed: %s", name, e)
            return f"error: tool {name} raised {type(e).__name__}: {e}"

    async def run(
        self,
        sb,
        *,
        session_id: str,
        adapter_url: str,
        prompt: str,
        max_tokens: int = 1024,
        model_label: str = "slime-actor",
    ) -> MiniAgentResult:
        """Run the agent loop against ``sb``, routing every model call through the adapter.

        ``adapter_url`` is the base URL of slime's AnthropicAdapter; ``session_id`` is echoed in
        the Bearer header so the adapter keys this run's trajectory. Returns a MiniAgentResult;
        the training samples are produced by the adapter's finish_session, not here.
        """
        import aiohttp

        headers = {
            "Authorization": f"Bearer {session_id}",  # adapter reads the sid from here
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        messages: list[dict] = [{"role": "user", "content": prompt}]
        result = MiniAgentResult()
        timeout = aiohttp.ClientTimeout(total=self.request_timeout_sec)

        async with aiohttp.ClientSession(timeout=timeout) as http:
            for turn in range(self.max_turns):
                body = {
                    "model": model_label,
                    "max_tokens": max_tokens,
                    "system": SYSTEM_PROMPT,
                    "messages": messages,
                    "tools": TOOLS,
                }
                async with http.post(f"{adapter_url}/v1/messages", json=body, headers=headers) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(f"adapter /v1/messages returned {resp.status}: {text[:500]}")
                    data = await resp.json()

                blocks = data.get("content") or []
                stop_reason = data.get("stop_reason") or "end_turn"
                result.turns = turn + 1
                result.stop_reason = stop_reason
                # Echo the assistant turn back verbatim (the adapter compares this against its
                # recorded leaf, so the block shapes must round-trip unchanged).
                messages.append({"role": "assistant", "content": blocks})

                tool_uses = [b for b in blocks if isinstance(b, dict) and b.get("type") == "tool_use"]
                if not tool_uses:
                    break  # model produced a plain answer -> episode's model interaction is done

                tool_results = []
                for tu in tool_uses:
                    result.tool_calls += 1
                    out = await self._call_tool(sb, tu.get("name", ""), tu.get("input") or {}, result)
                    tool_results.append(
                        {"type": "tool_result", "tool_use_id": tu.get("id", ""), "content": out}
                    )
                messages.append({"role": "user", "content": tool_results})
            else:
                result.stop_reason = "budget"  # ran out of turns

        result.transcript = messages
        logger.info(
            "[mini_agent] session=%s turns=%d tool_calls=%d stop=%s last_exit=%s",
            session_id, result.turns, result.tool_calls, result.stop_reason, result.last_exit_code,
        )
        return result


# Small convenience for logging/debugging a transcript without dumping token soup.
def summarize_transcript(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if isinstance(content, str):
            lines.append(f"{role}: {content[:120]}")
        elif isinstance(content, list):
            for b in content:
                bt = b.get("type") if isinstance(b, dict) else "?"
                if bt == "text":
                    lines.append(f"{role}[text]: {b.get('text', '')[:120]}")
                elif bt == "tool_use":
                    lines.append(f"{role}[tool_use]: {b.get('name')} {json.dumps(b.get('input', {}))[:120]}")
                elif bt == "tool_result":
                    lines.append(f"{role}[tool_result]: {str(b.get('content', ''))[:120]}")
    return "\n".join(lines)
