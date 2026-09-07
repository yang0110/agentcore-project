"""Offline smoke for the mini Claude-Code-like harness (Demos A3/B3).

No slime, no AWS, no GPU, no real policy: stands up a FAKE Anthropic Messages endpoint that
returns a scripted sequence of tool_use blocks, and drives the real MiniClaudeHarness against a
real LocalSandbox. Verifies the write->run->fix loop actually (a) routes model calls to the
adapter with the session Bearer header, (b) executes write_file/bash tools in the sandbox, and
(c) terminates on end_turn. This exercises everything in mini_agent.py except the token/trajectory
capture (which lives in slime's adapter and is covered by the real training path).

Run:  PYTHONPATH=code python3 code/scripts/mini_agent_smoke.py
"""

from __future__ import annotations

import asyncio
import sys

from aiohttp import web

from agentcore_slime_demo.mini_agent import MiniClaudeHarness
from agentcore_slime_demo.sandbox import LocalSandbox


class FakeAdapter:
    """Scripted Anthropic Messages server: one canned assistant turn per request."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.seen_bearer: list[str] = []
        # Scripted turns: write solution.py, run the checker, then finish.
        self._script = [
            [{"type": "tool_use", "id": "t1", "name": "write_file",
              "input": {"path": "solution.py", "content": "print('hello from solution')\n"}}],
            [{"type": "tool_use", "id": "t2", "name": "bash",
              "input": {"command": "python3 solution.py"}}],
            [{"type": "text", "text": "The solution runs and prints the expected output. Done."}],
        ]

    async def messages(self, request: web.Request) -> web.Response:
        body = await request.json()
        self.calls.append(body)
        self.seen_bearer.append(request.headers.get("Authorization", ""))
        idx = min(len(self.calls) - 1, len(self._script) - 1)
        blocks = self._script[idx]
        stop_reason = "tool_use" if any(b["type"] == "tool_use" for b in blocks) else "end_turn"
        return web.json_response({
            "id": "msg_fake", "type": "message", "role": "assistant", "model": "fake",
            "content": blocks, "stop_reason": stop_reason, "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        })


async def main() -> int:
    fake = FakeAdapter()
    app = web.Application()
    app.router.add_post("/v1/messages", fake.messages)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # bound ephemeral port
    adapter_url = f"http://127.0.0.1:{port}"

    ok = True
    try:
        harness = MiniClaudeHarness(max_turns=6)
        async with LocalSandbox(workdir="/workspace") as sb:
            result = await harness.run(
                sb, session_id="smoke-session-123", adapter_url=adapter_url,
                prompt="Write solution.py and run it.",
            )
            written = await sb.read_file("solution.py")

        checks = {
            "3 turns taken": result.turns == 3,
            "stop_reason end_turn": result.stop_reason == "end_turn",
            "2 tool calls": result.tool_calls == 2,
            "bash exit 0": result.last_exit_code == 0,
            "solution.py written": "hello from solution" in written,
            "session Bearer forwarded": all(b == "Bearer smoke-session-123" for b in fake.seen_bearer),
            "tools advertised": bool(fake.calls[0].get("tools")),
            "system prompt sent": bool(fake.calls[0].get("system")),
        }
        for name, passed in checks.items():
            print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
            ok = ok and passed
    finally:
        await runner.cleanup()

    print("mini_agent smoke:", "ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
