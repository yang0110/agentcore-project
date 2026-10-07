"""`DecoupledCliHarness` — the trainer-side client for the decoupled real-Claude-Code-CLI harness.

Mirror of `InSessionHarness.run`'s signature + result type, so `terminalbench_pool._run_episode`
swaps the object and shares all the downstream code (adapter open/finish/drop + grading). Instead of
running the agent loop in the sandbox session, it borrows a WARM harness Runtime session
(`HarnessCliRuntimePool`) and `invoke_agent_runtime`s it with a payload describing BOTH the proxy
(adapter) and the per-episode sandbox Runtime to drive. The trajectory is captured proxy-side by
slime's AnthropicAdapter (keyed on session_id); this returns only a summary.
"""

from __future__ import annotations

import os

from .mini_agent import MiniAgentResult


class DecoupledCliHarness:
    name = "claude_cli"
    in_session = False

    def __init__(self, pool, *, exec_timeout_sec: int = 180, request_timeout_sec: int = 300,
                 system_prompt: str | None = None) -> None:
        self._pool = pool
        self.exec_timeout_sec = exec_timeout_sec
        self.request_timeout_sec = request_timeout_sec
        self.system_prompt = system_prompt      # kept for interface parity; the CLI has its own prompt

    async def run(self, sb, *, session_id: str, adapter_url: str, prompt: str, arn: str,
                  max_turns: int = 8, max_tokens: int = 1024, model_label: str = "slime-actor",
                  http=None) -> MiniAgentResult:
        # The harness invoke must outlive `claude -p`; keep the CLI budget under the pool's read
        # timeout so the invoke doesn't time out before the CLI returns.
        budget = min(int(os.environ.get("AGENT_TIME_BUDGET_SEC", "900")),
                     max(60, self._pool._read_timeout - 60))
        payload = {
            "harness_kind": "claude",
            "prompt": prompt,
            "adapter_url": adapter_url,                 # the PROXY the CLI dials (ANTHROPIC_BASE_URL)
            "session_id": session_id,                   # adapter Bearer key == trajectory id
            "sandbox_runtime_arn": arn,                 # the per-episode sandbox Runtime to drive
            "sandbox_session_id": sb._session_id,       # the SAME session the trainer opened + grades
            "workdir": getattr(sb, "_workdir", "/workspace"),
            "max_turns": int(max_turns),
            "model": model_label,
            "time_budget_sec": budget,
        }
        async with self._pool.borrow() as harness_sess:
            res = await self._pool.invoke(harness_sess, payload)

        if isinstance(res, dict) and res.get("error"):
            raise RuntimeError(f"decoupled CLI harness error: {res['error']}")
        res = res or {}
        out = MiniAgentResult(
            turns=int(res.get("turns", 0)),
            tool_calls=int(res.get("tool_calls", 0)),
            stop_reason=res.get("stop_reason", "none"),
            last_exit_code=res.get("agent_exit_code"),
            transcript=[{"role": "user", "content": prompt}],
        )
        out.where = "harness_runtime"
        return out
