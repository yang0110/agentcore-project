"""`agentcore_runtime_generate.generate` — slime hook for the TWO-Runtime target (A3/B3 path).

This is the disaggregated custom-generate path and the ONLY home for the mini harness: it runs on
AgentCore Runtime #1 (harness_runtime_agent.py) and the sandbox on AgentCore Runtime #2
(sandbox_server_agent.py). The slime trainer only ever sees the LLM request/response pairs,
captured by the proxy (AnthropicAdapter) — exactly Agent Lightning. Demos A3 and B3 wire this hook
(the earlier in-process shortcut, mini_agent_generate.generate, has been removed — the mini harness
has ONE home now, its own Runtime).

It differs from the other hooks only in WHERE the harness runs:
  * generate.generate               — our run_episode, in-process, local `sb`  (Demos A/B).
  * claude_code_generate.generate   — Claude Code, in-sandbox                    (A2/B2 shortcut).
  * THIS                            — harness on Runtime #1, sandbox on Runtime #2 (A3/B3, target).

Flow per prompt:
  1. stand up the proxy (AnthropicAdapter) + open a session, as the other hooks do;
  2. invoke the HARNESS Runtime, passing adapter_url (proxy) + the SANDBOX Runtime ARN;
     the harness loops (LLM via proxy, tools via sandbox Runtime) and returns when done;
  3. score by invoking the sandbox Runtime's checker;
  4. finish_session drains the proxy-captured trajectory to loss-masked GRPO samples.

Wire with:
  --custom-generate-function-path agentcore_slime_demo.agentcore_runtime_generate.generate
and set AGENTCORE_HARNESS_RUNTIME_ARN + AGENTCORE_SANDBOX_RUNTIME_ARN + ADAPTER_PUBLIC_HOST (this
box's private IP, which Runtime #1 dials back to over the SG-gated private VPC — no public port).
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import threading

_AGENT_PROMPT = (
    "Read PROBLEM.md in this directory. Write a file named solution.py that satisfies it. "
    "Then run `python3 run_checks.py` and iterate until it exits 0 (all checks pass)."
)

# One SHARED adapter per (bind host, port) for the whole rollout. The harness runs on a SEPARATE
# Runtime (#1) and dials back to a PINNED port (ADAPTER_PORT) so a single narrow SG rule stays
# stable (no public port). A per-episode adapter on a fixed port collides ([Errno 98]); the
# AnthropicAdapter already multiplexes concurrent episodes by session_id (the Bearer session id),
# so we construct+start it ONCE and every episode opens/finishes/drops its own session on it.
_SHARED_ADAPTER: dict[tuple, tuple] = {}
_SHARED_LOCK = threading.Lock()


def _shared_adapter(args, bind_host: str, port: int):
    """Return (adapter, handle) for (bind_host, port), constructing + starting it once."""
    from slime.agent.adapters import AnthropicAdapter
    from slime.agent.aiohttp_threaded import run_app_in_thread
    from slime.utils.processing_utils import load_tokenizer

    key = (bind_host, port)
    with _SHARED_LOCK:
        hit = _SHARED_ADAPTER.get(key)
        if hit is not None:
            return hit
        tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        adapter = AnthropicAdapter(
            tokenizer=tokenizer,
            sglang_url=f"http://{args.sglang_router_ip}:{args.sglang_router_port}",
            tool_parser=getattr(args, "sglang_tool_call_parser", None) or None,
            reasoning_parser=getattr(args, "sglang_reasoning_parser", None) or None,
        )
        handle = run_app_in_thread(
            adapter.app, host=bind_host, port=port, thread_name="anthropic-adapter-runtime",
        )
        _SHARED_ADAPTER[key] = (adapter, handle)
        return adapter, handle


async def _invoke_runtime(arn: str, session_id: str, payload: dict, region: str, read_timeout: int) -> dict:
    import boto3
    from botocore.config import Config

    from .agentcore_runtime_sandbox import _retry_invoke

    cfg = Config(read_timeout=read_timeout, connect_timeout=60, retries={"max_attempts": 0})

    def _call() -> dict:
        client = boto3.client("bedrock-agentcore", region_name=region, config=cfg)
        # Retry transient 5xx/throttle (e.g. a 502 while a Runtime microVM is briefly unavailable) so a
        # single blip on one episode's invoke doesn't crash the whole ray training job. See _retry_invoke.
        resp = _retry_invoke(
            lambda: client.invoke_agent_runtime(
                agentRuntimeArn=arn, runtimeSessionId=session_id, payload=json.dumps(payload).encode()
            ),
            what=f"runtime invoke {payload.get('op') or payload.get('harness_kind') or ''}".strip(),
        )
        raw = resp["response"].read()
        return json.loads(raw.decode()) if raw else {}

    return await asyncio.to_thread(_call)


async def generate(args, base_sample, sampling_params, evaluation: bool = False):
    # imports kept local so this module loads without slime present (offline tooling/tests).
    from .generate import _task_from_sample

    region = os.environ.get("AGENTCORE_REGION", "us-west-2")
    read_timeout = int(os.environ.get("INVOKE_READ_TIMEOUT", "900"))
    harness_arn = os.environ["AGENTCORE_HARNESS_RUNTIME_ARN"]
    sandbox_arn = os.environ["AGENTCORE_SANDBOX_RUNTIME_ARN"]
    workdir = os.environ.get("SANDBOX_WORKDIR", "/workspace")

    task = _task_from_sample(base_sample)
    # AgentCore requires runtimeSessionId length in [33, 128]; token_hex(16)=32 + prefix clears 33.
    sandbox_session_id = f"sbx-{secrets.token_hex(16)}"   # one isolated sandbox session per episode

    # --- proxy: ONE shared AnthropicAdapter in front of SGLang (constructed once, reused) --------
    # The harness runs on a SEPARATE Runtime (#1) and dials back to a PINNED port, so all episodes
    # share one adapter on ADAPTER_PORT (multiplexed by session_id) — a per-episode adapter on a
    # fixed port collides. ADAPTER_PORT=0 => ephemeral (single-process in-process use only).
    adapter_public_host = os.environ.get("ADAPTER_PUBLIC_HOST")
    if not adapter_public_host:
        raise RuntimeError("ADAPTER_PUBLIC_HOST not set (harness Runtime #1 must dial back).")
    adapter, handle = _shared_adapter(
        args, os.environ.get("ADAPTER_BIND_HOST", "0.0.0.0"), int(os.environ.get("ADAPTER_PORT", "0")))
    adapter_url = f"http://{adapter_public_host}:{handle.port}"

    # Also the adapter Bearer session key; must be >=33 chars for the harness runtimeSessionId too.
    session_id = base_sample.session_id
    if not session_id or len(session_id) < 33:
        session_id = f"acrt-{task.task_id}-{secrets.token_hex(16)}"
    base_sample.session_id = session_id
    adapter.open_session(session_id, sampling_defaults=sampling_params)

    try:
        # --- seed the sandbox workspace (Runtime #2) with the problem + checker ------------
        await _invoke_runtime(sandbox_arn, sandbox_session_id, {
            "op": "write_file", "path": f"{workdir}/PROBLEM.md", "text": task.initial_user_prompt(),
        }, region, read_timeout)
        await _invoke_runtime(sandbox_arn, sandbox_session_id, {
            "op": "write_file", "path": f"{workdir}/run_checks.py", "text": task.runner_source(),
        }, region, read_timeout)

        # --- run the HARNESS on Runtime #1; it dials the proxy + the sandbox Runtime -------
        await _invoke_runtime(harness_arn, session_id, {
            "harness_kind": os.environ.get("HARNESS_KIND", "mini"),
            "prompt": _AGENT_PROMPT,
            "adapter_url": adapter_url,                    # the proxy, reachable from Runtime #1
            "session_id": session_id,
            "sandbox_runtime_arn": sandbox_arn,
            "sandbox_session_id": sandbox_session_id,
            "workdir": workdir,
            # Cap agentic turns per episode (both harness kinds honor it): fewer turns => shorter
            # trajectories => cheaper rollout AND ref/actor forward passes. Passed in the payload so
            # it can change per run without redeploying Runtime #1 (once #1 runs a build that reads
            # it). Default 3.
            "max_turns": int(os.environ.get("EPISODE_MAX_TURNS", "3")),
        }, region, read_timeout)

        # --- score by invoking the sandbox checker -----------------------------------------
        check = await _invoke_runtime(sandbox_arn, sandbox_session_id, {
            "op": "exec", "command": "python3 run_checks.py", "timeout": 120, "workdir": workdir,
        }, region, read_timeout)
        reward = task.reward(int(check.get("exit_code", 1)))

        return await adapter.finish_session(
            session_id, base_sample=base_sample, reward=float(reward),
            extra_metadata={"task_id": task.task_id, "solved": reward == 1.0},
        )
    finally:
        await adapter.drop_session(session_id, wait_timeout=30)
