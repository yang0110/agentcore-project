# The three decouplings (quick reference)

A one-page map of the architecture, to refer back to when reviewing the codebase. This project
*is* the Agent Lightning v1.0 design ("Towards Harnessed Agentic RL", arXiv 2608.17528): five
components separated by **three decouplings**. Longer treatment in
[harness-sandbox-decoupling.md](harness-sandbox-decoupling.md); the slide version is in
`slides/agentcore_slime_deck.tex` ("The three decouplings", "Why decouple?", "Target topology").

```
  ┌──────────────────────────┐
  │ Megatron trainer         │   ① train ⟂ inference
  │      ↕ weight sync       │   ────────────────────
  │ SGLang inference         │
  └──────────┬───────────────┘
             │ serve
  ┌──────────┴───────┐   Anthropic API   ┌─────────────┐   tool call/result   ┌──────────────┐
  │ LLM API Proxy    │◀─────────②───────▶│  Harness    │─────────③──────────▶│  Sandbox     │
  │ AnthropicAdapter │                    │ CC / mini   │                      │ CodeInt / GPU│
  └──────────────────┘                    └─────────────┘                      └──────────────┘
```

The five components: **trainer · inference · proxy · harness · sandbox** — each scales, fails, and
(for the sandbox on GPU rows) acquires GPUs independently.

---

## Alignment with Agent Lightning (Figure 1)

This is a direct mapping of AL's Figure-1 components onto ours — i.e. the design is AL, not an
approximation of it. The only gap is **implementation maturity** (the current rows are bring-up
shortcuts, see the table at the bottom), not architecture.

| Agent Lightning component (Fig. 1) | Our component | Decoupling it participates in |
|---|---|---|
| **Training Engine** | slime Megatron actor | ① (⟂ Inference Engine) |
| **Inference Engine** | SGLang | ① |
| **LLM API Proxy** | slime `AnthropicAdapter` | ② (*is* the seam between harness ↔ inference) |
| **Agents with Harness** | Claude Code (A2/B2) / mini harness (A3/B3) | ② and ③ |
| *(execution environment)* | AgentCore sandbox (Code Interpreter / GPU Runtime) | ③ (⟂ harness) |
| **Sample Adapter** | slime `TrajectoryManager` | — (turns → loss-masked samples) |

AL's three decouplings map 1:1 to ①/②/③ below.

---

## ① Train ⟂ inference

- **What:** the Megatron trainer GPUs and the SGLang inference GPUs are disjoint (no `--colocate`);
  async overlaps rollout *N+1* with training of *N*. = AL Training Engine ⟂ Inference Engine.
- **Where in code:** slime's `train_async.py` (asserts `not args.colocate`); driven by
  `code/p4_slime_train/run_async.sh` and each `code/demos/*/run.sh`. GPU split shows up as
  "4 train + 4 infer" (A rows) / "4 train + 2 infer + 2 env" (B rows, local stand-in).
- **Why:** long, variable agentic-rollout latency would idle the trainer under sync/colocate;
  disjoint GPUs let the two pipeline.

## ② Harness ⟂ inference engine — *via the proxy*

- **What:** the harness never calls SGLang directly. It calls the **proxy** (`ANTHROPIC_BASE_URL`
  → `AnthropicAdapter`), which routes to SGLang **and records every request/response pair**. This
  is the seam that makes RL work: the trainer observes only LLM req/resp pairs — AL's defining
  property of *harnessed* agentic RL. The core AL idea, verbatim: "connect arbitrary agents to RL
  training through an LLM endpoint proxy."
- **Where in code:** the proxy is slime's `AnthropicAdapter` (`slime.agent.adapters`;
  `open_session` / `finish_session` / `drop_session`), stood up via `run_app_in_thread`
  (`slime.agent.aiohttp_threaded`) in every generate hook —
  `agentcore_slime_demo/claude_code_generate.py` and `agentcore_runtime_generate.py` (the latter
  is the mini harness's only home now — its own Runtime). How the harness reaches it differs by
  harness kind: the **CLI
  harness** (`ClaudeCodeHarness`, A2/B2) sets `ANTHROPIC_BASE_URL` + `ANTHROPIC_AUTH_TOKEN` =
  session id; the **mini harness** (`MiniClaudeHarness`, A3/B3, and the Runtime harness) is given
  the adapter URL directly and carries the session id in an `Authorization: Bearer <sid>` header.
- **Why:** any black-box agent (real Claude Code, a mini loop, anything speaking the Anthropic
  Messages API) connects with **zero agent changes**; the trainer never imports the agent's code.

## ③ Harness ⟂ sandbox

- **What:** the **harness** owns the agent loop (tool selection, context, control flow); the
  **sandbox** is the execution environment where tool actions run. Two distinct component types —
  do not merge them.
- **Where in code:** harnesses = slime's `ClaudeCodeHarness` (A2/B2) or our own
  `MiniClaudeHarness` (`agentcore_slime_demo/mini_agent.py`, A3/B3), or the AgentCore-hosted
  `harness_runtime_agent.py` (which wraps `MiniClaudeHarness`). Sandboxes implement the `Sandbox` protocol
  (`code/agentcore_slime_demo/sandbox.py`): `LocalSandbox`, `AgentCoreSandbox` (Code Interpreter),
  `AgentCoreRuntimeSandbox` (separate Runtime). The MCP bridge (`sandbox_mcp.py`) redirects a
  black-box CLI's tools to a sandbox.
- **Why:** swap CPU ↔ GPU sandbox without touching the harness; isolate untrusted / CUDA tool exec
  off the trainer GPUs; scale harness (latency-bound) and sandbox (tool-exec-bound) separately.

---

## Current implementations vs. target

| Row | Harness | Sandbox | Status |
|---|---|---|---|
| A / B (`demo_a`, `demo_b`) | our `run_episode`, in-process | `AgentCoreSandbox` (A) / GPU stand-in (B) | reference loop |
| A2 / B2 (`demo_a2`, `demo_b2`) | real Claude Code, **in the sandbox** | Code Interpreter (A2) / GPU (B2) | co-location shortcut |
| A3 / B3 (`demo_a3`, `demo_b3`) | mini harness, **in trainer process** | Code Interpreter (A3) / GPU (B3) | in-process shortcut |

**Target for all four harness rows (A2/A3/B2/B3):** harness on **AgentCore Runtime #1**
(`harness_runtime_agent.py`), sandbox on **AgentCore Runtime #2** (`sandbox_server_agent.py`),
wired by `agentcore_runtime_generate.generate` with `SANDBOX=agentcore-runtime`
(`AgentCoreRuntimeSandbox`). For B rows the sandbox Runtime is a **real AgentCore GPU Runtime
instance** (capacity-provider EC2 GPU), not a local-GPU stand-in.

> The two-Runtime backend (`agentcore_runtime_sandbox.py`, `sandbox_server_agent.py`,
> `harness_runtime_agent.py`, `agentcore_runtime_generate.py`) is **untested scaffolding** — it has
> not been run against AWS (GPU instances cost money). The existing shortcut paths still work.
