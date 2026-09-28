# The three decouplings (quick reference)

A one-page map of the architecture, to refer back to when reviewing the codebase. This project
*is* the Agent Lightning v1.0 design ("Towards Harnessed Agentic RL", arXiv 2608.17528): five
components separated by **three decouplings**. Longer treatment in
[harness-sandbox-decoupling.md](harness-sandbox-decoupling.md); the slide version is in
`slides/aws/agentcore_slime.tex` ("The three decouplings", "Why decouple?", "Target topology").

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

> **Where AgentCore fits (the one claim this project makes).** AgentCore is used as the
> **sandbox** — the per-episode execution environment (Code Interpreter for CPU rows, a GPU
> Runtime instance for CUDA rows). That is a natural fit: a sandbox *wants* to be ad-hoc and
> isolated, which is exactly AgentCore's per-session microVM model. The **harness stays
> in-process** (a warm, pooled agent loop next to the trainer). AgentCore *can* also host a
> harness (its managed-agent product), but for RL rollouts we deliberately do **not** — episodes
> are independent, so the harness needs process *warmth*, not per-episode *isolation*, and minting
> a fresh AgentCore session per episode (a cold start) is the wrong lifecycle for it. See
> [harness-sandbox-decoupling.md](harness-sandbox-decoupling.md) for the full argument.

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
| **Agents with Harness** | our in-process mini harness `MiniClaudeHarness` | ② and ③ |
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
  (`slime.agent.aiohttp_threaded`) in the generate hook. The **mini harness**
  (`MiniClaudeHarness`), run in-process, is given the adapter URL directly and carries the session
  id in an `Authorization: Bearer <sid>` header.
- **Why:** any black-box agent (real Claude Code, a mini loop, anything speaking the Anthropic
  Messages API) connects with **zero agent changes**; the trainer never imports the agent's code.

## ③ Harness ⟂ sandbox

- **What:** the **harness** owns the agent loop (tool selection, context, control flow); the
  **sandbox** is the execution environment where tool actions run. Two distinct component types —
  do not merge them. This decoupling is about the **interface** (tool calls cross a boundary to a
  separate sandbox), *not* about which service hosts the harness. We keep the harness **in-process**
  and put **only the sandbox on AgentCore** — as its own Runtime (GPU) or a Code Interpreter session
  (CPU).
- **Where in code:** the harness = our own in-process `MiniClaudeHarness`
  (`agentcore_slime_demo/mini_agent.py`). Sandboxes implement the `Sandbox` protocol
  (`code/agentcore_slime_demo/sandbox.py`): `LocalSandbox`, `AgentCoreSandbox` (Code Interpreter),
  `AgentCoreRuntimeSandbox` (the sandbox on its own AgentCore Runtime — the GPU rows).
- **Why:** swap CPU ↔ GPU sandbox without touching the harness; isolate untrusted / CUDA tool exec
  off the trainer GPUs; scale harness (latency-bound) and sandbox (tool-exec-bound) separately.

---

## Current implementations vs. target

| Row | Harness | Sandbox on AgentCore | Status |
|---|---|---|---|
| A / B (`demo_cpu_tasks`, `demo_gpu_single`) | our `run_episode`, **in-process** | `AgentCoreSandbox` (A, Code Interpreter) / remote GPU Runtime (B) | **the path** |

The configuration is settled: **harness in-process, AgentCore = the sandbox only.** That is the
proven, natural-fit topology and the one this project's story is built on. Every demo folder now
varies the **task** and the **sandbox substrate** (CPU Code Interpreter vs GPU Runtime instance) —
never the harness placement.

(Two prior rows are gone: an A2/B2 row ran the **real Claude Code CLI** co-located in the sandbox,
and an A3/B3 row hosted the mini harness on its *own* AgentCore Runtime. Both are removed. AgentCore
*can* host a harness — its managed-agent product — but for RL rollouts it is the wrong lifecycle:
the harness wants a **warm, reused pool**, not a fresh per-episode session (a cold start), whereas
the sandbox wants exactly the ad-hoc per-episode isolation AgentCore provides. See
[harness-sandbox-decoupling.md](harness-sandbox-decoupling.md).)
