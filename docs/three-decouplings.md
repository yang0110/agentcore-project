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
| **Agents with Harness** | our mini harness `MiniClaudeHarness` (A3/B3) | ② and ③ |
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
  do not merge them. Note this decoupling is about the **interface** (tool calls cross a boundary
  to a separate sandbox), *not* about which service hosts the harness: the harness is decoupled
  from the sandbox whether it runs in-process or on its own Runtime. We keep the harness in-process
  and put **only the sandbox on AgentCore**.
- **Where in code:** the harness = our own `MiniClaudeHarness`
  (`agentcore_slime_demo/mini_agent.py`, A3/B3), run either in-process for the reference loop or
  hosted by `harness_runtime_agent.py` (which wraps `MiniClaudeHarness`, `HARNESS_KIND=mini`).
  Sandboxes implement the `Sandbox` protocol (`code/agentcore_slime_demo/sandbox.py`):
  `LocalSandbox`, `AgentCoreSandbox` (Code Interpreter), `AgentCoreRuntimeSandbox` (separate
  Runtime).
- **Why:** swap CPU ↔ GPU sandbox without touching the harness; isolate untrusted / CUDA tool exec
  off the trainer GPUs; scale harness (latency-bound) and sandbox (tool-exec-bound) separately.

---

## Current implementations vs. target

| Row | Harness | Sandbox on AgentCore | Status |
|---|---|---|---|
| A / B (`demo_cpu_single`, `demo_gpu_single`) | our `run_episode`, **in-process** | `AgentCoreSandbox` (A, Code Interpreter) / remote GPU Runtime (B) | **recommended path** |
| A3 / B3 (`demo_cpu_2runtime`, `demo_gpu_2runtime`) | mini harness *also* on an AgentCore Runtime | Code Interpreter (A3) / GPU Runtime (B3) | possible, **not recommended** — see below |

The **recommended path is A/B**: harness in-process, **AgentCore = the sandbox only**. That is the
proven, natural-fit configuration and the one this project's story is built on.

(A prior A2 / B2 row ran the **real Claude Code CLI** co-located in the sandbox; that path has been
removed in favor of the mini harness.)

**Why A3/B3 (harness on its own Runtime) is *not* the recommended path.** It also puts the harness
on AgentCore (`harness_runtime_agent.py`, `HARNESS_KIND=mini`), sandbox on a second Runtime
(`sandbox_server_agent.py`), wired by `agentcore_runtime_generate.generate` with
`SANDBOX=agentcore-runtime`. It is kept as a reference for the "AgentCore *can* host a harness"
claim, but for RL it is a poor fit: the current hook mints a **fresh harness session per episode**
(`agentcore_runtime_generate.py`), i.e. a cold start each episode, when the harness actually wants
to be a **warm, reused pool** — persistence the harness needs but the sandbox does not. If ever
pursued, the harness would need a bounded pool of reused `runtimeSessionId`s, not one per episode.

> The two-Runtime backend (`agentcore_runtime_sandbox.py`, `sandbox_server_agent.py`,
> `harness_runtime_agent.py`, `agentcore_runtime_generate.py`) is **untested scaffolding** — it has
> not been fully run against AWS (GPU instances cost money) and is retained for reference only. The
> recommended `demo_cpu_single` / `demo_gpu_single` paths (AgentCore = sandbox) are what work.
