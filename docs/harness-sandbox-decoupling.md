# Decoupling & disaggregating harness and sandbox (A3 / B3)

**Status: target architecture, partially wired (untested scaffolding).** In (harnessed) agentic RL
the **harness** and the **sandbox** are two distinct component types (Agent Lightning v1.0, §1). The
**target for the harness rows — A3 and B3 — is fully decoupled and disaggregated**: the harness runs
on **its own AgentCore Runtime** and the sandbox is a **separate AgentCore Runtime**, with the slime
trainer (policy) and the proxy (`AnthropicAdapter`) as the other two components. See **§3 for the
canonical target**; §1 walks through the intermediate topology and §2 covers the GPU sandbox.

The current implementation is a bring-up shortcut, not the target: A3 / B3 run the mini harness
(`MiniClaudeHarness`) *in the trainer/rollout process* against an in-process sandbox object (§1). The
target moves that same loop onto its own Runtime (`harness_runtime_agent.py`, `HARNESS_KIND=mini`),
with tool calls dispatched to a separate sandbox Runtime.

> **Removed:** an earlier variant co-located the **real Claude Code CLI** *inside* the sandbox (rows
> A2/B2) and redirected its built-in tools to the sandbox over an MCP bridge. That real-CLI path —
> and its custom multi-user container / CLI-deploy tooling — has been removed in favor of the mini
> harness. This doc describes the mini-harness A3/B3 path only. (The in-process MCP helper
> `sandbox_mcp.py` still exists in the tree if a black-box CLI is ever reintroduced, but it is not
> part of the current path.)

---

## 0. Alignment with Agent Lightning (Figure 1)

This design *is* the Agent Lightning v1.0 architecture. AL calls for **three** decouplings; we have
all three:

| Agent Lightning component (Fig. 1) | Our component | Decoupled from |
|---|---|---|
| **Training Engine** | slime Megatron actor | the Inference Engine |
| **Inference Engine** | SGLang | the Training Engine |
| **LLM API Proxy** | slime `AnthropicAdapter` | *sits between* harness ↔ inference |
| **Agents with Harness** | our mini harness `MiniClaudeHarness` (A3/B3) | the Inference Engine (via the proxy) *and* the sandbox |
| *(execution environment)* | AgentCore sandbox (Code Interpreter / GPU Runtime) | the harness |
| **Sample Adapter** | slime `TrajectoryManager` | — |

1. **Train ⟂ inference.** slime's async **decoupled placement** (disjoint GPUs, `not
   args.colocate`) = AL Training Engine ⟂ Inference Engine.
2. **Harness ⟂ inference engine — *via the proxy*.** The core AL idea: *"connect arbitrary agents
   to RL training through an LLM endpoint proxy."* The harness calls the **proxy**
   (`ANTHROPIC_BASE_URL` → `AnthropicAdapter`), never SGLang directly; the proxy routes to the
   inference engine **and** records every request/response pair. That indirection is what lets the
   **training engine observe only LLM request/response pairs** — AL's defining property of
   *harnessed* agentic RL.
3. **Harness ⟂ sandbox.** AL: *"the harness owns tools, execution environments, and the interaction
   loop."* Harness = the loop; sandbox = the environment → the two-Runtime target of §3.

Net: five independently-scalable services — trainer · inference · proxy · harness · sandbox. The
only gap vs. AL is **implementation maturity** (the current row is a bring-up shortcut, §1), not
architecture.

---

## 1. Topologies (bring-up → target)

Both keep the same four components (training engine · proxy · harness · sandbox); they differ only
in **where the harness runs** and therefore **how tool actions reach the sandbox**. §1A is the
stepping stone; **§3 is the target.**

### A. In-process harness (what the code does today)

```
  host / rollout worker                              AgentCore sandbox (any type)
  ┌────────────────────────────────────┐            ┌───────────────────────────┐
  │ slime generate()                    │            │  bash / write / read      │
  │  MiniClaudeHarness (HARNESS)        │            │  run in the sandbox       │
  │   pick tool ── sb.exec / write ─────┼────────────▶  (tool execution only)    │
  │ AnthropicAdapter (proxy)            │◀─Anthropic─ │  run_checks.py (reward)   │
  └────────────────────────────────────┘            └───────────────────────────┘
```

- `MiniClaudeHarness` (`agentcore_slime_demo/mini_agent.py`) runs the agent loop **in the rollout
  process**: call the model through the proxy, pick a tool, run it with `sb.exec` / `sb.write_file`,
  read the result, repeat. The sandbox is an in-process `Sandbox` object (`LocalSandbox`,
  `AgentCoreSandbox`, or `AgentCoreRuntimeSandbox`).
- **Pro:** minimal — no separate service to deploy; runs offline against `LocalSandbox`. This is the
  A3/B3 bring-up path and the reference for the harness loop.
- **Con:** the harness isn't on AgentCore at all — it lives in slime's process, so harness and
  trainer are physically coupled even though they're conceptually distinct components. The target
  (§3) moves the loop onto its own Runtime.

> **B3 caveat:** the KernelBench *checker* still needs CUDA, so B3's **sandbox** is a GPU Runtime
> (§2). Decoupling moves the *harness* off the GPU box, but the GPU sandbox stays (that's the
> task's requirement, not the harness's).

---

## 2. B / B3 sandbox = a real AgentCore GPU Runtime instance (not the local box)

**Current reality:** Demo B / B3 run the GPU sandbox as its **own remote AgentCore GPU Runtime
instance** (`AgentCoreRuntimeSandbox`, `SANDBOX=agentcore-runtime`) — a capacity-provider EC2 GPU (a
`g6`) launched by AgentCore. The checker's CUDA work (`python3 run_checks.py`) runs on that **remote
managed instance**, reached over `invoke_agent_runtime`, so **no** local GPUs are held back for env
checkers. slime runs **4 train + 4 infer** on all 8 local GPUs (like A/A3).

### What running the sandbox off-box gives us

- **GPU accounting is 4+4+0.** Because the sandbox lives on a *separate* AgentCore GPU instance,
  the trainer box reserves nothing for envs → slime uses all 8 local GPUs.
- **Fully physical split.** With the sandbox on a remote AgentCore GPU instance (and, in the target,
  the harness on its own Runtime per §3), the split is fully physical: trainer box = policy (8 GPU)
  + proxy + harness; AgentCore GPU instance = sandbox only. This is the cleanest form of the
  four-component architecture.
- **Sharing lives on the remote Runtime.** If many episodes must share one g6, that is a concern
  of the remote Runtime image (Demo B4 runs a Ray GPU-sharing farm *inside* the sandbox Runtime),
  not of the trainer box.
- **Cost/lifecycle:** a GPU Runtime instance bills while up (~$0.80–1/hr + EBS on `g6.xlarge`) and
  must be torn down — see the 08/09 `OBSERVATIONS.md` and the
  capacity-provider/`securityGroups`/`LD_LIBRARY_PATH` gotchas found there. A rollout that
  launches one instance per episode would be far too slow/expensive; the realistic design is a
  **pool of long-lived GPU Runtime sessions** the rollouts share.

> **Honest status:** this is item (b) of demos-guide.md §8 "Not yet" — a full logged *Demo B on a
> real AgentCore GPU Runtime instance* training run. The `AgentCoreRuntimeSandbox` backend and the
> GPU sandbox Runtime image are built and deploy; the end-to-end logged run is the pending piece.

---

## 3. The real target: TWO AgentCore Runtimes — one harness, one sandbox

**Correction to §1A.** Running A3/B3's mini harness **in-process** is a bring-up shortcut, not the
target. AgentCore is not only a sandbox — it can host the **harness** too (its own Runtime running
the agent loop, or the managed harness product, use case 10). So the target topology puts **each
component type on its own AgentCore Runtime**:

```
  trainer box (slime)                AgentCore Runtime #1              AgentCore Runtime #2
  ┌────────────────────┐             ┌──────────────────────┐         ┌──────────────────────┐
  │ Megatron + SGLang  │             │  HARNESS             │         │  SANDBOX             │
  │ AnthropicAdapter   │◀─Anthropic──┤  agent loop:         │──tool──▶│  bash/write/read     │
  │ (proxy endpoint)   │   (LLM)     │  mini harness (A3/B3)│  calls  │  run_checks.py       │
  │                    │             │  HARNESS_KIND=mini   │         │  (Code Interp / GPU) │
  └────────────────────┘             └──────────────────────┘         └──────────────────────┘
       policy GPUs                    AgentCore-hosted harness          AgentCore-hosted sandbox
                                      (NOT in the trainer process)      (GPU Runtime for B)
```

- **Harness = AgentCore Runtime #1.** The agent loop is deployed *as an AgentCore Runtime*
  (`harness_runtime_agent.py`, `HARNESS_KIND=mini`, wrapping `MiniClaudeHarness`), **not** run inside
  the slime rollout process. It reaches the policy through the proxy and issues tool calls to
  Runtime #2.
- **Sandbox = AgentCore Runtime #2.** A *separate* AgentCore Runtime executes the tool calls —
  Code Interpreter for CPU rows (A3), a **GPU Runtime instance** for B rows (B3, §2).
- **Two Runtimes, two roles.** This is the clean four-component split with AgentCore owning both
  the harness and the sandbox as distinct managed services.

### How the row gets to the two-Runtime target

- **A3 / B3 (mini harness):** deploy `MiniClaudeHarness` as a `BedrockAgentCoreApp` on its own
  AgentCore Runtime (`harness_runtime_agent.py`). Its LLM calls go to the proxy; its tool calls go
  to the **sandbox Runtime** (Code Interpreter for A3, GPU Runtime for B3) via `invoke_agent_runtime`
  instead of a local `sb` object. This is a clean port since we own the harness code — the
  tool-dispatch just changes from an in-process `sb.exec` to an invoke of the sandbox Runtime. The
  generate hook (`agentcore_runtime_generate.generate`, `SANDBOX=agentcore-runtime`) wires the two
  Runtime ARNs together per rollout; build/create both with `code/deploy/deploy_two_runtime.py`.

> **Wiring per rollout** (from `harness_runtime_agent.py`): `agentcore_runtime_generate.generate`
> invokes the harness Runtime with `{prompt, adapter_url, session_id, model, sandbox_runtime_arn}`;
> the harness runs the loop, dispatches each tool call to sandbox Runtime #2, and returns a summary
> (`turns`, `stop_reason`, …) — the trajectory itself is captured proxy-side, so GRPO sample capture
> is unchanged whether the harness is in-process or on Runtime #1.

> **Takeaway:** the goal for **A3/B3** is *harness on one AgentCore Runtime, sandbox on another*.
> The in-process harness is a bring-up shortcut to be replaced. The two-Runtime backend
> (`agentcore_runtime_sandbox.py`, `sandbox_server_agent.py`, `harness_runtime_agent.py`,
> `agentcore_runtime_generate.py`) is built but is **untested scaffolding** — it has not been fully
> run against AWS (GPU instances cost money).
