# Decoupling harness and sandbox — and why AgentCore is the sandbox, not the harness

**The one claim: AgentCore is the *sandbox*.** In (harnessed) agentic RL the **harness** and the
**sandbox** are two distinct component types (Agent Lightning v1.0, §1). This project uses
**AgentCore as the sandbox** — the per-episode execution environment (Code Interpreter for CPU
rows, a GPU Runtime instance for CUDA rows). That is the natural fit and the proven path. The
**harness stays in-process** (a warm agent loop next to the trainer, §1).

**Why not put the harness on AgentCore too?** You *can* — AgentCore has a managed-agent product —
but for RL it is the wrong lifecycle, so we don't:

- A **sandbox** wants to be **ad-hoc and isolated**: a fresh, throwaway environment per episode.
  That is exactly AgentCore's per-session microVM model → perfect fit.
- A **harness** wants to be **long-standing and warm**: episodes are independent, so the harness
  carries *no* cross-episode state and gains nothing from per-episode isolation — it only wants to
  avoid cold starts. AgentCore's unit of compute is the session, spun up per invoke and reclaimed
  on idle; using it for the harness means a **cold start per episode** unless you hand-build a warm
  reused-session pool. That is fighting the platform to get something the harness didn't need from
  it.

So the topology is settled: **sandbox → AgentCore; harness → in-process.** §1 covers the in-process
harness (the path), and §2 covers the GPU sandbox running as its own AgentCore Runtime.

> **In progress (2026-09-26, not done).** `demo_cpu_tasks`' default is switching from the Code
> Interpreter to our own Runtime (the generic stdlib image, VPC mode) with the harness running
> **inside the sandbox session**: one agent loop per sandbox, harness and tools in the SAME AgentCore
> Runtime session (the DSec pattern; today only `demo_cpu_branching` does this). The Code Interpreter
> stays as a zero-setup opt-in. Until that lands and is measured, everything below describes the
> current in-process path, which is still what every committed result used.

> **Removed variants.** Two earlier variants are gone:
> - **A2/B2** co-located the **real Claude Code CLI** *inside* the sandbox and redirected its tools
>   over an MCP bridge. That real-CLI path — and its custom multi-user container / CLI-deploy
>   tooling — was removed in favor of the mini harness.
> - **A3/B3** (`demo_cpu_2runtime` / `demo_gpu_2runtime`) hosted the mini harness on its **own**
>   AgentCore Runtime (Runtime #1) driving a sandbox Runtime #2. Removed — for RL the harness wants
>   a warm reused-session pool, not a fresh per-episode session (a cold start). The demos now vary
>   only the **task** and the **sandbox substrate**, never the harness placement.

> (The in-process MCP helper `sandbox_mcp.py` still exists in the tree if a black-box CLI is ever
> reintroduced, but it is not part of the current path.)

---

## 0. Alignment with Agent Lightning (Figure 1)

This design *is* the Agent Lightning v1.0 architecture. AL calls for **three** decouplings; we have
all three:

| Agent Lightning component (Fig. 1) | Our component | Decoupled from |
|---|---|---|
| **Training Engine** | slime Megatron actor | the Inference Engine |
| **Inference Engine** | SGLang | the Training Engine |
| **LLM API Proxy** | slime `AnthropicAdapter` | *sits between* harness ↔ inference |
| **Agents with Harness** | our in-process mini harness `MiniClaudeHarness` | the Inference Engine (via the proxy) *and* the sandbox |
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
   loop."* Harness = the loop (in-process); sandbox = the environment (AgentCore, §2). This
   decoupling is about the **interface** — tool calls crossing a boundary to a separate sandbox —
   not about which service hosts the harness.

Net: five independently-scalable services — trainer · inference · proxy · harness · sandbox. The
only gap vs. AL is **implementation maturity** (a logged Demo-B GPU-Runtime run, §2), not
architecture.

---

## 1. Topology: in-process harness, AgentCore sandbox

The four components (training engine · proxy · harness · sandbox) sit like this — the harness runs
in the rollout process, and tool actions cross the boundary to the AgentCore sandbox:

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
  `AgentCoreSandbox` for Code Interpreter, or `AgentCoreRuntimeSandbox` for the GPU Runtime).
- **Why in-process:** episodes are independent, so the harness carries no cross-episode state and
  needs *warmth*, not per-episode isolation. Running it in-process avoids a cold start per episode
  and keeps the loop trivially debuggable (it runs offline against `LocalSandbox`). The sandbox —
  which *does* want ad-hoc isolation — is the piece that lives on AgentCore.
- The harness being in-process does **not** couple it to the sandbox: tool actions still cross a
  real boundary (`sb.exec` → Code Interpreter session / GPU Runtime invoke). Decoupling ③ is about
  that interface, not about where the harness process lives.

> **B caveat:** the KernelBench *checker* still needs CUDA, so Demo B's **sandbox** is a GPU Runtime
> (§2). The harness stays in-process on the trainer box; only the sandbox is on the GPU instance
> (that's the task's requirement, not the harness's).

---

## 2. Demo B sandbox = a real AgentCore GPU Runtime instance (not the local box)

**Current reality:** Demo B runs the GPU sandbox as its **own remote AgentCore GPU Runtime
instance** (`AgentCoreRuntimeSandbox`, `SANDBOX=agentcore-runtime`) — a capacity-provider EC2 GPU (a
`g6`) launched by AgentCore. The checker's CUDA work (`python3 run_checks.py`) runs on that **remote
managed instance**, reached over `invoke_agent_runtime`, so **no** local GPUs are held back for env
checkers. slime runs **4 train + 4 infer** on all 8 local GPUs (like Demo A).

This is the sandbox-on-its-own-Runtime case — and it is exactly the fit AgentCore is good at: an
ad-hoc, isolated, per-session GPU environment. It does **not** contradict "harness in-process": the
*harness* still runs on the trainer box; only the *sandbox* is the remote Runtime.

### What running the sandbox off-box gives us

- **GPU accounting is 4+4+0.** Because the sandbox lives on a *separate* AgentCore GPU instance,
  the trainer box reserves nothing for envs → slime uses all 8 local GPUs.
- **Fully physical split.** With the sandbox on a remote AgentCore GPU instance, the split is
  physical: trainer box = policy (8 GPU) + proxy + in-process harness; AgentCore GPU instance =
  sandbox only. This is the four-component architecture, cleanly.
- **Sharing lives on the remote Runtime.** If many episodes must share one g6, that is a concern
  of the remote Runtime image (a Ray GPU-sharing farm runs *inside* the sandbox Runtime — the
  remote form of the local farm in `demo_gpu_farm`, `FARM=fractional|whole` — merged 2026-09-26 from `demo_gpu_farm_fractional` + `demo_gpu_farm_whole`; formerly `demo_gpu_shared`), not of the trainer box. That farm is **trusted-only** (shared kernel + shared GPU: no MIG, a cooperative memory cap, one shared uid on AgentCore); full isolation is one session per episode (one g6 each), which no current demo runs; `demo_gpu_farm LOCAL_GPU=0` (formerly `demo_gpu_single`) shares one warm g6.
- **Cost/lifecycle:** a GPU Runtime instance bills while up (~$0.80–1/hr + EBS on `g6.xlarge`) and
  must be torn down — see the 08/09 `OBSERVATIONS.md` and the
  capacity-provider/`securityGroups`/`LD_LIBRARY_PATH` gotchas found there. A rollout that
  launches one instance per episode would be far too slow/expensive; the realistic design is a
  **pool of long-lived GPU Runtime sessions** the rollouts share.

> **Honest status:** this is item (b) of demos-guide.md §8 "Not yet" — a full logged *Demo B on a
> real AgentCore GPU Runtime instance* training run. The `AgentCoreRuntimeSandbox` backend and the
> GPU sandbox Runtime image are built and deploy; the end-to-end logged run is the pending piece.

---

## 3. Deploying the sandbox Runtime

Both remaining sandbox substrates are created by the shared orchestrator
`code/deploy/deploy_two_runtime.py` (the name predates the in-process-harness decision — it now
builds and creates the **sandbox** Runtime, IAM roles, ECR repos, VPC endpoints, and the GPU
capacity provider; it no longer deploys a harness Runtime):

- **CPU rows:** an AgentCore **Code Interpreter** session (`AgentCoreSandbox`).
- **GPU rows:** a **GPU Runtime instance** (`AgentCoreRuntimeSandbox`, `sandbox_server_agent.py` on a
  capacity-provider `g6`), reached per rollout via `invoke_agent_runtime`.

The harness never gets a Runtime of its own; it dispatches tool calls from the in-process loop to
whichever sandbox the demo wired up. GRPO sample capture is unaffected — the trajectory is captured
**proxy-side** (every LLM request/response pair), so it does not matter where the tool actions run.
