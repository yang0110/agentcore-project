# Agentic RL: AgentCore as Sandbox and Harness

**Kaige Yang** — Senior Applied Scientist, 亚马逊云科技 (AWS)

September 10, 2026

---

## Outline

1. **Agentic RL & its reference architecture** — what it is, and the harnessed-RL components it proposes.
2. **Scaling up the rollout** — why the sandbox/env layer is the constraint, and how to size every tier to the trainer.
3. **Use cases** — where a customer needs a harness + sandbox for agentic RL.
4. **slime** — the agentic-RL framework: architecture, design, training modes.
5. **AWS Bedrock AgentCore** — the managed agent platform; its harness and CPU/GPU sandbox roles.
6. **Realizing it on slime + AgentCore** — the three decouplings; one generate hook.
7. **Demos** — two harness stories × a CPU/GPU sandbox each.
8. **Networking, isolation & lifecycle** — private-VPC dial-back, the VPC-mode trap, and per-`session_id` routing.

---

## The main theme — AgentCore as harness and sandbox

This deck shows how to use **AWS Bedrock AgentCore as both the harness and the sandbox** for an agentic-RL training workload.

**Project theme — AgentCore fills the two RL roles around a slime trainer:**

- **AgentCore as harness** — runs the agent loop → produces completions / trajectories → feeds **slime** (trainer + inference, the policy under training).
- **AgentCore as sandbox** — runs tools + reward checker → returns tool call / result to the harness.

Definitions:

- **Harness** = the agent loop (our in-process loop, or a ~1-file mini agent on its own Runtime), hosted on an AgentCore Runtime.
- **Sandbox** = the isolated exec env (CPU Code-Interpreter or GPU Runtime) where tools run and reward is scored.
- **Payoff:** train the real harness, keep untrusted exec isolated, and scale each tier on its own.

---

# Section 1 — Agentic RL & the Agent Lightning reference architecture

## What is Agentic RL?

A paradigm shift from LLM-RL [2]: the LLM becomes an autonomous agent acting over many turns — a single-step MDP becomes a multi-turn POMDP.

- Act → observe → revise, retrying until solved.
- RL sharpens agentic skills: planning, tool use, memory, reasoning, reflection.
- Trajectory → loss-masked data (only the model's tokens are trained).

**Loop (a POMDP):** Act (write / call tool) → Run (in sandbox) → Observe (stdout / exit) → Revise (next action) → repeat.

- **Reward** = task passes its checker (e.g. unit tests).
- **GRPO** [3]: advantage = reward vs. group mean.
- Trajectory → tokenize + loss-mask → GRPO advantages → policy update.

## Agentic RL — use cases

The default coding-agent recipe, and anything where a checker can score a multi-turn attempt.

**Use cases:**

- **Coding agents** — write → run tests → read failure → fix (MBPP [15], SWE-style [8]).
- **Tool-using assistants** — shell, browser, API, MCP tools.
- **GPU-eval tasks** — write/repair a CUDA kernel, optimize a fused op; reward earned only by running on cuda (KernelBench [16]).
- **Long-horizon, stateful agents** — many turns against a persistent workspace.

**Why it is hard to wire:**

- The agent's harness (tools, context, control flow) is not the LLM — but it shapes every trajectory.
- Tool execution is untrusted and sometimes needs a GPU — it must be isolated from the trainer.
- Train/deploy gap: RL usually trains a bare LLM, then ships it inside a different harness.

**Agent Lightning's answer:** put the real harness in the loop.

## What is a harness?

The harness is the orchestration layer around the LLM — the code that turns a bare model into an agent. It owns:

- **the control loop** — call model → parse a tool call → run it → feed the observation back → repeat until done;
- **context** — system prompt, tool schemas, message history, truncation;
- **tools** — `write_file` / `bash` / `read_file`, browser, MCP servers — and where they execute (the sandbox).

The model chooses; the harness executes and decides what the model sees next. In RL it shapes every trajectory — so training the model without its harness leaves a gap.

**Examples:**

- **Claude Code CLI** (`claude -p`) — the shipped production agent.
- **OpenHands, SWE-agent, Aider** — coding agents.
- **AgentCore Strands** — AWS's managed model→tool loop (config, not code).
- **LangGraph / AutoGen** loops.
- **MiniClaudeHarness** — our ~1-file reimplementation (Demo 3).

*Harness control loop:* model ↔ tool (tool call → observation).

## Where the harness sits — data flow across the pipeline

The harness is the hub of the rollout loop: each turn it asks the inference engine (through the proxy) for the next action, runs it in the sandbox, then hands the finished trajectory + reward to the trainer (via the buffer).

**Data flow:**

- **Trainer** (Megatron · GPU) ↔ **Inference** (SGLang · GPU) via weight sync.
- Inference serves through the **Proxy** (LLM-API); each turn: prompt → tokens.
- **Harness** (agent loop · CPU) drives the loop; tool call / result goes to the **Sandbox** (CPU / GPU).
- **Buffer** receives trajectory + reward; Trainer consumes batches.

The harness is the hub — generation (proxy→inference), action (sandbox), learning (buffer→trainer). Only Trainer & Inference (and a GPU sandbox, when needed) sit on GPUs; the harness & proxy run on cheap CPU.

## What is a sandbox (in agentic RL)?

The sandbox is the isolated environment where the agent's tool actions actually run — write→run→read. It exists because model-generated code is untrusted and must be kept off the training cluster.

- **Isolation** — a crashing / malicious command can't touch the trainer or other episodes (one microVM/container per session).
- **Statefulness** — a persistent workspace across turns (files, processes).
- **The reward source** — it runs the checker (unit tests, `allclose` on cuda) that produces the scalar reward.
- **Right-sized** — CPU for shell/tests; a GPU only when the task needs one.

**Examples:**

- **AgentCore Code Interpreter** — serverless CPU microVM (this work).
- **AgentCore GPU Runtime** — capacity-provider g6 for CUDA tasks.
- **E2B [14], Modal, Daytona** — hosted code sandboxes.
- **Firecracker microVM [12], gVisor [13], Docker** — the isolation primitives.
- **SWE-bench [8] Docker eval harness** — per-task containers.

## Why a microVM? — Docker vs gVisor vs Firecracker vs E2B

The sandbox runs untrusted, model-written code multi-tenant, sometimes on a GPU — it needs a hard boundary and a cheap per-episode boot (the two pull opposite ways).

| Option | Isolation primitive | Pros | Cons |
|---|---|---|---|
| **Docker** | Shared host kernel (ns + cgroups) | Ubiquitous, fastest boot, densest | Weak boundary — kernel-escape surface; unsafe for adversarial code |
| **gVisor** [13] | User-space kernel over syscalls | Stronger than Docker, still light | Syscall-compat gaps; per-syscall overhead |
| **Firecracker** [12] | Own guest kernel via KVM | VM-grade boundary and ~125 ms boot | You operate the VMM / host fleet |
| **E2B** [14] | Firecracker microVM (managed SaaS) | Strong boundary, agent SDK, snapshots | 3rd-party cloud — egress, separate auth; no VPC/GPU |
| **AgentCore** [22] | Per-session microVM (AWS-managed) | Strong boundary + IAM/SigV4, VPC, in-account, GPU | AWS-only; session limits (8 h / 14 d) |

**Why microVM (not Docker):** untrusted rollouts need a kernel-level boundary that Firecracker gives at ~125 ms/episode. **Why AgentCore (not E2B):** both are microVMs, but AgentCore is native AWS — SigV4, private-VPC dial-back, in-account data, and a GPU path E2B lacks.

## Agent Lightning — the reference architecture

**Harnessed agentic RL (Microsoft) [1].** Modern agents are not standalone LLMs — they run inside an agent harness that manages tools, context, and control flow. Harnessed agentic RL puts that deploy-time harness directly in the training loop, narrowing the train/use gap.

- **Disaggregated architecture:** connect any agent to RL through an LLM endpoint proxy — the trainer never imports the agent's code.
- The proxy serves every completion from the policy under training and records each turn; a **Sample Adapter** turns trajectories into loss-masked training samples.
- This is the reference architecture we adopt. Adopted by verl Uni-Agent, AReaL 2.0, slime v0.3.0, Polar.

## Agent Lightning — components & data flow

- **Agents + Harness** — any black-box agent; just an HTTP client of the proxy. No fork.
- **LLM API Proxy** — the single seam: serves every completion from the training policy, records turns.
- **Rollout Controller** — dispatches episodes; **Sample Adapter** — trajectories → loss-masked samples.
- **Split engine + buffer** — Training consumes batches from the **Replay Buffer**; Inference serves the proxy; weights sync train → infer.

*Flow:* Agents with Harness → (Rollout API) → LLM API Proxy → Rollout Controller → Sample Adapter → Replay Buffer → Training Engine (Megatron) → (weight sync) → Inference Engine (SGLang) → serves policy under training back to the proxy.

---

# Section 2 — Scaling up the rollout

## Why scale up the sandbox? — throughput & task diversity

In agentic RL the trainer GPUs are rarely the bottleneck — the rollout / environment-execution layer is. Every gradient step needs fresh trajectories, produced by running the policy against real, verifiable environments. Two pressures push on the sandbox:

**1. High-throughput rollout**

- Trainer GPUs starve without a fast stream of rollouts; throughput is gated by how many sandboxed environments run in parallel.
- Modern systems decouple generation from training to keep trainers busy — AReaL [4], verl / HybridFlow [5], Agent Lightning [1].
- Large-scale RL post-training depends on this rollout infra — DeepSeek-R1 [6], Kimi K2 [7].

**2. Diversity of tasks / environments**

- Agent generality scales with the number and diversity of executable, verifiable tasks — SWE-bench [8], SWE-Gym [9], OpenHands [10], Environments Hub [11].
- Each task runs untrusted model-generated code, so it must execute in a fast, strongly-isolated sandbox — Firecracker microVM [12], gVisor [13], E2B [14].

**Punchline.** Scaling the sandbox/environment layer is what unlocks both high-throughput rollouts and task diversity — so it is the real constraint on agentic RL. AgentCore fits: serverless microVMs that scale on demand (one `session_id` each), with a GPU only when the task needs one.

## How to scale — the trainer sets the pace; size everything to D

RL post-training is rollout-bound: trainer GPUs are the costly resource and only earn their keep if a fast stream of fresh trajectories feeds them. Because rollout is async, sizing is throughput-matching with one anchor — the trainer's appetite.

- **Demand rate** `D = B/Tstep` samples/s. Batch `B` (capped by trainer GPU memory) and trainer GPU count set `D` — the input to every other tier.
- **Little's Law:** to hide per-episode latency `Le`, keep `C = D × Le` episodes in flight — this sizes harness & sandbox.
- **Staleness budget:** async runs `K` steps ahead; size to `D` within that budget.
- **Watch trainer-GPU utilization:** target ≈100% within budget; a dip means an upstream tier is the bottleneck.

**Method:** measure step ⇒ `D`; profile one episode ⇒ `C`; provision `Ninf`, `Nh`, `Nsbx`; run; add capacity at the tier pegged at 100% while trainer idles; repeat.

**Punchline — the GPU budget.** Only trainer, inference, and the GPU sandbox compete for GPUs. Harness and proxy scale on cheap CPU — off the GPU budget entirely. That is why they rarely need "scaling up" in the expensive sense.

## Scaling the sandbox — do you also scale inference?

Rollout is a 3-tier pipeline (harness → inference → sandbox); throughput is the slowest tier. So yes — size every tier to demand `D` — but not 1:1: while a tool runs in the sandbox the engine serves another episode, so one engine serves several (engines scale sub-linearly; sandboxes & harness ~ `C`).

*Scaled rollout:* `C` harnesses & sandboxes, a shared pool of `E` SGLang engines. A router demuxes by `sid`.

**Workload-dependent (think/act split):** light tools ⇒ inference-bound (add engines); heavy tools (big suites, GPU bench, browser) ⇒ env-bound (add sandboxes). **Method:** watch trainer-GPU util; add capacity where a tier pegs at 100%.

## Scaling the plumbing — buffer size & proxy robustness

Scaling `C` and `E` raises the rollout production rate. Two shared components must keep up, or they become the new bottleneck (or a correctness hazard).

**Replay / data buffer — yes, resize it:**

- It decouples the fast producer (rollout) from the trainer. Faster production ⇒ more completed-but-unconsumed episodes sit in flight, so raise capacity or producers block.
- **Rule of thumb:** hold ≳ `(K+1) × B` samples — the async staleness window of `K` steps plus the current group, and the `C` episodes still generating.
- But bigger is not better: an over-large buffer serves stale, off-policy data ⇒ drift. Size it to the staleness budget, then apply backpressure when full — don't just grow it.

**LLM-API proxy / router — harden it:**

- It is the fan-in/out point for all `C` streams. Make it stateless & horizontally scaled (replicas behind a load balancer), fully async, with connection pooling.
- Load-balance across the `E` engines (least-loaded / KV-cache-aware); health-check & fail over when an engine dies.
- Backpressure + queueing, per-request timeouts, and idempotent retries keyed by `session_id` so a retried turn is not double-counted.
- Emit queue-depth & per-engine load metrics — that is how you see which tier to scale next.

**Net:** buffer capacity is a staleness knob (size to `K`, then backpressure); the proxy is a throughput knob (scale out, load-balance, retry) — both sized to the same demand `D` as the engines and sandboxes.

---

# Section 3 — Use cases

## Use cases — where a harness + sandbox pay off

Any multi-turn task with a programmatic checker needs a harness (loop, tools, context) + an isolated sandbox (runs untrusted actions and the checker). Families:

| # | Use case | What the harness drives | Sandbox + reward checker |
|---|---|---|---|
| 1 | Coding & SWE | edit repo, run tests, read failures, fix | repo container; tests pass [15, 8] |
| 2 | Tool & computer use | shell / browser / API / MCP tools, RPA | OS / browser microVM; completion oracle |
| 3 | Systems & GPU kernels | author & optimize CUDA / fused ops | GPU sandbox; allclose + speedup [16] |
| 4 | Data, analytics & SQL | pandas / SQL / notebooks over real data | CPU sandbox + dataset; result-set match |
| 5 | Math & formal proof | Lean / Coq / SymPy derivation or proof | prover sandbox; proof compiles |
| 6 | Long-horizon ops | multi-step automation, deploy pipelines | persistent workspace; end-state assertion |
| 7 | Security & CTF | find, exploit, patch & harden | strongly-isolated microVM; vuln closed |
| 8 | Agentic RAG / research | search, read, synthesize, cite | networked read sandbox; QA / citation check |

Each needs a harness + an isolated sandbox.

## Use cases — concrete examples

Same shape every time: a task prompt the harness drives to completion, run in a sandbox, scored by a programmatic reward. A few made concrete (# maps to the catalog):

| # | Example task (prompt) | Harness drives | Sandbox | Reward signal |
|---|---|---|---|---|
| 1 | "Fix the failing test in this Flask repo." | edit repo, run pytest | CPU repo container | pytest all green [8] |
| 3 | "Write a fused-softmax CUDA kernel beating PyTorch." | edit, nvcc, benchmark | GPU sandbox | allclose + ≥1.5× speedup [16] |
| 4 | "Which region grew fastest last quarter? (sales.csv)" | pandas / SQL query | CPU + dataset | result-set == gold answer |
| 2 | "Complete and submit the signup form on this page." | browser: click / type | Browser microVM | completion oracle (state check) |
| 5 | "Prove n + 0 = n in Lean." | edit, run Lean | Prover sandbox | proof compiles |
| 7 | "Find the overflow, exploit it, read /flag." | shell, craft exploit | Isolated microVM | flag captured |
| 8 | "Answer with citations from these 10 PDFs." | search, read, cite | Networked read sandbox | answer + citation check |

## Use-case architectures — same skeleton, 3 slots change

Each is the same reference architecture (harness → proxy → split engine, with a sandbox + serve loop). Only the **orange slots** differ: harness tools, the sandbox, the reward checker. The invariant middle — proxy + split `train | buffer | inference` — is exactly the architecture shown earlier.

**1 — Coding & SWE:** tools = edit / bash; sandbox = repo CPU microVM; reward = tests pass.

**2 — Tool & computer use:** tools = browser / shell / MCP; sandbox = OS / browser microVM; reward = completion oracle.

**3 — Systems & GPU kernels:** tools = edit / nvcc; sandbox = GPU (g6) sandbox; reward = allclose + speedup. *(swaps the sandbox to a GPU.)*

**4 — Data, analytics & SQL:** tools = pandas / SQL; sandbox = CPU sandbox + dataset; reward = result-set match.

**5 — Math & formal proof:** tools = editor; sandbox = prover (Lean/Coq); reward = proof compiles.

**6 — Long-horizon ops:** tools = shell / deploy; sandbox = persistent workspace; reward = end-state assertion.

**7 — Security & CTF:** tools = exploit / shell; sandbox = isolated microVM (strongest isolation, adversarial code); reward = vuln closed / flag.

**8 — Agentic RAG / research:** tools = search / read; sandbox = networked read sandbox; reward = QA / citation check.

## Use cases — the common shape, and the axes a customer sizes

The shape is always the same: a verifiable checker (the reward) wrapped around untrusted, multi-turn execution — which is exactly what a harness + sandbox provide. What differs between customers is how each axis is sized:

**Axes that vary:**

- **Compute** — CPU shell/tests vs. a GPU sandbox (systems & ML tasks).
- **Horizon / state** — one-shot check vs. a long-horizon persistent workspace.
- **Isolation strength** — benign code vs. adversarial (security / CTF).
- **Harness fidelity** — a bespoke loop vs. ship-the-real-agent (Claude Code CLI).
- **Verifiability** — a crisp checker (tests, allclose) vs. a fuzzy LLM-judge.

**Why the two seams matter:**

- The harness shapes every trajectory — train it, don't approximate it.
- The sandbox keeps untrusted tool exec (and any GPU) off the trainer.
- Swapping the sandbox (CPU↔GPU) should not touch the harness.

**Punchline.** AgentCore spans all of these — serverless CPU microVM by default, a GPU provider only when the task needs one; harness in-process or on its own Runtime.

---

# Section 4 — slime, the agentic RL framework

## slime — an LLM post-training framework for RL scaling

slime (THUDM) [17, 18] connects Megatron (training) with SGLang (rollout) into one RL post-training stack. Docs: <https://thudm.github.io/slime/>.

**Two core capabilities:**

- **High-performance training** — Megatron parallelism (TP/PP/EP), dense and MoE.
- **Flexible data generation** — arbitrary rollout workflows via a custom generate interface over server-based SGLang engines.

**Design principles:**

- Three decoupled modules (train / rollout / buffer) — swap any one.
- **Hooks, not forks** — `--custom-generate-function-path`, `--custom-rm-path`.
- **Async by design** — generate rollout N+1 while training on N.

**Training modes supported:**

- **Algorithms:** SFT, PPO, GRPO, REINFORCE++-style.
- **Placement:** sync (colocate GPUs) or async (disjoint GPUs).
- **Rollouts:** single-turn RLVR and multi-turn agentic (tools, session-affinity routing).

**Use cases it targets:** Reasoning RLVR (math / code), agentic tool-use, and any task with a custom rollout + programmatic verifier — exactly the harness + sandbox shape.

## slime — architecture: three decoupled modules

Rollout generates data (+ rewards), the Data Buffer bridges, and training consumes batches then syncs weights back — a closed RL loop.

*Flow:* `--custom-generate` / `--custom-rm` hooks feed **Rollout** (SGLang engines + router, + reward / verifier) → rollout data → **Data Buffer** (prompts & trajectories) → sampled batches → **Training** (Megatron TP/PP/EP) → weight sync (update policy) back to rollout.

- **Rollout** — SGLang generation behind a router; custom generate builds trajectories, custom rm scores them.
- **Data Buffer** — owns prompt & rollout data; decouples generation cadence from training cadence.
- **Training** — Megatron optimizes on sampled batches, then pushes updated weights to the rollout engines.

---

# Section 5 — AWS Bedrock AgentCore

## AWS Bedrock AgentCore — a managed platform for agents at scale

Amazon Bedrock AgentCore [22] runs, secures & operates AI agents and tools — any framework, any model — with no infra to manage. Modular services used together or standalone. Docs: <https://docs.aws.amazon.com/bedrock-agentcore/>.

**Key services (high level):**

- **Runtime** — serverless, true session isolation (per-session microVM), fast cold start, long-running (async) sessions.
- **Code Interpreter** — isolated sandbox to execute code (Python/JS/TS).
- **Browser** — managed cloud browser for web actions.
- **Memory** — short- & long-term agent memory.
- **Gateway / Identity** — APIs→MCP tools; auth & access.
- **Observability** — OTEL traces of every step.

**Use cases:**

- Autonomous agents (support, coding, analysis) with isolated sessions.
- Tools & MCP servers from existing APIs / Lambda.
- Internal agent platforms with governed, shared services.

**Why it fits agentic RL.** Per-session microVM isolation + a code/tool sandbox + a container Runtime you own — exactly the two seams RL needs. Next: map these onto harness & sandbox.

## AgentCore for agentic RL — two roles, two compute modes

For RL we use AgentCore in two distinct roles — the same platform, two seams:

- **Harness** — host the agent loop in a container you own on a Runtime (drives turns, calls tools).
- **Sandbox** — the isolated place the policy's actions execute (write→run→read), off the training cluster.

**Two compute modes:**

- **Code Interpreter** [24] — serverless CPU microVM, ≤8-hr sessions (no GPU).
- **Runtime + GPU provider** [23] — your EC2 (e.g. g6 / L4), up to 14-day sessions.

**SDK seam:** `invoke("executeCode")` / `invoke_agent_runtime(ARN, sid)`.

## Using AgentCore as the harness — host the agent loop

Package your agent loop as a container and deploy it on a Runtime; the trainer reaches it with one `invoke_agent_runtime` call, one session per episode.

**What you do:**

1. Put the loop (context, tool schema, turn cap) in a container; expose the Runtime entrypoint.
2. Point completions at the proxy (`AnthropicAdapter`) so every "model" call is the policy under training.
3. slime's generate hook (`agentcore_runtime_generate`) invokes the Runtime; `session_id` = episode.
4. Harness calls its own sandbox for tool exec; returns turns.

**Why:** the harness that ships is the harness you train — no re-impl.

## Using AgentCore as the sandbox — CPU & GPU

One SDK seam, two backends. The harness is identical; only the sandbox target changes — swap CPU↔GPU without touching the loop.

**CPU sandbox — Code Interpreter.** Serverless microVM, Python/JS, ≤8-hr sessions, no GPU. Cheap, scales wide. Seam: `invoke("executeCode")`.

- **MBPP / SWE** — write code, run pytest; reward = tests pass [15, 8].
- **Data / SQL** — pandas over a dataset; result-set match.
- Tool & shell automation; end-state assertion.

**GPU sandbox — Runtime + GPU provider.** Your EC2 (g6 / L4), up to 14-day sessions, CUDA toolchain. For tasks that must run on a GPU. Seam: `invoke_agent_runtime(ARN, sid)`.

- **KernelBench** — write/optimize CUDA; reward = allclose + speedup [16].
- Fused-op / Triton kernels; correctness + latency.
- **GPU ML jobs** — train/eval a model; metric threshold.

Same reward shape everywhere: run untrusted actions in isolation, then a programmatic checker scores the result. The sandbox choice is just compute.

---

# Section 6 — Realizing it on slime + AgentCore

## slime — the trainer, run async/decoupled

slime (THUDM/slime) is the Training + Inference Engine of the reference architecture: Megatron for training + SGLang for generation, run decoupled.

**Async (decoupled):**

- Trainer GPUs and SGLang GPUs are disjoint pools (no `--colocate`).
- `train_async.py` generates rollout N+1 while training on N — the two pools run concurrently.
- A buffer decouples them: inference fills it with fresh trajectories; the trainer draws batches from it.
- Updated weights sync train → infer, so generation stays near-on-policy.

Overlap ⇒ trainer GPUs stay busy instead of idling while rollouts run.

**Extensible via hooks:** `--custom-generate-function-path` (rollout), `--custom-rm-path` (reward). No fork of slime needed.

## How to integrate AgentCore into slime — the recipe

Nothing in slime is forked. AgentCore drops into two of slime's seams — the rollout hook (harness) and, inside it, the sandbox.

**The four steps:**

1. **Proxy in front of SGLang** — run slime's `AnthropicAdapter` so the harness speaks the Messages API; every completion is the policy under training.
2. **Harness via the generate hook** — `--custom-generate-function-path` points at a `generate()` that drives the agent loop and records turns.
3. **Sandbox via AgentCore SDK** — each tool call is an `invoke_agent_runtime(ARN, session_id)` (SigV4); one session = one episode.
4. **Reward + samples** — Trajectory Manager emits loss-masked GRPO samples; `--custom-rm-path` scores them.

**Interfaces & connections:**

- **harness → proxy:** Anthropic Messages API (HTTP).
- **proxy → policy:** SGLang serve (in-cluster).
- **harness → sandbox:** SigV4 `invoke_agent_runtime`; boto3 sync → `asyncio.to_thread`.
- **proxy → trainer:** recorded turns → loss-masked samples.

**Placement.** Harness runs in-process or on its own AgentCore Runtime; sandbox is a CPU Code-Interpreter microVM or a GPU Runtime — swapped without touching slime.

## The three decouplings — mapping onto slime + AgentCore

Each reference component has a home; the three seams make RL work — seam ② (harness reaches the policy only through the proxy) is the core idea.

- **① Train ⊥ inference** — slime async, no `--colocate` (= AL Training vs. Inference Engine).
- **② Harness ⊥ inference, via the proxy** — slime's `AnthropicAdapter` is AL's LLM proxy; every "Claude" completion is the policy under training.
- **③ Harness ⊥ sandbox** — harness owns the loop; sandbox is the execution env ⇒ up to two AgentCore Runtimes.

*Mapping:* Megatron trainer —①weight sync→ SGLang inference —serve→ LLM API Proxy (AnthropicAdapter) —②Anthropic API→ Harness (Claude Code / mini) —③tool call / result→ Sandbox (Code Interp / GPU). The slime cluster (disjoint GPUs) holds trainer + inference + proxy.

## Why decouple — five services, each scaled to its own bottleneck

Design principle of this project: decouple into five independently-scalable services — scale each to its own bottleneck.

| Component | Scale it by | Bound by |
|---|---|---|
| Trainer (Megatron) | more train GPUs / TP·DP | gradient compute |
| Inference (SGLang) | more engines / GPUs | rollout token throughput |
| Proxy (AnthropicAdapter) | more replicas / shard by session | request fan-in (CPU/IO) |
| Harness (Claude Code / mini) | more Runtimes / CONC | agent-loop latency |
| Sandbox (Code Interp / GPU) | more sessions / GPU Runtimes | tool-exec (CPU or CUDA) |

**Why it is worth it:**

- **Fidelity** — train the real harness, zero agent changes.
- **Isolation** — untrusted tool exec off the trainer GPUs.
- **Right-sized GPUs** — sandbox gets a GPU only when the task does.

**What the seams buy:**

- ① overlap rollout N+1 with training of N.
- ② connect any black-box agent; trainer never imports its code.
- ③ swap sandbox (CPU↔GPU) without touching the harness.

## Integration is one hook; bound cost by capping turns

One seam: a custom generate hook. It drives the harness, records turns through the adapter, and returns loss-masked samples.

- `generate.generate` — our in-process `run_episode` loop.
- `agentcore_runtime_generate.generate` — mini harness on its own Runtime.

boto3 is sync → calls offloaded via `asyncio.to_thread`; one microVM per rollout.

**Bounding episode cost.** A step's total tokens drive three costs at once — rollout, ref-logprob, actor-train. A small turn cap is the biggest lever on wall-clock/step.

**One knob:** `EPISODE_MAX_TURNS` (default 3). Precedence: payload > env > default. One export re-caps the run.

*The generate hook:* Agent harness —Messages API→ AnthropicAdapter (proxy) —serve→ SGLang policy; tool call / result ↔ AgentCore sandbox; record turns → Trajectory Manager → loss-masked GRPO samples.

---

# Section 7 — Demos

## The demo matrix — two harnesses × two sandboxes

Two orthogonal axes; policy is always **Qwen3-4B** trained by slime. Four demos = two harnesses × two sandboxes.

| | CPU sandbox — MBPP | GPU sandbox — KernelBench |
|---|---|---|
| Our in-process harness | demo_a | demo_b |
| Mini harness on own Runtime | demo_a3 | demo_b3 |

**Columns = sandbox / use case:** CPU = coding/SWE (#1); GPU = systems & GPU kernels (#3). **Rows vary harness fidelity, not the use case:**

- **Row 1 (a/b)** — baseline: harness in the trainer process; cheapest, offline-capable.
- **Row 2 (a3/b3)** — clean disaggregation: harness *and* sandbox each their own AgentCore Runtime.

All six share async/decoupled `train_async.py`, GRPO, the checkpoint-deadlock fix, and save-only-at-end. Run in container `agentcore_p4` via `run.sh`.

## "a/b have no harness?" — having a harness ≠ a harness Runtime

A harness = the agent loop (prompt → parse → run a tool → feed the result back → repeat). a/b have one — `generate.generate`'s lightweight `run_episode` — it just runs in the trainer process, not on its own Runtime. All six demos have a harness; the rows differ only in which harness and where:

| Row | Harness | Where |
|---|---|---|
| a/b | our `run_episode` | in-process (trainer) |
| a3/b3 | MiniClaudeHarness | own Runtime #1 |

Progression of harness realism: in-process baseline → decoupled BYO harness on its own Runtime (a3/b3).

**So what's the point of a/b?** The baseline / control row:

- Isolate the spine from the integration. `a` runs fully offline (`SANDBOX=local`). If it trains (MBPP reward ~0.69), the whole spine — task, checker, dataset, GRPO, trainer, checkpoint — is proven before any Runtime/VPC complexity. So a bug in a3 is a harness bug, not a training bug.
- Cheapest, fastest iteration — no cold-starts, no capacity waits, no dial-back.
- `b` still proves the GPU sandbox — the KernelBench checker runs on cuda in a remote g6, the same env b3 uses. `b` ≠ "a on a GPU."

## Demo a — our in-process harness on a CPU sandbox

**Use case:** coding/SWE (#1), the baseline in-process harness. **What it is:** the policy solves MBPP — write `solution.py` → run tests → read failure → fix, over many turns. Harness = our `run_episode` loop in the trainer process; sandbox = a CPU Code-Interpreter microVM. The only demo that also runs fully offline (`SANDBOX=local`).

*Data flow (slime — async, 4 train + 4 infer):* harness `run_episode` (in trainer) ↔ sandbox Code-Interpreter microVM (CPU) via write / run, stdout/exit; harness → model turns → SGLang policy; loss-masked samples → Megatron trainer → weight sync → SGLang.

harness ⊥ sandbox already: the loop is in the trainer, the microVM only executes tools + the checker. Sandbox class: `AgentCoreSandbox`. Box GPUs 4+4+0.

## Demo a — what the CPU microVM contains (sandbox only)

The agent loop runs in the trainer process; the microVM is only the sandbox — it holds the workspace, runs the tools, and runs the checker.

```python
sandbox = make_sandbox(os.environ["SANDBOX"])
async with sandbox as sb:               # 1 microVM
    traj = await run_episode(policy,    # loop is
                   sb, task, ...)       # IN-PROCESS
    reward = traj.reward          # checker ran in sb
```

The `run_episode` loop lives in the trainer; only tool execution and the checker touch the microVM. `SANDBOX=local` runs it fully offline (no microVM at all).

**Inside the microVM (sandbox):**

- **Workspace:** `PROBLEM.md`, agent-written `solution.py`, `run_checks.py`.
- **Tool exec:** `write_file` / `bash` / `read_file` act on this FS.
- **Reward:** the MBPP checker runs here.
- **Not here:** the agent loop — it is in the trainer.

**Cheapest to stand up.** No serving container, no CLI, no dial-back — and `SANDBOX=local` removes the microVM entirely for offline debugging.

## Demo a, generalized — one harness across the agentic-RL task zoo

The *same* in-process `run_episode` harness generalizes across task families — only the **prompt**, the **seed workspace**, and the **checker** change. All eleven families are **built and running** on the one CPU harness (`agentcore_slime_demo/task_families.py`), each with a deterministic, stdlib-only RLVR checker (no LLM judge):

| Task family | Seed / benchmark shape | Reward checker (exit code) | Status |
|---|---|---|---|
| Coding (unit tests) | MBPP [15], HumanEval | hidden asserts pass | **built** |
| SWE (bug-fix) | buggy module + failing suite [8] | test suite green | **built** |
| Tool-use | task + provided `tools.py` | result = gold ∧ tools called | **built** |
| Data / SQL | schema + seeded SQLite DB | result-set = gold rows | **built** |
| Math | word problem | `|answer − gold| ≤ ε` | **built** |
| Security / CTF | a verifier `check(flag)` | `check(flag)` True | **built** |
| Agentic RAG | question + local `./docs` corpus | answer = grounded fact | **built** |
| Regex synthesis | positive + negative examples | `pattern` full-matches all pos, no neg | **built** |
| Planning / search | discrete puzzle (queens, subset-sum) | hidden `verify(plan)` True | **built** |
| Config repair | broken config + constraints | every constraint satisfied | **built** |
| Data / ETL | `input.json` + transform spec | `output` = gold transform | **built** |

Select one with `FAMILY=sql bash run.sh`; datasets come from `build_family_datasets.py` (87 seed tasks across the ten families beyond MBPP). The harness writes each family's seed files + checker, asks the policy for a submission, writes it to the family's artifact (`solution.py`, or `query.sql` for SQL), runs the checker command, and turns the exit code into the reward. Most families reuse the MBPP assert-checker (`task.assert_runner`) — the agent's `solution.py` just sets a module variable the asserts read; only SQL needs a custom runner (it executes the query against an in-memory DB). Same `generate.py` + Sample contract for all — `metadata.task.family` dispatches to the right `Task` class. Every family passes its checker end-to-end offline (`SANDBOX=local`, `tests/test_smoke.py`) **and** on the remote AgentCore Runtime (`smoke_remote_families.py`), and the full set trains one-by-one on real slime (`run_family_campaign.sh`).

## Demo b — same loop, sandbox swapped for a remote GPU Runtime

**Use case:** systems & GPU kernels (#3). **What it is:** identical in-process `run_episode` loop as demo_a — only the sandbox changes. KernelBench: write a `ModelNew` that matches a torch reference on cuda. The checker runs on a separate remote GPU sandbox Runtime (g6) — not the trainer's GPUs. The harness stays in-process, so the sandbox is the only AgentCore Runtime.

*Data flow (slime — async, 4 train + 4 infer):* harness `run_episode` (in trainer) ↔ sandbox remote GPU Runtime (g6, only Runtime) via write / run, cuda result; harness → model turns → SGLang policy; loss-masked samples → Megatron trainer → weight sync.

Sandbox class: `AgentCoreRuntimeSandbox`. The checker's GPU stays remote (a disposable g6), so the box stays 4+4+0 — no local env GPUs.

## Demo b — what the GPU microVM contains

Same as demo_a except the microVM is a GPU Runtime: it builds and runs the candidate CUDA kernel and scores allclose against the torch reference.

```python
os.environ["SANDBOX"] = "agentcore_runtime"
sandbox = make_sandbox(...)   # ARN -> GPU g6
async with sandbox as sb:            # microVM
    traj = await run_episode(policy, # loop still
                   sb, task, ...)    # IN-PROCESS
    reward = traj.reward   # allclose on cuda, in sb
```

Zero loop change vs. demo_a: only `SANDBOX` (and the ARN) differ.

**Inside the GPU microVM (sandbox):**

- **Workspace:** torch reference, agent-written `ModelNew`.
- **Tool exec:** build + run the kernel on cuda.
- **Reward:** allclose + speed vs. the reference, here on g6.

**Why remote GPU.** Untrusted CUDA never shares VRAM/host with training state; the checker's GPU is disposable and off the trainer's 4+4.

## Demo a3 — mini harness on its own Runtime (CPU sandbox)

**Use case:** coding/SWE (#1), a bring-your-own harness on its own Runtime. **What it is:** the clean disaggregated target — a ~1-file agent (MiniClaudeHarness) runs on its own AgentCore Runtime #1, reproducing Claude Code's loop (call model → pick a tool → run it), driving a separate CPU sandbox Runtime #2. No Node, no CLI; it dials the adapter back over the private VPC.

*Data flow (slime — async):* harness Runtime #1 (dial-back, VPC) = MiniClaudeHarness —Messages API→ `AnthropicAdapter` —serve→ SGLang policy → Megatron trainer → weight sync; tool call / result ↔ sandbox Runtime #2 (CPU microVM); record → Trajectory Manager → samples.

Files: `harness_runtime_agent.py` (#1, hosts `mini_agent.py`), `sandbox_server_agent.py` (#2). Box GPUs 4+4+0. Validated live (2026-09-07).

## Demo a3 — two microVMs, tools bridged by ARN

The clean disaggregated target: two microVMs. Runtime #1 hosts the agent loop; Runtime #2 is the sandbox. Tool calls cross from #1 to #2 by ARN.

```python
sandbox = AgentCoreRuntimeSandbox(       # Runtime #2
             runtime_arn, session_id)
async with sandbox as sb:              # microVM #2
    harness = MiniClaudeHarness(max_turns) # on #1
    result = await harness.run(sb, ...) # tools->#2
```

The loop runs on Runtime #1; each tool call is dispatched to Runtime #2 over `invoke_agent_runtime` (the `mcp__sandbox__*` bridge). One `session_id` = one episode.

**Runtime #1 microVM (harness):**

- MiniClaudeHarness ~1-file loop (call model → pick tool → run).
- No Node, no CLI; dials the adapter back over the VPC.

**Runtime #2 microVM (sandbox):**

- Workspace + `write_file` / `bash` / `read_file`.
- Reward: the MBPP checker runs here (CPU).

**Why.** The clean end-state — harness *and* sandbox each their own AgentCore Runtime, so each tier scales, fails, and is secured on its own.

## Demo b3 — identical code, sandbox on the GPU Runtime

**Use case:** systems & GPU kernels (#3). **What it is:** identical code to a3 — the mini harness on Runtime #1 drives a sandbox Runtime #2; the only change is the sandbox ARN now points at a GPU (g6) Runtime. The KernelBench checker runs on cuda there, off the trainer's GPUs.

*Data flow (slime — async):* harness Runtime #1 (dial-back, VPC) = MiniClaudeHarness —Messages API→ `AnthropicAdapter` —serve→ SGLang policy → Megatron trainer → weight sync; tool call / cuda result ↔ sandbox Runtime #2 (GPU microVM, g6); record → Trajectory Manager → samples.

Identical code to a3; only the sandbox ARN differs (GPU g6). Box GPUs 4+4+0 — the checker's GPU is remote and disposable.

## Demo b3 — the cleanest scale-and-secure story

b3 = a3's topology with a GPU sandbox: harness and sandbox are separate Runtimes, and the GPU is remote — so all four axes move independently.

**What changes vs. a3:**

- Sandbox ARN → a GPU (g6) Runtime.
- Checker runs allclose on cuda.
- Nothing else — same harness, same bridge.

**Why it is the target.** Fidelity (bring-your-own harness) + isolation (untrusted CUDA off the trainer) + right-sized GPUs (sandbox gets a GPU only when the task does) + independent scaling of every tier. Box stays 4+4+0.

---

# Section 8 — Demos: comparison & scaling

## In-process harness vs. two-Runtime target — side by side

The baseline (a/b) keeps the harness in the trainer process, so the sandbox is the only AgentCore Runtime. The target — a3/b3 — lifts the harness onto its own Runtime, so harness *and* sandbox are each an AgentCore Runtime and each scales, fails, and is secured on its own.

| | a/b — in-process | a3/b3 — two Runtimes |
|---|---|---|
| **Topology** | harness in trainer, one sandbox Runtime | harness #1 ⊥ sandbox #2 |
| **Tool calls** | in-process → the sandbox Runtime | bridged to #2 (`mcp__sandbox__*`) |
| **Reward** | checker on the sandbox Runtime | checker on the sandbox Runtime |
| **Scaling** | harness tied to the trainer | each tier replicates independently |
| **Blast radius** | harness shares the trainer | isolated per Runtime |
| **Why** | fewest moving parts for bring-up | fidelity + independent scale/security |

## Replicating the demos — one adapter, C concurrent episodes

The async engine drives many episodes concurrently against a pool of SGLang engines; one adapter multiplexes all sessions. Each tier replicates independently.

```
C : E : C  =  CONC : 1 : CONC  ──(CONC=4)──▶  4 : 1 : 4
```

where engines `E = ROLLOUT_GPUS / rollout-num-gpus-per-engine (=2)` and concurrent episodes `C = CONC × E`.

| demo | ROLLOUT_GPUS | E | CONC | C | GPU split |
|---|---|---|---|---|---|
| a / a3 | 4 | 2 | 4 | 8 | 4 train + 4 infer + 0 env |
| b (our harness) | 4 | 2 | 4 | 8 | 4 train + 4 infer + 0 env (remote g6) |
| b3 | 4 | 2 | 4 | 8 | 4 train + 4 infer + 0 env (remote g6) |

**Scale out:** raise `CONC` for more harness sessions + sandboxes per engine; raise `ROLLOUT_GPUS` for more engines. No demo competes for box GPUs — every row keeps the checker's GPU remote (a disposable g6) or CPU, so the box stays 4+4.

## Demo b GPU topology — 2 engines on the box, 1 sandbox on the g6

Two separate machines. The table above is the ceiling; the single-g6 reality of b/b3 pins the sandbox tier to one concurrent session, so we run them at CONC=1.

**Trainer box (8 GPUs) — generation & training:**

- `--rollout-num-gpus 4`, `--rollout-num-gpus-per-engine 2` ⇒ `E = 4/2 = 2` SGLang engines (each TP = 2).
- Split: 4 Megatron actors + 4 SGLang inference + 0 env.
- No SGLang runs on the g6 — the g6 only runs the torch/CUDA checker.

**GPU sandbox = Runtime #2 (the g6):**

- GPU is not on a microVM: it is AgentCore's **Instances** compute type — EC2 via a capacity provider. A session maps 1:1 to an EC2 instance.
- Our CP holds one g6.xlarge (1 L4) ⇒ 1 concurrent GPU-sandbox session.

**Concurrency math:**

- Generation cap = `Semaphore(CONC × E)` = `CONC × 2`.
- At CONC=1: 2 in-flight generations, but episodes serialize on the one L4 when they hit the checker.
- So CONC=1 matches generation to the single-GPU sandbox and dodges the g6 capacity-shortage fragility seen in earlier runs.

**To get more sandbox concurrency:**

- Scale the capacity provider to N instances ⇒ ~N isolated sessions.
- Or put many agents in one session — they share the instance GPU (no isolation). See the next slide for the full menu.

## The GPU-sandbox limitation — and how you'd share one L4

**Key fact (AWS docs):** AgentCore GPU is not a microVM feature. It exists only on the **Instances** compute type (EC2 via a capacity provider), and a session maps 1:1 to an EC2 instance. There is no documented GPU time-slicing across sessions: one session = one instance = its whole GPU(s).

**What AgentCore does give you:**

- More instances ⇒ more isolated sessions (1 GPU each). Clean, secure, but N× the cost.
- Many agents in one session: "all agents on the same instance share access to its GPUs." Native, free concurrency — but no isolation between agents (shared FS + GPU; "must be mutually trusted"). This is the lever we are not using at CONC=1.

**Ways to share one physical GPU:**

| mechanism | isolation | AgentCore? |
|---|---|---|
| Agents / session | none | native |
| NVIDIA MPS | none | self-mgd |
| Time-slicing | none | self-mgd |
| K8s device-plugin | none | self-mgd |
| Ray `num_gpus<1` | none | self-mgd |
| NVIDIA MIG | hard | no (L4) |

**The hardware catch.** L4 (g6) does not support MIG — only A100 / A30 / H100-class can be hardware-partitioned. So on our L4 there is no hard-isolated fractioning; only cooperative sharing is possible.

**Takeaway:** isolated scaling = more instances; fractional sharing needs either many-agents-per-session (no isolation) or your own Ray/EKS fleet.

---

# Section 9 — Networking, isolation & lifecycle

## Networking & security: dial back over the VPC — no public port

The "no public port" rule follows from one fact: the harness dials back to the private adapter. Everything else follows.

| Component | Hosting | Network | Why |
|---|---|---|---|
| Trainer + infer + AnthropicAdapter | this EC2 box | private VPC | only `ADAPTER_PORT` open, only to the runtime SG |
| Harness #1 | managed serverless | networkMode=VPC | the only thing that dials back → VPC egress + one SG rule |
| CPU sandbox (A3) | managed serverless | PUBLIC ok | reached via IAM invoke; never dials back |
| GPU sandbox (B3) | capacity provider g6 | CP's VPC | reached via IAM, not a private IP |

- **Pull, don't expose.** Every sandbox is reached with SigV4 `invoke_agent_runtime` — no inbound port on any sandbox; bearer token never in a Runtime env.
- **One inbound rule, ever:** tcp/`ADAPTER_PORT` on the box SG, source = the runtime SG (not a CIDR).
- **Two GPU pools stay separate:** train/infer GPUs (box) vs. the sandbox's g6 — untrusted CUDA never shares VRAM/host with training state.

## The VPC-mode trap: private endpoints + AZ pinning (or it never starts)

A `networkMode=VPC` Runtime has no internet route, so it reaches AWS only through private VPC endpoints. Miss one — or land the ENI in the wrong AZ — and create fails with a null reason and an empty log group: nothing started, nothing to read.

| Endpoint | Type | Why the harness needs it |
|---|---|---|
| `ecr.api` + `ecr.dkr` | Interface | pull the image (auth + registry) |
| `s3` | Gateway | ECR image layers live in S3 |
| `logs` | Interface | ship container logs to CloudWatch |
| `sts` | Interface | assume the execution role (SigV4) |
| `bedrock-agentcore` | Interface | the Runtime's phone-home |

- One tcp/443 ingress from the VPC CIDR on the endpoint SG. `PrivateDnsEnabled` overrides ECR DNS VPC-wide, so without it the box's own ECR access breaks.
- **Gotcha 1:** a default VPC can have two subnets in one AZ ⇒ `DuplicateSubnetsInSameZone`. De-dupe.
- **Gotcha 2:** not every AZ serves `bedrock-agentcore` (e.g. not us-west-2d). Pin the harness to subnets whose AZ is served by every endpoint.

## "Can we just expose the adapter publicly?" — No

The `AnthropicAdapter` is not "just a proxy": it fronts SGLang serving the policy under training, and keys GRPO samples on the Bearer `session_id`. Exposing it to the internet exposes:

- **Your training loop's inference server** — free compute, model-behavior extraction, DoS / cost abuse of GPU time.
- **A trajectory-injection surface** — a crafted request with a guessed `session_id` can push fake trajectories into GRPO and quietly corrupt the run. This is the scary one.
- **Weak auth by design** — the `session_id` bearer assumes a trusted private network; it is not real authN/Z.

**Verdict.** No — private VPC + SG, or PrivateLink. A public adapter also contradicts the pitch ("bring your own harness and keep it private"). Narrow exception if truly blocked: a single client /32 over TLS with a real token — "reachable from one place," not "public."

## Routing, isolation & lifecycle — everything keyed by session_id

C episodes share one adapter, router, and trajectory buffer — correctness rests on one invariant: everything is keyed by a per-episode `session_id`.

- **Routing:** every `/v1/messages` carries `Bearer <sid>` → that session's turn stream; `TrajectoryManager` keeps one token seq + `loss_mask` per `sid`.
- **Sandbox:** one `runtimeSessionId` per episode pins one warm microVM, isolated across sessions. Gotcha: it must be 33–128 chars (an 18-char id silently fails).
- **Lifecycle:** first invoke cold-launches a microVM (a g6 for GPU rows); same `sid` re-routes to the same warm VM; idle-out (≈600 s) Stops it; a later invoke resumes on fresh compute (workspace re-written — fault recovery).
- **Guarantee:** distinct `sid` ⇒ distinct Bearer, sandbox session and trajectory bucket — no cross-wiring (hence the adapter must stay private).

---

## Summary

- **Agentic RL:** the policy learns by acting over many turns; trajectories become loss-masked GRPO data.
- **Agent Lightning** is the reference architecture — harnessed agentic RL via an LLM endpoint proxy; the real harness joins training.
- **slime** is the Training+Inference Engine (async/decoupled overlaps rollout N+1 with training of N); `AnthropicAdapter` is the proxy.
- **AgentCore** is the execution sandbox, and can also host the harness — up to two Runtimes.
- One shared **generate hook** powers a 3×2 demo matrix; only the GPU sandbox spends the GPU budget — harness and proxy scale on CPU.

---

## References

1. Agent Lightning v1.0 — Towards Harnessed Agentic RL. <https://arxiv.org/abs/2508.03680>
2. The Landscape of Agentic RL — A Survey. <https://arxiv.org/abs/2509.02547>
3. GRPO — DeepSeekMath. <https://arxiv.org/abs/2402.03300>
4. AReaL — Large-Scale Asynchronous RL for Language Reasoning. <https://arxiv.org/abs/2505.24298>
5. verl / HybridFlow — Flexible & Efficient RLHF Framework. <https://arxiv.org/abs/2409.19256>
6. DeepSeek-R1 — RL at scale. <https://arxiv.org/abs/2501.12948>
7. Kimi K2 — Open Agentic Intelligence. <https://arxiv.org/abs/2507.20534>
8. SWE-bench. <https://arxiv.org/abs/2310.06770>
9. SWE-Gym. <https://arxiv.org/abs/2412.21139>
10. OpenHands. <https://arxiv.org/abs/2407.16741>
11. Environments Hub (Prime Intellect). <https://www.primeintellect.ai/blog/environments>
12. Firecracker — Lightweight Virtualization (NSDI 2020). <https://www.amazon.science/publications/firecracker-lightweight-virtualization-for-serverless-applications>
13. gVisor. <https://gvisor.dev>
14. E2B. <https://e2b.dev>
15. MBPP — Program Synthesis with Large LMs. <https://arxiv.org/abs/2108.07732>
16. KernelBench. <https://huggingface.co/datasets/ScalingIntelligence/KernelBench>
17. slime (RL trainer). <https://github.com/THUDM/slime>
18. slime documentation. <https://thudm.github.io/slime/>
19. SGLang (inference). <https://github.com/sgl-project/sglang>
20. Agent Lightning (code). <https://github.com/microsoft/agent-lightning>
21. Claude Code. <https://code.claude.com/docs/en/overview>
22. Amazon Bedrock AgentCore — Developer Guide. <https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/what-is-bedrock-agentcore.html>
23. AgentCore Runtime. <https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime.html>
24. AgentCore Code Interpreter. <https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/code-interpreter-tool.html>
25. AgentCore pricing. <https://aws.amazon.com/bedrock/agentcore/pricing/>

---

*Thank you — Questions?*
