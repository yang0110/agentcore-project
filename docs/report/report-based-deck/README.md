# AgentCore as the Sandbox in Agentic RL — deck

Source of truth is **`deck.md`** (pandoc → Beamer). Rebuild with **`./build_deck.sh`**
(needs `pandoc`, `xelatex`, DejaVu fonts) → produces `report-based-deck.{tex,pdf}`.
Do **not** hand-edit the `.tex` — it is generated. The deck stands on its own; it was
seeded from `../agentcore_slime_report.md`.

## What this is about

**Agentic RL.** A policy acts over *many turns* against an executable environment
(write code, run it, read the result, try again). A **programmatic checker** — not an
LLM judge — turns the outcome into a scalar reward (**RLVR**: reward from verifiable
rules). This is the dominant frontier-lab recipe: untrusted, model-generated output runs
in an **isolated sandbox**, and a verifier produces the reward (Kimi K2 reports >10,000
concurrent sandboxes).

**AgentCore as the sandbox.** AWS Bedrock **AgentCore Runtime** is the isolated,
per-episode execution box:

- **Runtime** — a versioned container behind an ARN; the deployable unit, no per-user state.
- **Session** (`runtimeSessionId`) — one episode; first invoke cold-starts a **microVM**
  (own kernel/CPU/mem/FS, sanitized on termination), same id re-routes to that warm box.
- **Cardinality 1:N:1** — one Runtime → many sessions → one microVM each. GPU work runs on
  **Instances** (EC2 in your account); the serverless microVM is CPU-only.

**Integration with slime.** [slime](https://github.com/THUDM/slime) (Megatron + SGLang) is
the trainer. The agent loop plugs into slime's **custom generation hook** — one
`generate()` per episode, harness **in-process** (warm), each tool call → one SigV4
`invoke_agent_runtime` to the sandbox, the in-sandbox checker gives the reward. **CPU↔GPU
is one env var** (`SANDBOX=`). See the deck for the call path, TITO, and the key code.

## The demos — 21, one design

Policy always **Qwen3-4B**, harness **in-process** (except `cpu_branching`), AgentCore the
**sandbox**; only the *substrate* and *task* vary. Each demo below is a **self-contained,
standalone recipe** — its own dir with a `README.md`, vendored `agentcore_slime_demo/`, own
`deploy/`, `run.sh`, and `train.sh`. Links go to each demo's own README.

### Canonical (6)

| Demo | Task | Real use case |
|---|---|---|
| [cpu_single](../../../code/demos/demo_cpu_single/README.md) | MBPP repair | test-driven coding (the reference demo, **trained**) |
| [cpu_math](../../../code/demos/demo_cpu_math/README.md) | math words | math post-training |
| [gpu_single](../../../code/demos/demo_gpu_single/README.md) | KernelBench | CUDA-kernel agent on a real GPU (**trained**) |
| [gpu_shared](../../../code/demos/demo_gpu_shared/README.md) | KernelBench (Ray farm) | GPU packing when the fleet is scarce/costly |
| [cpu_quiesce](../../../code/demos/demo_cpu_quiesce/README.md) | heartbeat | keep a warm sandbox alive across the weight-sync pause |
| [gpu_snapshot](../../../code/demos/demo_gpu_snapshot/README.md) | snapshot | survive a GPU-instance recycle (app-level S3 snapshot) |

### Per-task CPU fan-out (8, real HF datasets)

| Demo | Task | Real use case |
|---|---|---|
| [cpu_gsm8k](../../../code/demos/demo_cpu_gsm8k/README.md) | GSM8K | grade-school math |
| [cpu_math500](../../../code/demos/demo_cpu_math500/README.md) | MATH-500 | competition math |
| [cpu_deepscaler](../../../code/demos/demo_cpu_deepscaler/README.md) | DeepScaleR | scaled math reasoning |
| [cpu_humaneval](../../../code/demos/demo_cpu_humaneval/README.md) | HumanEval | function-synthesis coding |
| [cpu_gretelsql](../../../code/demos/demo_cpu_gretelsql/README.md) | Gretel SQL | text-to-SQL analytics |
| [cpu_hotpotqa](../../../code/demos/demo_cpu_hotpotqa/README.md) | HotpotQA | multi-hop RAG / search |
| [cpu_xlam](../../../code/demos/demo_cpu_xlam/README.md) | xLAM-60k | function-calling agent |
| [cpu_mcp](../../../code/demos/demo_cpu_mcp/README.md) | ToolACE (MCP) | MCP tool-schema agent |

### Scale a new dimension (2)

| Demo | Task | Real use case |
|---|---|---|
| [cpu_harness_scale](../../../code/demos/demo_cpu_harness_scale/README.md) | DeepScaleR (fixed) | scale the rollout driver (the forgotten 3rd tier) |
| [cpu_mixtask](../../../code/demos/demo_cpu_mixtask/README.md) | math+SQL+CTF+tool | one generalist policy across many families |

### Substrate & design variants (5)

| Demo | Task | Real use case |
|---|---|---|
| [cpu_branching](../../../code/demos/demo_cpu_branching/README.md) | Claude Code CLI | RL on a **real production coding agent** (subagents + summarization) |
| [cpu_imagepool](../../../code/demos/demo_cpu_imagepool/README.md) | datasci (pandas/scipy/sklearn) | data-science agent with heavy, incompatible deps → per-image Runtimes |
| [cpu_customruntime](../../../code/demos/demo_cpu_customruntime/README.md) | trivial | managed Code Interpreter vs a custom Runtime (capability compare) |
| [gpu_customruntime](../../../code/demos/demo_gpu_customruntime/README.md) | trivial | GPU available only via a custom Runtime (capability compare) |
| [gpu_local](../../../code/demos/demo_gpu_local/README.md) | KernelBench | local whole-GPU-per-session Ray farm |

**Status (honest):** only `cpu_single` and `gpu_single` carry ~10-step training curves; the
rest are `NR=3` smoke runs or ready-to-run recipes. See each demo's README for specifics.
