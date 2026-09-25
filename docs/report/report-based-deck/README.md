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

## The demos — 12, one design

Policy always **Qwen3-4B**, harness **in-process** (except `cpu_branching`), AgentCore the
**sandbox**; only the *substrate* and *task* vary. Each demo below is a **self-contained,
standalone recipe** — its own dir with a `README.md`, vendored `agentcore_slime_demo/`, own
`deploy/`, `run.sh`, and `train.sh`. Links go to each demo's own README.

### CPU (8)

| Demo | Task | Real use case |
|---|---|---|
| [cpu_tasks](../../../code/demos/demo_cpu_tasks/README.md) | `DATASET=` mbpp (default) · math · gsm8k · math500 · deepscaler · humaneval · gretelsql · hotpotqa · xlam · mcp · a comma list (`gsm8k,gretelsql`) · `mixed` (math+SQL+CTF+datasci) | the reference demo: coding, math, SQL, RAG and tool-use agents from one folder, alone or mixed into one generalist stream |
| [cpu_scale_lanes](../../../code/demos/demo_cpu_scale_lanes/README.md) | DeepScaleR (fixed) | 4 × 4 × 4 lanes: harness worker : SGLang engine : microVM (was `cpu_harness_scale`) |
| [cpu_imagepool](../../../code/demos/demo_cpu_imagepool/README.md) | datasci (pandas/scipy/sklearn) | data-science agent with heavy, incompatible deps → per-image Runtimes |
| [cpu_terminalbench](../../../code/demos/demo_cpu_terminalbench/README.md) | Terminal-Bench (~229 task images) | terminal / DevOps agent; one Runtime per task, fresh microVM per episode |
| [cpu_branching](../../../code/demos/demo_cpu_branching/README.md) | Claude Code CLI | RL on a **real production coding agent** (subagents + summarization) |
| [cpu_customruntime](../../../code/demos/demo_cpu_customruntime/README.md) | DeepScaleR | managed Code Interpreter vs a custom Runtime |
| [cpu_bigimage](../../../code/demos/demo_cpu_bigimage/README.md) | DeepScaleR, ~4 GB image | an image past the microVM cap, on a warm CPU Instance |
| [cpu_heartbeat](../../../code/demos/demo_cpu_heartbeat/README.md) | heartbeat | keep a warm sandbox alive across the weight-sync pause |

`cpu_tasks` replaces ten per-dataset folders (`demo_cpu_mbpp`, `demo_cpu_math`,
`demo_cpu_gsm8k` … `demo_cpu_mcp`, merged 2026-09-23) and `demo_cpu_mixtask` (now
`DATASET=mixed`, merged 2026-09-25) — they differed only in their data.

### GPU (4)

| Demo | Task | Real use case |
|---|---|---|
| [gpu_single](../../../code/demos/demo_gpu_single/README.md) | KernelBench | CUDA-kernel agent on a real GPU (**trained**) |
| [gpu_customruntime](../../../code/demos/demo_gpu_customruntime/README.md) | KernelBench | GPU available only via a custom Runtime |
| [gpu_snapshot](../../../code/demos/demo_gpu_snapshot/README.md) | snapshot | survive a GPU-instance recycle (app-level S3 snapshot) |
| [gpu_farm](../../../code/demos/demo_gpu_farm/README.md) | KernelBench (Ray farm) | `FARM=fractional`: two episodes share ONE GPU (0.5 + 0.5); `FARM=whole`: one whole GPU per episode across two GPUs |

**Status (honest):** only `cpu_tasks` (`DATASET=mbpp`) and `gpu_single` carry ~10-step
training curves; the rest are 10-step (`NR=11`) smoke runs or ready-to-run recipes
(`cpu_terminalbench` has no run yet). See each demo's README for specifics.
