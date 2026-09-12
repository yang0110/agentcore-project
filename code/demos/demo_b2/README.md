# Demo B2 — KernelBench (GPU) with the real Claude Code CLI, the two-Runtime target

**Demo B2 = Demo B3 + the real Claude Code CLI.** Same task as [Demo B](../demo_b/) (KernelBench
CUDA ops; the checker runs `torch` on a GPU) and the **same two-Runtime wiring as
[B3](../demo_b3/)** — but the **harness** (the agent loop) is the real **Claude Code**
(`claude -p`), not our mini `MiniClaudeHarness`.

> **Harness ≠ sandbox — two distinct AgentCore Runtimes.** The **harness** owns the agent loop
> (tool selection, context, control flow) and calls the LLM; the **sandbox** is the execution
> environment where its tool actions run. B2 runs them on **two separate AgentCore Runtimes**:
> - **Harness Runtime #1** — the real `claude -p` (`Dockerfile.harness.claude`, `HARNESS_KIND=claude`).
>   Its built-in Bash/Edit/Write/Read/Web tools are **disabled**; its only tools are the
>   `mcp__sandbox__*` bridge (`sandbox_mcp.py` → FastMCP over localhost), which forwards every
>   file/shell action to Runtime #2. CPU-only, `networkMode=VPC` so it can dial the adapter back.
> - **Sandbox Runtime #2** — the GPU-backed `sandbox_server_agent.py` on a **g6 capacity provider**
>   (`Dockerfile.sandbox.gpu`), where the KernelBench `torch`/CUDA checker runs. **This is the exact
>   same GPU sandbox Runtime B3 uses** (`deploy_two_runtime.py all`), with the CUDA `torch` stack
>   installed onto its volume by the one-time `{"mode":"prepare"}` invoke.
>
> This replaces the earlier bring-up shortcut (the CLI co-located *inside* the GPU sandbox via
> `claude_code_generate.py`). B2 now drives the **same hook as B3**
> (`agentcore_slime_demo.agentcore_runtime_generate.generate`); `HARNESS_KIND=claude` is the only
> switch that selects the real CLI over the mini agent. See
> [docs/harness-sandbox-decoupling.md](../../../docs/harness-sandbox-decoupling.md) §3, §6, §7.

The **Qwen3-4B policy is still trained by slime**. Claude Code never talks to a hosted Anthropic
model: slime's **AnthropicAdapter** proxy points `ANTHROPIC_BASE_URL` at slime's policy/SGLang, so
the agent's turns *are* policy rollouts and the whole trajectory (captured proxy-side, keyed by the
session-id bearer) becomes GRPO training data.

## Component wiring (B2 — two Runtimes)

```
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  TRAINING ENGINE — slime  (agentcore_p4 container, 8×L40S)                │
  │  Megatron actor (4 GPU: 0-3)  +  SGLang inference (4 GPU: 4-7)  + GRPO    │
  │  Sees ONLY the LLM request/response pairs (never the tool loop).          │
  └───────────────▲──────────────────────────────────────┬──────────────────┘
                  │ captured req/resp pairs + reward       │ SGLang completions
                  │   → GRPO samples                       │
  ┌───────────────┴──────────────────────────┐            │
  │  LLM API PROXY — slime AnthropicAdapter    │◀───────────┘
  │  http://$ADAPTER_PUBLIC_HOST:18021         │  (pinned port, one SG rule)
  │  Anthropic API → SGLang; records pairs     │
  └───────────────▲────────────────────────────┘
                  │ Anthropic API calls (ANTHROPIC_BASE_URL = adapter_url), dialed back over VPC
                  │
  ┌───────────────┴───────────────────────────────┐        ┌──────────────────────────────────┐
  │  HARNESS Runtime #1 — Claude Code (`claude -p`) │  MCP   │  SANDBOX Runtime #2 — GPU (g6)     │
  │  built-ins DISABLED; only mcp__sandbox__* tools │───────▶│  sandbox_server_agent.py           │
  │  CPU, networkMode=VPC                           │ bash/  │  write_file/exec/read_file         │
  │                                                 │ write/ │  KernelBench checker: torch on CUDA│
  │                                                 │ read   │  (torch on the prepared volume)    │
  └─────────────────────────────────────────────────┘        └──────────────────────────────────┘
       (both reached from the trainer via invoke_agent_runtime; the reward checker
        `python3 run_checks.py` is invoked on Runtime #2 after the harness returns)
```

Flow per prompt (`agentcore_runtime_generate.generate`): (1) seed Runtime #2's workspace with
`PROBLEM.md` + `run_checks.py`; (2) invoke Runtime #1 with `adapter_url` + the sandbox ARN — the CLI
loops, LLM via the proxy, tools via the MCP bridge into Runtime #2; (3) score by invoking
`run_checks.py` on Runtime #2; (4) `finish_session` drains the proxy-captured trajectory to
loss-masked GRPO samples.

## Deploy (once)

```bash
# 1. GPU sandbox Runtime #2 (+ CPU sandbox + mini harness) and prepare its torch volume:
ADAPTER_PUBLIC_HOST=PRIVATE-IP-f99864bc python3 code/deploy/deploy_two_runtime.py all
# 2. The CLAUDE harness Runtime #1 (node + claude CLI + mcp/uvicorn, HARNESS_KIND=claude):
ADAPTER_PUBLIC_HOST=PRIVATE-IP-f99864bc python3 code/deploy/deploy_two_runtime.py harness-claude
```

`all` prints the GPU sandbox ARN (`AGENTCORE_SANDBOX_GPU_RUNTIME_ARN`); `harness-claude` prints the
claude harness ARN (`AGENTCORE_HARNESS_CLAUDE_RUNTIME_ARN`). The GPU sandbox's one-time
`{"mode":"prepare"}` invoke (run by `all` unless `SKIP_PREPARE=1`) installs the CUDA `torch` stack
onto its volume — without it the checker can't import torch and every reward is 0.

## Build the dataset (already present)

```bash
docker exec agentcore_p4 python /work/code/demos/demo_b2/build_dataset.py \
    --dataset kernelbench --level level_1 --limit 100 --out /work/code/demos/demo_b2/data/kernelbench_train.jsonl
docker exec agentcore_p4 python /work/code/demos/demo_b2/build_dataset.py \
    --dataset kernelbench --level level_2 --limit 20 --out /work/code/demos/demo_b2/data/kernelbench_eval.jsonl
```

The real **KernelBench** (`ScalingIntelligence/KernelBench`, 270 CUDA problems). Train = level_1
(100 single ops); held-out eval = 20 level_2 fusions (disjoint). Each problem gives a reference
`Model`; the policy writes an equivalent **`ModelNew`** that runs on CUDA and matches it.

## Required environment (real run only — this demo does NOT run offline)

| var | what |
|---|---|
| `AGENTCORE_HARNESS_RUNTIME_ARN` | the CLAUDE harness Runtime #1 (`deploy_two_runtime.py harness-claude`) |
| `AGENTCORE_SANDBOX_RUNTIME_ARN` | the GPU sandbox Runtime #2 (`deploy_two_runtime.py all`) |
| `ADAPTER_PUBLIC_HOST` | this box's PRIVATE IP the harness Runtime #1 dials back to (e.g. `PRIVATE-IP-f99864bc`) |
| `ADAPTER_PORT` | pinned adapter port (default `18021`); one narrow SG ingress rule |
| `AWS_REGION` | AgentCore region (default `us-west-2`) |

## Run

```bash
docker exec agentcore_p4 bash -lc \
    'cd /work && AGENTCORE_HARNESS_RUNTIME_ARN=... AGENTCORE_SANDBOX_RUNTIME_ARN=... \
       ADAPTER_PUBLIC_HOST=PRIVATE-IP-f99864bc NR=11 EVAL_INTERVAL=10 \
       bash code/demos/demo_b2/run.sh'
```

`HARNESS_KIND` defaults to `claude` here. The SGLang tool-call parser (`qwen25`) is REQUIRED — the
CLI drives its tools off Anthropic `tool_use` blocks, which the adapter only produces from Qwen3's
Hermes `<tool_call>` output via that parser (left unset, reward is 0 for every rollout).

### Size / save-only-at-end note

The train pool is **100 prompts** (level_1). Keep `NR × RB ≤ 100` so one epoch spans the run and the
only checkpoint is the final one (slime force-saves at every epoch boundary,
`step % (dataset_size/RB) == 0`; with `RB=4` the epoch is 25 steps). For a 10-train + 1-eval smoke
run use **`NR=11 EVAL_INTERVAL=10`** (real steps = NR−1 = 10). Defaults are `RB=4 NS=8 NR=11`. The
checkpoint-save deadlock is fixed via `--async-save` +
`--custom-megatron-init-path …megatron_init.enable_persistent_async_ckpt` (see demos-guide.md §7).

## Metrics

```bash
python3 code/demos/demo_b2/metrics.py <run>.log \
    --csv <run>.csv --eval-csv <run>.eval.csv --plot <run>.png
```

Per-step train CSV (`rollout/raw_reward`, `train/grad_norm`, `train/kl_loss`, `perf/*`) + a per-eval
CSV (`eval/step` + held-out solve rate) + a multi-panel PNG.

## Legacy

This dir's `Dockerfile` (CUDA + torch + Node + `claude` baked into one GPU sandbox image) and
`claude_code_generate.py` are the **old co-located shortcut** (CLI inside the sandbox, in-process
`ClaudeCodeHarness`). The two-Runtime path above supersedes it for B2; the co-located image is kept
only for reference.
