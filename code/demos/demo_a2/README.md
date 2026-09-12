# Demo A2 — MBPP (CPU sandbox) with the real Claude Code CLI, the two-Runtime target

**Demo A2 = Demo A + the real Claude Code CLI, deployed the two-Runtime target way.** Same task
(MBPP), same slime async/decoupled training, same **Qwen3-4B policy under training** as
[Demo A](../demo_a/). The one difference is the **harness** (the agent loop): instead of our
lightweight `run_episode`, the agent is the real **Claude Code** (`claude -p`) — and it runs on its
**own AgentCore Runtime**, driving a **separate** CPU sandbox Runtime. A2 is exactly
[Demo B2](../demo_b2/) but with the **MBPP** task and a **CPU** sandbox Runtime #2 instead of
KernelBench on a GPU g6.

> **Harness ≠ sandbox — two distinct AgentCore Runtimes.** The **harness** owns the agent loop
> (tool selection, context, control flow) and calls the LLM; the **sandbox** is the execution
> environment where its tool actions run. A2 runs them on **two separate AgentCore Runtimes**:
> - **Harness Runtime #1** — the real `claude -p` (`Dockerfile.harness.claude`, `HARNESS_KIND=claude`).
>   Its built-in Bash/Edit/Write/Read/Web tools are **disabled**; its only tools are the
>   `mcp__sandbox__*` bridge (`sandbox_mcp.py` → FastMCP over localhost), which forwards every
>   file/shell action to Runtime #2. CPU-only, `networkMode=VPC` so it can dial the adapter back.
>   **This is the exact same harness Runtime B2 uses** (`deploy_two_runtime.py harness-claude`).
> - **Sandbox Runtime #2** — the CPU-backed `sandbox_server_agent.py` managed microVM
>   (`acs_sandbox_cpu`, from `deploy_two_runtime.py all`), where the harness's `write_file` /
>   `exec` / `read_file` actions and the MBPP `run_checks.py` reward run. No GPU needed.
>
> This replaces the earlier bring-up shortcut (the CLI co-located *inside* the sandbox via
> `claude_code_generate.py`, one Runtime). A2 now drives the **same hook as A3/B2/B3**
> (`agentcore_slime_demo.agentcore_runtime_generate.generate`); `HARNESS_KIND=claude` is the only
> switch that selects the real CLI over the mini agent. See
> [docs/harness-sandbox-decoupling.md](../../../docs/harness-sandbox-decoupling.md) §3, §6, §7.

The **Qwen3-4B policy is still trained by slime**. Claude Code never talks to a hosted Anthropic
model: slime's **AnthropicAdapter** proxy points `ANTHROPIC_BASE_URL` at slime's policy/SGLang, so
the agent's turns *are* policy rollouts and the whole trajectory (captured proxy-side, keyed by the
session-id bearer) becomes GRPO training data. This is Agent Lightning's "train with real harnesses,
zero changes."

## Component wiring (A2 — two Runtimes)

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
  │  HARNESS Runtime #1 — Claude Code (`claude -p`) │  MCP   │  SANDBOX Runtime #2 — CPU microVM │
  │  built-ins DISABLED; only mcp__sandbox__* tools │───────▶│  sandbox_server_agent.py           │
  │  CPU, networkMode=VPC                           │ bash/  │  write_file/exec/read_file         │
  │                                                 │ write/ │  MBPP checker: python3 run_checks.py│
  └─────────────────────────────────────────────────┘ read  └──────────────────────────────────┘
       (both reached from the trainer via invoke_agent_runtime; the reward checker
        `python3 run_checks.py` is invoked on Runtime #2 after the harness returns)
```

Flow per prompt (`agentcore_runtime_generate.generate`): (1) seed Runtime #2's workspace with
`PROBLEM.md` + `run_checks.py`; (2) invoke Runtime #1 with `adapter_url` + the sandbox ARN — the CLI
loops, LLM via the proxy, tools via the MCP bridge into Runtime #2; (3) score by invoking
`run_checks.py` on Runtime #2; (4) `finish_session` drains the proxy-captured trajectory to
loss-masked GRPO samples.

This directory is **independently runnable**: it owns its `data/` (500-prompt MBPP train pool +
held-out eval) and `ckpt/` output dir. The `agentcore_slime_demo` library is imported from the
**shared** copy at `code/agentcore_slime_demo/` (on `PYTHONPATH` as `/work/code`). It also
references two **shared** things: the container `agentcore_p4` and the base Megatron checkpoint at
`/work/code/p4_slime_train/ckpt/Qwen3-4B_torch_dist` (override with `CKPT_BASE`).

## Deploy (once)

Both runtimes come from the shared `deploy_two_runtime.py`; A2 reuses **exactly what A3/B2 already
deploy** — no A2-specific image:

```bash
# 1. The CPU sandbox Runtime #2 (+ GPU sandbox + mini harness). Prints AGENTCORE_SANDBOX_CPU_RUNTIME_ARN:
ADAPTER_PUBLIC_HOST=PRIVATE-IP-f99864bc python3 code/deploy/deploy_two_runtime.py all
# 2. The CLAUDE harness Runtime #1 (node + claude CLI + mcp/uvicorn, HARNESS_KIND=claude):
ADAPTER_PUBLIC_HOST=PRIVATE-IP-f99864bc python3 code/deploy/deploy_two_runtime.py harness-claude
```

`all` prints the CPU sandbox ARN (`AGENTCORE_SANDBOX_CPU_RUNTIME_ARN`); `harness-claude` prints the
claude harness ARN (`AGENTCORE_HARNESS_CLAUDE_RUNTIME_ARN`). The harness's steering system prompt is
task-agnostic (it just points the model at the three `mcp__sandbox__*` tools), so the **same** claude
harness Runtime serves both A2 (MBPP) and B2 (KernelBench).

## Build the datasets

```bash
python3 build_dataset.py --dataset mbpp --split "train+test" --limit 500 --out data/mbpp_train.jsonl
python3 build_dataset.py --dataset mbpp --split validation --limit 8   --out data/mbpp_eval.jsonl
```

## Required environment (real run only — this demo does NOT run offline)

| var | what |
|---|---|
| `AGENTCORE_HARNESS_RUNTIME_ARN` | the CLAUDE harness Runtime #1 (`deploy_two_runtime.py harness-claude`) |
| `AGENTCORE_SANDBOX_RUNTIME_ARN` | the CPU sandbox Runtime #2 (`deploy_two_runtime.py all` → `acs_sandbox_cpu`) |
| `ADAPTER_PUBLIC_HOST` | this box's PRIVATE IP the harness Runtime #1 dials back to (e.g. `PRIVATE-IP-f99864bc`) |
| `ADAPTER_PORT` | pinned adapter port (default `18021`); one narrow SG ingress rule |
| `AWS_REGION` | AgentCore region (default `us-west-2`) |

## Run

```bash
docker exec agentcore_p4 bash -lc \
    'cd /work && AGENTCORE_HARNESS_RUNTIME_ARN=... AGENTCORE_SANDBOX_RUNTIME_ARN=... \
       ADAPTER_PUBLIC_HOST=PRIVATE-IP-f99864bc NR=11 EVAL_INTERVAL=10 \
       bash code/demos/demo_a2/run.sh'
```

`HARNESS_KIND` defaults to `claude` here. `NR=11 EVAL_INTERVAL=10` is a 10-train + 1-eval smoke
(real steps = NR−1 = 10). Defaults are `RB=4 NS=8 NR=11`. The SGLang tool-call parser (`qwen25`) is
REQUIRED — the CLI drives its tools off Anthropic `tool_use` blocks, which the adapter only produces
from Qwen3's Hermes `<tool_call>` output via that parser (left unset, reward is 0 for every rollout).

### Size / save-only-at-end note

The train pool is **500 prompts**. Keep `NR × RB ≤ dataset_size` so one epoch spans the run and the
only checkpoint is the final one (slime force-saves at every epoch boundary, `step % (dataset_size/RB)
== 0`; with `RB=4` the epoch is 125 steps). The checkpoint-save is **synchronous** here (do NOT enable
`--async-save`): A2 saves once at the end, and both async backends fail under A2's required PP=2 config
(fork deadlock / `pidfd_getfd` EPERM) — see the header comment in `run.sh` and demos-guide.md §7.

## Metrics

```bash
python3 metrics.py train.twortt.log --csv train.csv --eval-csv train.eval.csv --plot train.png
```

Per-step train CSV (`rollout/raw_reward`, `train/grad_norm`, `train/kl_loss`, `perf/*`) + a per-eval
CSV (`eval/step` + held-out solve rate) + a multi-panel PNG.

## Legacy

This dir's `Dockerfile` (Node 22 + `claude` baked into a multi-user Code-Interpreter-style sandbox
image) and `agentcore_slime_demo/claude_code_generate.py` are the **old co-located shortcut** (CLI
inside the sandbox, one Runtime, `install_cli`). The two-Runtime path above supersedes it; the
co-located image is kept only for reference.
