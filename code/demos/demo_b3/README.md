# Demo B3 — KernelBench (GPU) with a mini Claude-Code-like harness

**Demo B3 = Demo B + a harness we wrote ourselves that mimics Claude Code.** Same task (real
[KernelBench](https://huggingface.co/datasets/ScalingIntelligence/KernelBench): write a `ModelNew`
that matches a reference `Model` numerically **and** runs on CUDA), same GPU-in-sandbox story as
[Demo B](../demo_b/) (the checker runs torch on a GPU inside the sandbox; a `GpuPool` dense-packs
the bursty env checkers onto physical GPUs 6,7), same slime async/decoupled training, same
**Qwen3-4B policy**. The difference is the **rollout harness**: instead of our lightweight
`run_episode` loop, the agent is
[`agentcore_slime_demo.mini_agent.MiniClaudeHarness`](../../agentcore_slime_demo/mini_agent.py) — a
~1-file pure-Python coding agent (tools: `write_file`, `bash`, `read_file`) that reproduces Claude
Code's model → tool → sandbox → result loop, matching AWS's [AgentCore "agent harness"
description](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/harness.html).

> **Target: decoupled + disaggregated (two AgentCore Runtimes).** Today B3 runs the mini harness
> **in the trainer/rollout process** and the GPU sandbox on **this box's L40S** (GpuPool on 6,7) —
> both bring-up shortcuts. The target for A2/A3/B2/B3 alike is the harness on its **own AgentCore
> Runtime** (a `BedrockAgentCoreApp` wrapping `MiniClaudeHarness`) whose tool calls invoke a
> **separate sandbox AgentCore Runtime** — for B3 a **real AgentCore GPU Runtime instance**
> (capacity-provider EC2 GPU), not local CUDA. See
> [docs/harness-sandbox-decoupling.md](../../../docs/harness-sandbox-decoupling.md) §6–§7.

**How it trains (Agent Lightning decoupling).** The agent is an HTTP client of slime's
**`AnthropicAdapter`**; it speaks the Anthropic Messages API, the adapter serves every completion
from the policy under training (SGLang) and records each turn into a per-session
`TrajectoryManager` keyed by the `Authorization: Bearer <session_id>` header. `finish_session`
drains it into loss-masked GRPO samples. See
[`mini_agent_generate.py`](../../agentcore_slime_demo/mini_agent_generate.py).

## Demo B3 vs Demo B2 (why this exists)

[Demo B2](../demo_b2/) runs the **real Claude Code CLI** inside the sandbox, which needs a custom
container with **CUDA + torch AND Node 22 + Claude Code**. **Demo B3 runs the agent in the trainer
process instead**, so the sandbox needs only **CUDA + torch** for the KernelBench checker — which
the standard GPU Runtime image already has. Hence **no Node/CLI, no custom multi-user container,
no Dockerfile** here, and no dial-back host (the agent reaches the adapter over `127.0.0.1`). B3 is
the simplest path that still couples a real agentic loop with the GPU-in-sandbox task.

The KernelBench checker exec is gated onto the env GPUs by the process-wide `GpuPool`
(`AGENTCORE_GPU_GATE=all`, `GPU_POOL_DEVICES=6,7`) exactly as in Demo B — that wiring is unchanged,
because it lives in the sandbox/score layer, not the harness.

This directory is **independently runnable**: it owns its `data/` (level_1 100-problem train pool
+ 20 held-out level_2 fusions for eval) and `ckpt/`. The `agentcore_slime_demo` library is imported
from the **shared** copy at `code/agentcore_slime_demo/` (on `PYTHONPATH` as `/work/code`). It also
references the shared container `agentcore_p4` and base checkpoint `Qwen3-4B_torch_dist` (override `CKPT_BASE`).

## Offline smoke (harness logic only; no GPU/slime)

```bash
PYTHONPATH=. python3 ../../scripts/mini_agent_smoke.py
```

## Build the datasets

```bash
python3 build_dataset.py --dataset kernelbench --level level_1 --limit 100 --out data/kernelbench_train.jsonl
python3 build_dataset.py --dataset kernelbench --level level_2 --limit 20  --out data/kernelbench_eval.jsonl
```

## Run

```bash
docker exec agentcore_p4 bash -lc 'cd /work && SANDBOX=agentcore \
  GPU_POOL_CONCURRENCY=16 bash code/demos/demo_b3/run.sh'
```

Knobs: `RB`, `NS`, `NR` (num-rollout; real steps = NR−1 — keep **`NR × RB ≤ 100`** so
save-only-at-end holds against the 100-problem level_1 pool), `TEMP`, `CONC`, `EVAL_INTERVAL`
(default 5), `EVAL_NS`, `MINI_AGENT_MAX_TURNS`. GPU-env knobs: `GPU_POOL_CONCURRENCY` (envs on the
GPU at once), `GPU_POOL_PER_ENV_MIB`, `GPU_POOL_DEVICES`. GPU split is 4 train + 2 infer + 2 env.

`SANDBOX=agentcore` needs `AWS_BEARER_TOKEN_BEDROCK` + `AWS_REGION`; `SANDBOX=local` runs the
checker on this box's GPUs directly.

## Metrics

```bash
python3 metrics.py train_b3.log --csv train_b3.csv --eval-csv train_b3.eval.csv --plot train_b3.png
```
