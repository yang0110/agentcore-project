# Demo B3 — KernelBench (GPU) with a mini Claude-Code-like harness

**Demo B3 = Demo B + a harness we wrote ourselves that mimics Claude Code.** Same task (real
[KernelBench](https://huggingface.co/datasets/ScalingIntelligence/KernelBench): write a `ModelNew`
that matches a reference `Model` numerically **and** runs on CUDA), same GPU-in-sandbox story as
[Demo B](../demo_b/) (the checker runs torch on a GPU **inside the sandbox** — here that GPU is on
the **remote** sandbox Runtime #2 instance, not this box), same slime async/decoupled training, same
**Qwen3-4B policy**. The difference is the **rollout harness**: instead of our lightweight
`run_episode` loop, the agent is
[`agentcore_slime_demo.mini_agent.MiniClaudeHarness`](../../agentcore_slime_demo/mini_agent.py) — a
~1-file pure-Python coding agent (tools: `write_file`, `bash`, `read_file`) that reproduces Claude
Code's model → tool → sandbox → result loop, matching AWS's [AgentCore "agent harness"
description](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/harness.html).

> **Deployment: decoupled + disaggregated (two AgentCore Runtimes).** B3 now runs the mini harness
> the **target** way — one home, its **own AgentCore Runtime #1** (a `BedrockAgentCoreApp` wrapping
> `MiniClaudeHarness`, `harness_kind=mini`) whose tool calls **and** the KernelBench reward checker
> invoke a **separate sandbox AgentCore Runtime #2** that is a **real GPU Runtime instance**
> (a g6 capacity-provider EC2 GPU) — so the checker's CUDA runs *there*, not on this box. There is
> **no local `GpuPool`** any more. The harness Runtime dials the `AnthropicAdapter` proxy on this
> box back over the **private VPC** (security-group gated, **no public port**). Wired via
> `agentcore_runtime_generate.generate`; the in-process shortcut has been removed. See
> [docs/harness-sandbox-decoupling.md](../../../docs/harness-sandbox-decoupling.md) §6–§7.

**How it trains (Agent Lightning decoupling).** The agent is an HTTP client of slime's
**`AnthropicAdapter`**; it speaks the Anthropic Messages API, the adapter serves every completion
from the policy under training (SGLang) and records each turn into a per-session
`TrajectoryManager` keyed by the `Authorization: Bearer <session_id>` header. `finish_session`
drains it into loss-masked GRPO samples. The mini harness runs on its own AgentCore Runtime #1 and
its tool calls hit a separate GPU sandbox Runtime #2 — see
[`agentcore_runtime_generate.py`](../../agentcore_slime_demo/agentcore_runtime_generate.py).

## Demo B3 vs Demo B2 (why this exists)

[Demo B2](../demo_b2/) runs the **real Claude Code CLI** inside the sandbox, which needs a custom
container with **CUDA + torch AND Node 22 + Claude Code**. **Demo B3 runs our ~1-file
`MiniClaudeHarness` instead** (same two-Runtime deployment), so the **harness** Runtime #1 image is
a slim CPU Python container (no Node/CLI, no slime), and only the **sandbox** Runtime #2 image needs
**CUDA + torch** for the KernelBench checker. B3 is the lightest harness that still couples a real
agentic loop with the GPU-in-sandbox task.

The KernelBench checker runs **on the sandbox Runtime #2 GPU instance** (invoked by
`agentcore_runtime_generate` for both tool calls and scoring). There is no local `GpuPool` here any
more — GPU admission control, if needed, lives on the sandbox Runtime, not on this trainer box.

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

First deploy the two Runtimes — harness #1 (CPU) + sandbox #2 (**GPU** g6 capacity provider) — and
open the adapter port from the runtime security group (see
[docs/harness-sandbox-decoupling.md](../../../docs/harness-sandbox-decoupling.md) §7), then:

```bash
docker exec agentcore_p4 bash -lc 'cd /work \
  && SANDBOX=agentcore-runtime AWS_REGION=us-west-2 \
     AGENTCORE_HARNESS_RUNTIME_ARN=arn:aws:bedrock-agentcore:...:runtime/harness-... \
     AGENTCORE_SANDBOX_RUNTIME_ARN=arn:aws:bedrock-agentcore:...:runtime/sandbox-gpu-... \
     ADAPTER_PUBLIC_HOST=PRIVATE-IP-f99864bc ADAPTER_PORT=18021 \
     bash code/demos/demo_b3/run.sh'
```

Knobs: `RB`, `NS`, `NR` (num-rollout; real steps = NR−1 — keep **`NR × RB ≤ 100`** so
save-only-at-end holds against the 100-problem level_1 pool), `TEMP`, `CONC`, `EVAL_INTERVAL`
(default 5), `EVAL_NS`, `MINI_AGENT_MAX_TURNS`. GPU split **on this box** is 4 train + 4 infer +
0 env — the KernelBench checker's GPU is on the remote sandbox Runtime #2.

Both Runtimes need `AWS_BEARER_TOKEN_BEDROCK` + `AWS_REGION`; `ADAPTER_PUBLIC_HOST` must be this
box's **private IP** and `ADAPTER_PORT` must be open to the runtime security group (no public port).

## Metrics

```bash
python3 metrics.py train_b3.log --csv train_b3.csv --eval-csv train_b3.eval.csv --plot train_b3.png
```
