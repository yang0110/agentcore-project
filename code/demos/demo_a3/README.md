# Demo A3 — MBPP (CPU) with a mini Claude-Code-like harness

**Demo A3 = Demo A + a harness we wrote ourselves that mimics Claude Code.** Same task (MBPP),
same CPU sandbox family, same slime async/decoupled training, same **Qwen3-4B policy under
training**. The difference from [Demo A](../demo_a/) is the **rollout harness**: instead of our
lightweight `run_episode` loop, the agent is
[`agentcore_slime_demo.mini_agent.MiniClaudeHarness`](../../agentcore_slime_demo/mini_agent.py) — a
~1-file pure-Python coding agent that reproduces Claude Code's essential loop:

> call the model → it picks a tool → run the tool in the sandbox → feed the result back → repeat.

That is exactly the "agent harness" AWS describes in the [AgentCore harness
docs](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/harness.html) (the
orchestration layer that calls the model, picks tools, and passes results back), backed here by
an **AgentCore Runtime microVM** as the sandbox. The agent has three tools — `write_file`, `bash`,
`read_file` — a short system prompt, and a flat message list. No CLI, no Node, no MCP, no
subagents, no compaction: that minimalism is the point.

> **Deployment: decoupled + disaggregated (two AgentCore Runtimes).** A3 now runs the mini harness
> the **target** way — the harness has ONE home, its **own AgentCore Runtime #1** (a
> `BedrockAgentCoreApp` wrapping `MiniClaudeHarness`, `harness_kind=mini`) whose tool calls invoke a
> **separate sandbox AgentCore Runtime #2** (a CPU Code-Interpreter-class microVM for A3). The
> harness Runtime dials the `AnthropicAdapter` proxy on this box back over the **private VPC**
> (security-group gated, **no public port**). Harness and sandbox are two distinct component types
> (Agent Lightning §1); AgentCore hosts both, as two Runtimes. Wired via
> `agentcore_runtime_generate.generate`; the earlier in-process shortcut has been removed. See
> [docs/harness-sandbox-decoupling.md](../../../docs/harness-sandbox-decoupling.md) §7.

**How it trains for free (Agent Lightning decoupling).** The agent is just an HTTP client of
slime's **`AnthropicAdapter`**. It speaks the Anthropic Messages API (`POST /v1/messages` with
`tools`); the adapter — in front of SGLang — serves every completion from the policy under
training and records each turn into a per-session `TrajectoryManager`. The session id rides in the
`Authorization: Bearer <session_id>` header. `finish_session` then drains that trajectory into
loss-masked GRPO samples. So `mini_agent.py` contains **zero** training code.

## Why the mini harness (vs the real Claude Code CLI)

Running the **real Claude Code CLI** as the harness is faithful, but it needs a custom multi-user
container with Node 22 + Claude Code installed (that heavier real-CLI variant has been retired).
**Demo A3 runs our ~1-file `MiniClaudeHarness` instead** (same two-Runtime deployment), so:

- **no Node, no CLI in the harness image** — the harness Runtime #1 image is a slim CPU Python
  container (aiohttp + boto3 + bedrock_agentcore, no slime), and the sandbox Runtime #2 is a
  CPU Code-Interpreter-class microVM. Much lighter than A2's full-CLI container.
- **the harness dials back** — Runtime #1 reaches the `AnthropicAdapter` proxy at
  `ADAPTER_PUBLIC_HOST:ADAPTER_PORT` (this box's **private IP**, a pinned port opened only from the
  runtime security group). `SANDBOX=agentcore-runtime` and the two Runtime ARNs are **required**.

Think of it as: A2 = maximum fidelity (real CLI), A3 = the simplest harness that is still a real
agentic loop — both deployed the same target way (harness Runtime + sandbox Runtime).

This directory is **independently runnable**: it owns its `data/` (500-prompt MBPP train pool +
held-out eval) and `ckpt/` output dir. The `agentcore_slime_demo` library is imported from the
**shared** copy at `code/agentcore_slime_demo/` (on `PYTHONPATH` as `/work/code`). It also
references two **shared** things: the container `agentcore_p4` and the base Megatron checkpoint at
`/work/code/p4_slime_train/ckpt/Qwen3-4B_torch_dist` (override with `CKPT_BASE`).

## Offline smoke (no slime, no AWS, no GPU)

Proves the harness loop — model call → tool exec in a `LocalSandbox` → result → terminate —
against a fake scripted adapter:

```bash
PYTHONPATH=. python3 ../../scripts/mini_agent_smoke.py
```

## Build the datasets

```bash
python3 build_dataset.py --dataset mbpp --split "train+test" --limit 500 --out data/mbpp_train.jsonl
python3 build_dataset.py --dataset mbpp --split validation --limit 8   --out data/mbpp_eval.jsonl
```

## Run

First deploy the two Runtimes (harness #1 + sandbox #2) and open the adapter port from the runtime
security group (see [docs/harness-sandbox-decoupling.md](../../../docs/harness-sandbox-decoupling.md)
§7), then export the ARNs + private IP:

```bash
docker exec agentcore_p4 bash -lc 'cd /work \
  && SANDBOX=agentcore-runtime AWS_REGION=us-west-2 \
     AGENTCORE_HARNESS_RUNTIME_ARN=arn:aws:bedrock-agentcore:...:runtime/harness-... \
     AGENTCORE_SANDBOX_RUNTIME_ARN=arn:aws:bedrock-agentcore:...:runtime/sandbox-... \
     ADAPTER_PUBLIC_HOST=PRIVATE-IP-f99864bc ADAPTER_PORT=18021 \
     bash code/demos/demo_a3/run.sh'
```

Knobs: `RB` (prompts/step), `NS` (GRPO group size), `NR` (num-rollout; real steps = NR−1), `TEMP`,
`CONC`, `EVAL_INTERVAL` (eval every N steps, `0` disables), `EVAL_NS`, `MINI_AGENT_MAX_TURNS`
(max model↔tool turns per episode). GPU split on this box is 4 train + 4 infer + 0 env (CPU
sandbox Runtime). Save-only-at-end holds while `dataset_size ≥ NR × RB` (500 ≥ 100 × 4). The
checkpoint-save deadlock is fixed via `megatron_init.py` (both `--async-save` and
`--custom-megatron-init-path` are baked into `run.sh`).

Both Runtimes need `AWS_BEARER_TOKEN_BEDROCK` + `AWS_REGION`; `ADAPTER_PUBLIC_HOST` must be this
box's **private IP** and `ADAPTER_PORT` must be open to the runtime security group (no public port).

## Metrics

```bash
python3 metrics.py train_a3.log --csv train_a3.csv --eval-csv train_a3.eval.csv --plot train_a3.png
```
