# Demo A2 — MBPP (CPU) with the real Claude Code harness

**Demo A2 = Demo A + Claude Code.** Same task (MBPP), **same sandbox type** (AgentCore Code
Interpreter CPU microVM), same slime async/decoupled training, same **Qwen3-4B policy under
training**. The one difference from [Demo A](../demo_a/) is the **harness** — the agent loop:
instead of our lightweight `run_episode` loop, the agent is **Claude Code** (`claude -p`), wired
via `agentcore_slime_demo.claude_code_generate.generate`.

> **Harness ≠ sandbox — two distinct component types.** In (harnessed) agentic RL the *harness*
> and the *sandbox* are separate components (Agent Lightning v1.0, §1): the **harness** owns the
> agent loop (tool selection, context construction, control flow) and calls the LLM; the
> **sandbox** is the execution environment where the harness's tool actions (`Bash`/edit/run)
> actually run and produce observations. **A2 changes only the harness** (our loop → Claude
> Code); the **sandbox is unchanged** — still the AgentCore Code Interpreter, exactly as Demo A.
> The training engine (slime) sees *neither* directly — only the LLM request/response pairs the
> harness makes, captured through the adapter.

> **Target: decoupled + disaggregated (two AgentCore Runtimes).** Today A2 **co-locates** the
> Claude Code CLI *inside* the sandbox (see below) — a bring-up shortcut, not the intended
> deployment. The target for A2/A3/B2/B3 alike is Claude Code on its **own AgentCore Runtime**,
> built-in tools disabled and redirected via an MCP bridge to a **separate sandbox AgentCore
> Runtime** (Code Interpreter for A2, a real GPU Runtime instance for B2). See
> [docs/harness-sandbox-decoupling.md](../../../docs/harness-sandbox-decoupling.md) §3, §7.

Claude Code's "Anthropic" API traffic is pointed at slime's **`AnthropicAdapter`**
(`ANTHROPIC_BASE_URL` → adapter → SGLang), so every completion Claude Code makes is actually
served by the policy being trained, and its tokens are captured for GRPO. This is Agent
Lightning's "train with real harnesses, zero changes" — the *only* edit to slime's shipped
`coding_agent_rl` generate is swapping `E2BSandbox` → `AgentCoreSandbox`.

## Four-component wiring (A2)

Agent Lightning's disaggregated architecture has four components. A2 maps to them as:

```
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  TRAINING ENGINE — slime  (agentcore_p4 container, 8×L40S)                │
  │  Megatron actor (4 GPU)  +  SGLang inference (4 GPU)  +  GRPO + wt sync   │
  │  Sees ONLY the LLM request/response pairs (never the tool loop).          │
  └───────────────▲──────────────────────────────────────┬──────────────────┘
                  │ (2) captured req/resp pairs           │ SGLang completions
                  │     + reward  → GRPO samples          │
  ┌───────────────┴──────────────────────────┐           │
  │  LLM API PROXY — slime AnthropicAdapter    │◀──────────┘
  │  http://$ADAPTER_PUBLIC_HOST:<port>        │
  │  Anthropic API  →  SGLang;  records pairs  │
  └───────────────▲────────────────────────────┘
                  │ (1) Anthropic API calls  (ANTHROPIC_BASE_URL = adapter_url)
                  │
  ┌───────────────┴───────────────────────────────────────────────────────────┐
  │  SANDBOX — AgentCore Code Interpreter (CPU microVM)   [same type as Demo A] │
  │  ┌─────────────────────────────────────────────────┐                       │
  │  │  HARNESS — Claude Code  (`claude -p`)            │  (3) Bash/edit/run    │
  │  │  owns the agent loop; co-located in the sandbox  │──────────────────────▶│
  │  └─────────────────────────────────────────────────┘   PROBLEM.md,         │
  │  (4) reward: `python3 run_checks.py` (exit code)        solution.py, …      │
  └───────────────────────────────────────────────────────────────────────────┘
```

Flow per rollout: (1) Claude Code (harness) makes Anthropic API calls → adapter → SGLang policy;
(2) the adapter records each request/response pair, pairs them with the (4) reward from the
sandbox's checker, and hands GRPO samples to slime; (3) the harness's tool calls execute in the
sandbox. **The harness (Claude Code) and the sandbox (Code Interpreter) are two different
components** — the box-in-box only reflects the current co-location (see below), not a merge of
the two roles.

This directory is **independently runnable**: it owns its `data/` (500-prompt MBPP train pool +
held-out eval) and `ckpt/` output dir. The `agentcore_slime_demo` library is imported from the
**shared** copy at `code/agentcore_slime_demo/` (on `PYTHONPATH` as `/work/code`). It also
references two **shared** things: the container `agentcore_p4` and the base Megatron checkpoint at
`/work/code/p4_slime_train/ckpt/Qwen3-4B_torch_dist` (override with `CKPT_BASE`).

## Build the datasets

```bash
python3 build_dataset.py --dataset mbpp --split "train+test" --limit 500 --out data/mbpp_train.jsonl
python3 build_dataset.py --dataset mbpp --split validation --limit 8   --out data/mbpp_eval.jsonl
```

## Build the sandbox container (harness co-located — implementation detail)

The current implementation **co-locates the harness inside the sandbox**: `claude_code_generate.py`
calls `install_cli(sb)` and then launches `claude -p` *in the same sandbox session* where its tool
calls run (slime's `ClaudeCodeHarness.launch_and_wait` → `run_agent(sb, …)`). This is an
**implementation detail, not a change of sandbox type** — conceptually the harness and the
code-interpreter sandbox stay separate components (see the wiring diagram above); they just happen
to share one container here.

Because a full multi-user CLI *process* can't run in the **managed** serverless Code Interpreter,
this co-location requires a **custom AgentCore Runtime container** (multi-user: `useradd`/`setsid`/
`npm`) to host the same Code-Interpreter-style sandbox. Node 22 + Claude Code are injected at build
time from host tarballs:

```bash
docker build -f Dockerfile \
  --build-arg NODE_TARBALL=node-v22.x-linux-x64.tar.xz \
  --build-arg CC_TARBALL=claude-code-x.y.z.tgz \
  -t agentcore-demo-a2 .
```

## Required environment (this path does NOT run offline)

| Env var | What it is |
|---|---|
| `ADAPTER_PUBLIC_HOST` | address the sandbox dials back to reach slime's adapter (**required**) |
| `SLIME_AGENT_NODE_TARBALL` | Node 22 tarball for `install_cli` |
| `SLIME_AGENT_CC_TARBALL` | Claude Code tarball for `install_cli` |
| `ANTHROPIC_MODEL` | model name Claude Code requests (served by the policy) |
| `AWS_BEARER_TOKEN_BEDROCK`, `AWS_REGION` | AgentCore microVM credentials |

## Run

```bash
docker exec agentcore_p4 bash -lc \
  'cd /work && ADAPTER_PUBLIC_HOST=... SLIME_AGENT_NODE_TARBALL=... SLIME_AGENT_CC_TARBALL=... \
   ANTHROPIC_MODEL=... bash code/demos/demo_a2/run.sh'
```

Knobs: `RB` (prompts/step), `NS` (GRPO group size), `NR` (num-rollout; real steps = NR−1),
`TEMP`, `CONC`, `EVAL_INTERVAL` (eval every N steps, `0` disables), `EVAL_NS`. GPU split is
4 train + 4 infer + 0 env (CPU envs). Save-only-at-end holds while `dataset_size ≥ NR × RB`
(500 ≥ 100 × 4). The checkpoint-save deadlock is fixed via `megatron_init.py` (both `--async-save`
and `--custom-megatron-init-path` are baked into `run.sh`).

## Metrics

Tee the ray-job output to a log, then parse to CSVs + a PNG (same as Demo A):

```bash
python3 metrics.py train_a2.log --csv train_a2.csv --eval-csv train_a2.eval.csv --plot train_a2.png
```

## Pending pieces (honest status)

Demo A2 is the **less battle-tested** path relative to Demo A. To run it end-to-end you still need:

- the **custom container image** built above (multi-user; the serverless Code Interpreter won't do);
- the **Node 22 + Claude Code tarballs** staged on the host and pointed at by the env vars;
- a **publicly reachable `ADAPTER_PUBLIC_HOST`** so the sandbox can call slime's adapter back.

It does **not** run offline. If you just want a working MBPP RL loop today, use
[Demo A](../demo_a/); Demo A2 is the "real harness" variant on top of it.
