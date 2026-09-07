# Demo B2 — KernelBench (GPU) with the real Claude Code agent harness

**Demo B2 = Demo B + Claude Code.** Same task and **same sandbox type** as Demo B — KernelBench
CUDA ops, the checker runs `torch` on a GPU *inside* the sandbox, and a `GpuPool` dense-packs the
bursty env checkers onto physical GPUs **6,7** — but the **harness** (the agent loop) is **Claude
Code** (`claude -p`), not our lightweight `run_episode`.

> **Harness ≠ sandbox — two distinct component types.** As in [A2](../demo_a2/), the *harness* and
> the *sandbox* are separate components (Agent Lightning v1.0, §1): the **harness** owns the agent
> loop (tool selection, context, control flow) and calls the LLM; the **sandbox** is the execution
> environment where the harness's tool actions run. **B2 changes only the harness** (our loop →
> Claude Code); the **GPU sandbox is unchanged** from Demo B. B2 therefore combines the two
> least-mature axes — the GPU sandbox *and* the Claude Code harness.

> **Target: decoupled + disaggregated (two AgentCore Runtimes).** Today B2 **co-locates** the CLI
> inside the sandbox and runs that sandbox on **this box's L40S** — both bring-up shortcuts. The
> target for A2/A3/B2/B3 alike is Claude Code on its **own AgentCore Runtime**, built-in tools
> redirected via an MCP bridge to a **separate sandbox AgentCore Runtime** — for B2 a **real
> AgentCore GPU Runtime instance** (capacity-provider EC2 GPU), not local CUDA. See
> [docs/harness-sandbox-decoupling.md](../../../docs/harness-sandbox-decoupling.md) §3, §6, §7.

The **Qwen3-4B policy is still trained by slime**. Claude Code never talks to a hosted Anthropic
model: an OpenAI→Anthropic **adapter** points `ANTHROPIC_BASE_URL` at slime's policy/SGLang, so the
agent's turns *are* policy rollouts and the whole trajectory becomes GRPO training data. The only
swap from slime's stock Claude Code harness is `E2BSandbox → AgentCoreSandbox`
(`agentcore_slime_demo/claude_code_generate.py`).

## Four-component wiring (B2)

Same four Agent Lightning components as A2, but the **sandbox is GPU** and its GPU-touching exec
(the KernelBench checker) is admission-controlled by `GpuPool` onto physical GPUs 6,7:

```
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  TRAINING ENGINE — slime  (agentcore_p4 container, 8×L40S)                │
  │  Megatron actor (4 GPU: 0-3)  +  SGLang inference (2 GPU: 4-5)  + GRPO    │
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
  │  SANDBOX — AgentCore GPU Runtime container            [same type as Demo B] │
  │  ┌─────────────────────────────────────────────────┐                       │
  │  │  HARNESS — Claude Code  (`claude -p`)            │  (3) Bash/edit/run    │
  │  │  owns the agent loop; co-located in the sandbox  │──────────────────────▶│
  │  └─────────────────────────────────────────────────┘   ModelNew.py, …      │
  │  (4) reward: `python3 run_checks.py` — torch on CUDA ─────┐                 │
  └──────────────────────────────────────────────────────────┼─────────────────┘
                                                              │ GPU exec gated by
                                                    ┌─────────▼──────────┐
                                                    │  GpuPool → GPUs 6,7 │  (2 env GPUs)
                                                    └────────────────────┘
```

Flow is identical to A2 — (1) harness→adapter→policy, (2) adapter pairs req/resp with the (4)
reward → GRPO samples, (3) tool calls run in the sandbox — **plus** the KernelBench checker's
CUDA work is routed through `GpuPool` (`AGENTCORE_GPU_GATE=all`) onto the 2 reserved env GPUs.
As in A2, the box-in-box is only the current **co-location** of the harness process inside the
sandbox, not a merge of the harness and sandbox roles.

> **Local stand-in vs. target.** The diagram's `GpuPool → GPUs 6,7` is **today's stand-in**: the
> GPU sandbox runs on *this trainer box's* L40S, so slime is squeezed to 4 train + 2 infer to free
> 2 local env GPUs. **The target for B2 (and B3) is a real AgentCore GPU Runtime instance** — a
> capacity-provider EC2 GPU (as use cases 08/09 ran on a `g6.xlarge`/L4), so the checker's CUDA
> runs on a *remote* managed instance. Moving it off-box frees GPUs 6,7 and lets slime go back to
> 4 train + 4 infer. That managed-GPU-Runtime backend is unbuilt (`AgentCoreSandbox` only speaks
> the Code Interpreter client today) — see `docs/harness-sandbox-decoupling.md` §6 and demos-guide
> §8 "Not yet" (b).

This directory is **independently runnable**: it owns its `data/` and `ckpt/`. The
`agentcore_slime_demo` library is imported from the **shared** copy at `code/agentcore_slime_demo/`
(on `PYTHONPATH` as `/work/code`). It also references the shared base Megatron checkpoint at
`/work/code/p4_slime_train/ckpt/Qwen3-4B_torch_dist` (override with `CKPT_BASE`) and `/root/slime`
inside the `agentcore_p4` container.

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
(`--dataset kernelbench-stub` still builds the old 3-op smoke set.)

## Build the sandbox container (harness co-located — implementation detail)

As in [A2](../demo_a2/), the current implementation **co-locates the harness inside the sandbox**:
`claude_code_generate.py` installs and launches `claude -p` in the *same* sandbox session where its
tool calls and the checker run. This is an **implementation detail, not a change of sandbox type** —
the harness and the GPU sandbox stay conceptually separate (see the wiring diagram above).

Because that co-located CLI needs a full multi-user Linux + CUDA, the sandbox is a **custom
AgentCore Runtime image** (this dir's `Dockerfile`): CUDA 12.4 + Node 22 + the `claude` CLI + a
CUDA build of torch (the checker needs it), non-root multi-user.

```bash
docker build -f code/demos/demo_b2/Dockerfile \
    --build-arg SLIME_AGENT_CC_TARBALL=./claude-code-<version>.tgz \
    -t agentcore-demo-b2 code/demos/demo_b2
```

## Required environment (real run only — this demo does NOT run offline)

| var | what |
|---|---|
| `ADAPTER_PUBLIC_HOST` | address the sandbox dials back to reach the OpenAI→Anthropic adapter |
| `SLIME_AGENT_NODE_TARBALL` / `SLIME_AGENT_CC_TARBALL` | Node 22 + Claude Code tarballs the sandbox installs |
| `ANTHROPIC_MODEL` | model name Claude Code requests (mapped to the slime policy by the adapter) |
| `AWS_BEARER_TOKEN_BEDROCK`, `AWS_REGION` | AgentCore auth for the real sandbox (`SANDBOX=agentcore`) |
| `GPU_POOL_CONCURRENCY`, `GPU_POOL_PER_ENV_MIB`, `GPU_POOL_DEVICES` | GpuPool knobs for env checkers on GPUs 6,7 |

## Run

```bash
docker exec agentcore_p4 bash -lc \
    'cd /work && ADAPTER_PUBLIC_HOST=... SLIME_AGENT_CC_TARBALL=... ANTHROPIC_MODEL=... \
       bash code/demos/demo_b2/run.sh'
```

### Size / save-only-at-end note

The train pool is now **100 prompts** (level_1). Keep `NR × RB ≤ 100` so one epoch spans the run
and the only checkpoint is the final one (slime force-saves at every epoch boundary,
`step % (dataset_size/RB) == 0`; with `RB=4` the epoch is 25 steps and the default `NR=20` = 19
real steps never crosses it). Defaults: **`RB=4 NS=8 NR=20`**, `EVAL_INTERVAL=5`.
Other knobs: `NS`, `TEMP`, `CONC`, `SAVE` (`SAVE_INTERVAL` is huge → save-only-at-end; the final
rollout force-saves regardless). The checkpoint-save deadlock is fixed via `--async-save` +
`--custom-megatron-init-path …megatron_init.enable_persistent_async_ckpt` (see demos-guide.md §7).

## Metrics

```bash
python3 code/demos/demo_b2/metrics.py <run>.log \
    --csv <run>.csv --eval-csv <run>.eval.csv --plot <run>.png
```

Per-step train CSV (`rollout/raw_reward`, `train/grad_norm`, `train/kl_loss`, `perf/*`) + a per-eval
CSV (`eval/step` + held-out solve rate) + a multi-panel PNG.

## Pending pieces (be honest)

B2 combines the **two least-mature pieces** of the project — the **GPU sandbox** *and* the **Claude
Code harness** — so it has the most prerequisites and **does not run offline**:

- a **custom GPU sandbox container** (this dir's `Dockerfile`: CUDA + torch + Node + `claude`),
- the **Node 22 + Claude Code tarballs** wired via `SLIME_AGENT_*_TARBALL`,
- a **reachable `ADAPTER_PUBLIC_HOST`** the sandbox can dial back to hit the adapter,
- and the managed **GPU Runtime** wiring for AgentCore (today the GPU path runs on this box's L40S;
  see `docs/demo-plan.md` Option B).

Until those are in place, use **Demo A** (offline-capable, our harness) or **Demo B** (GPU sandbox,
our harness) to exercise the training path.
