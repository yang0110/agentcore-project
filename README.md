# AgentCore × slime — agentic RL with an AWS-hosted sandbox

This repo demonstrates using **AWS Bedrock AgentCore Runtime as the execution sandbox** for
**agentic reinforcement learning**, with **slime** (THUDM/slime) as the trainer. An LLM policy
learns to solve tasks by *acting* in a real sandbox — writing code, running it, reading the
failure, fixing it — and the whole multi-turn trajectory is turned into loss-masked RL training
data. The design follows the **Agent Lightning v1.0** principle of decoupling the agent harness
from the trainer (arXiv 2608.17528).

> New here? Read this file, then `docs/demos-guide.md` (the operator's manual) and
> `code/README.md` (the code walkthrough).

---

## What is AgentCore?

**AWS Bedrock AgentCore Runtime** is a managed service for running AI agents and their tools.
We use it as the **sandbox** where the policy's tool actions (running code) execute — isolated
from the training cluster. Two AgentCore modes map to our two demos:

- **Code Interpreter** — a serverless, CPU **microVM** you drive over an SDK
  (`CodeInterpreter.invoke("executeCode"/"executeCommand")`), returning `stdout/stderr/exitCode`.
  Cheap, fast to start, 8-hour sessions. This is the default for RL: tool execution is just
  shell/IO, so it needs no GPU. → **Demo A**.
- **Runtime instances + a GPU capacity provider** — agents run on your own EC2 (incl. GPU
  families like `g6.xlarge` L40S), sessions up to 14 days. Use this only when the *task itself*
  needs CUDA inside the sandbox. → **Demo B** (today the GPU path runs on this box's L40S; the
  managed GPU Runtime container is the remaining piece — see `docs/demo-plan.md` Option B).

**Key point (Agent Lightning decoupling):** the *policy's* training GPUs always live on slime's
cluster. The *sandbox* only needs a GPU when the *task* does. Details + citations in
`docs/agentcore-slime-integration.md` and `docs/sandbox-comparison.md`.

---

## Which sandbox does a task need? Three tiers

> The current demo set is **3 main demos** (`demo_cpu_tasks` [+ `SCALE=1` harness-scaling preset, + Terminal-Bench as
> `DATASET=terminalbench`, formerly `demo_cpu_terminalbench`], `demo_cpu_swebench` [+ its EKS twin `demo_cpu_swebench_eks`], `demo_gpu_farm` `FARM=fractional|whole`, local by default or
> `LOCAL_GPU=0` on a remote g6) + 1 experimental
> (`demo_cpu_branching`) in **`code/demos/`**. In every demo the mini_claude loop runs inside the
> episode's sandbox session, dispatched by the decoupled harness tier (`harness_pool.py`). Start at
> [`code/demos/README.md`](code/demos/README.md). The "Demo A / Demo B" sections below are the
> original two-demo layout and are kept as history.

| Tier | Sandbox | Use it when | Demos |
|---|---|---|---|
| **1 · Managed Code Interpreter** | `SANDBOX=agentcore`, no image, no Runtime | **probes only** (since 2026-09-28): no route back to the adapter, so the in-session agent loop can't run there | `demo_cpu_tasks/compare_backends.py`, `demo_cpu_tasks/terminalbench/warmstart/` |
| **2 · Own Runtime, generic image** | `SANDBOX=agentcore-runtime` + stdlib / `libs` image (microVM) | no task-specific environment: **every literature task** (code, math, SQL, RAG, tool calls, data science); VPC dial-back, heartbeat, > 30 invokes/s | `demo_cpu_tasks` (incl. `SCALE=1`, 256 sessions in flight), `demo_cpu_tasks/terminalbench/warmstart/` |
| **3 · Own Runtime, custom image** | task-specific image | the environment differs: own OS per task (Terminal-Bench), own repo+env per SWE-bench instance (CPU Instance), Claude Code CLI, CUDA (GPU Instance) | `demo_cpu_tasks DATASET=terminalbench`, `demo_cpu_swebench`, `demo_cpu_branching`, `demo_gpu_farm` |

Pick the lowest tier that works. A tier-1 task gives the same reward on tier 2 (`compare_backends.py`).

---

## The demos

Both demos are the **same** multi-turn loop (write `solution.py` → run a checker in the sandbox
→ read the failure → fix → repeat), the **same** policy contract, and the **same** slime
training-data path. They differ in exactly two things: the **sandbox** and the **task**.

| | **Demo A — CPU microVM** | **Demo B — GPU instance** |
|---|---|---|
| AgentCore mode | Code Interpreter (serverless microVM) | Runtime instance + GPU capacity provider |
| Dataset | **MBPP** (`mbpp.py`) | **KernelBench** (`kernelbench.py`, real `ScalingIntelligence/KernelBench`, 270 CUDA problems) |
| Checker verifies | assert-style unit tests pass | output matches a torch reference **and** ran on `cuda` |
| Sandbox compute | CPU | CUDA inside the sandbox |
| Scaling story | many cheap microVMs | **GpuPool** dense-packs ~100 envs on a few GPUs |

Everything runs on top of one **shared harness** (`code/agentcore_slime_demo/harness.py`,
`run_episode`) with three pluggable pieces: a **policy** (`policy.py`), a **sandbox**
(`sandbox.py`: `LocalSandbox` offline / `AgentCoreSandbox` on AWS), and a **task** (`task.py`).
Swap the sandbox + task and you switch demos; the loop is unchanged.

### Async mode → disaggregation at two levels

Agentic rollouts have long, variable latency, so the canonical run path uses slime's **async
driver `train_async.py`**, which overlaps rollout *N+1* with training of *N*. Async forces
**disaggregation**, and we apply it at **two levels**:

1. **Trainer ⟂ inference (slime's decoupled placement).** The Megatron **trainer** GPUs and the
   SGLang **inference** GPUs are disjoint — no `--colocate`. Async hard-requires this
   (`train_async.py` asserts `not args.colocate`).
2. **Harness ⟂ sandbox (Agent Lightning disaggregation).** The **harness** (the agent loop) and
   the **sandbox** (where its tool actions execute) are separate components too. **AgentCore is
   used as the sandbox, and the harness now runs in it** (since 2026-09-28): the mini_claude loop
   runs inside the episode's own sandbox session and dials the trainer's per-engine
   `AnthropicAdapter` back, so it scales **with the sandbox, not as part of the trainer** (DSec's
   "move the agent loop out of the trainer"). What stays in the trainer is a warm dispatch tier
   (`harness_pool.py`: `HARNESS_WORKERS` workers, default RB×NS) and the adapters. The harness must
   scale along with inference and the sandbox, or a few coroutines become the ceiling:
   `throughput = min(inference, sandbox, harness)` (`docs/harness-scaling.md`). The earlier
   in-process design and the removed two-Runtime variant are in `docs/harness-sandbox-decoupling.md`.

**Why this matters: all five components scale independently.** Disaggregating both levels means
the pipeline is five separately-scalable services — **trainer**, **inference engine**, **proxy**
(`AnthropicAdapter`), **harness**, and **sandbox** — each sized to its own bottleneck rather than
lock-stepped to the others:

| Component | Scale it by | Bound by |
|---|---|---|
| **Trainer** (Megatron) | more train GPUs / TP·DP | gradient compute |
| **Inference** (SGLang) | more engines / GPUs | rollout token throughput |
| **Proxy** (`AnthropicAdapter`) | more adapter replicas | request fan-in (CPU/IO) |
| **Harness** (mini_claude, in the sandbox session) | `HARNESS_WORKERS` dispatch workers + more sessions | agent-loop latency; trainer-side dispatch |
| **Sandbox** (Code Interpreter / GPU Runtime) | more sessions / `GpuPool` gate | tool-exec (CPU or CUDA) |

GPU split on this 8×L40S box (trainer + inference; sandbox is CPU for A, and for B a *local*
stand-in on GPUs 6,7 until it moves to a real GPU Runtime instance):

| | Train (Megatron) | Inference (SGLang) | Envs (sandbox checkers) |
|---|---|---|---|
| **Demo A** | 4 GPUs | 4 GPUs | 0 (CPU envs) |
| **Demo B** | 4 GPUs | 2 GPUs | **2** (physical 6,7 via GpuPool) |

For Demo B, ~100 concurrent envs share just 2 GPUs because GPU use is *bursty* and
*memory*-bound: `GpuPool` is a semaphore + per-env memory cap that admits a few checkers at a
time and queues the rest (see `docs/gpu-density.md` and `docs/demos-guide.md` §6).

---

## Run the demos manually

### 0. Offline sanity (no AWS, no GPU, no slime)

```bash
cd code
python3 scripts/smoke.py                              # multi-turn loop, LocalSandbox + MockPolicy
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests -q   # 19 tests
python3 scripts/e2e_rl.py --profile cpu --policy mock --limit 3   # Demo A dry run (MBPP)
python3 scripts/e2e_rl.py --profile gpu --policy mock             # Demo B dry run (KernelBench)
```

### 1. Real policy rollouts (Qwen3-4B), on real AWS AgentCore microVMs (Demo A)

```bash
cd code
HF_HOME=./.hf_cache SANDBOX=agentcore AWS_REGION=us-west-2 \
    python3 scripts/e2e_rl.py --profile cpu --policy real --limit 3
```

### 2. Full slime training (async/decoupled) — the canonical path

All slime training runs inside the persistent container `agentcore_p4` (`slimerl/slime`, torch
2.11 + clean Megatron/SGLang). Start it once with `code/p4_slime_train/00_start_container.sh`.

```bash
# Build the datasets (once) — a train set and a held-out eval set per demo
docker exec agentcore_p4 python /work/code/p4_slime_train/02_build_dataset.py \
    --dataset mbpp --split "train+test" --limit 500 --out /work/code/p4_slime_train/data/mbpp_train.jsonl
docker exec agentcore_p4 python /work/code/p4_slime_train/02_build_dataset.py \
    --dataset mbpp --split validation --limit 8 --out /work/code/p4_slime_train/data/mbpp_eval.jsonl
docker exec agentcore_p4 python /work/code/p4_slime_train/02_build_dataset.py \
    --dataset kernelbench --level level_1 --limit 100 --out /work/code/p4_slime_train/data/kernelbench_train.jsonl
docker exec agentcore_p4 python /work/code/p4_slime_train/02_build_dataset.py \
    --dataset kernelbench --level level_2 --limit 20 --out /work/code/p4_slime_train/data/kernelbench_eval.jsonl  # 20 held-out fusions

# Demo A — MBPP, GPU split 4 train + 4 infer + 0 env
docker exec agentcore_p4 bash -lc \
    'cd /work && DEMO=a SANDBOX=local bash code/p4_slime_train/run_async.sh'

# Demo B — KernelBench, GPU split 4 train + 2 infer + 2 env (GpuPool on physical 6,7)
# NR=20 keeps NR*RB (80) <= the 100-prompt level_1 pool (save-only-at-end); see run_async.sh DEMO=b note.
docker exec agentcore_p4 bash -lc \
    'cd /work && DEMO=b NR=20 SANDBOX=local GPU_POOL_CONCURRENCY=16 bash code/p4_slime_train/run_async.sh'
```

`SANDBOX=agentcore` swaps the env to real AWS Code Interpreter microVMs (needs
`AWS_BEARER_TOKEN_BEDROCK` + `AWS_REGION` in the container env). Tunables: `RB` (prompts/step),
`NS` (GRPO group size), `NR` (num-rollout; real steps = NR−1), `TEMP`, `CONC`, `EVAL_INTERVAL`
(eval every N steps, default 10; `0` disables), `EVAL_NS` (samples/eval prompt). Demo B also:
`GPU_POOL_CONCURRENCY` (envs on the GPU at once), `GPU_POOL_PER_ENV_MIB`, `GPU_POOL_DEVICES`.

### Checkpoints & eval cadence

To save disk we do **not** snapshot mid-training. slime always force-saves at the final rollout,
so `run_async.sh` sets `--save-interval` huge (`SAVE_INTERVAL`, default 1,000,000). slime *also*
force-saves at every **epoch boundary** (`step % (dataset_size/RB) == 0`), so to get exactly one
checkpoint the train set must be large enough that one epoch spans the whole run:
`dataset_size ≥ NR × RB`. With the defaults (`NR=100`, `RB=4`) the MBPP train pool is built at 500
prompts (125 steps/epoch ≥ 99 real steps) → the **only** checkpoint written is the one at the
**end** of training (under `--save`). Meanwhile a **held-out eval runs every `EVAL_INTERVAL` (default 10) training steps**:
slime routes each eval prompt through our *same* custom generate (`evaluation=True`) — one full
multi-turn agentic episode, no gradient — and logs a solve rate as `eval/<name>`. So per-step
training metrics and per-10-step eval land in the same log, and go to two separate CSVs (below).

> `run_agentcore_mbpp_qwen3_4b.sh` is the legacy **sync/colocate** script (first proven P4 run);
> `run_async.sh` is the current async default.

---

## Logs, metrics → CSV, and figures

slime logs metrics to stdout (and to wandb/tensorboard if configured). Each `run_*.sh` tees the
full ray-job output to a log file under `code/p4_slime_train/` (e.g. `train_async_demoB.log`).
slime prints these metric families, each tagged with the step id:

- `step N: {train/pg_loss, train/grad_norm, train/kl_loss, train/train_rollout_logprob_abs_diff, …}` — **per step**
- `rollout N: {rollout/raw_reward (solve rate), rollout/advantages, rollout/response_lengths, …}` — **per step**
- `perf N: {perf/step_time, perf/wait_time_ratio, perf/actor_train_tok_per_s, …}` — **per step**
- `eval N: {eval/<name> (held-out solve rate), eval/step, …}` — **every `EVAL_INTERVAL` (10) steps**

**`code/p4_slime_train/metrics.py`** parses the log into **two CSVs** — a **per-step train CSV**
(`--csv`, one row per training step, `step`/`rollout`/`perf` merged) and a **per-eval CSV**
(`--eval-csv`, one row per eval, keyed by `eval/step`, written only if eval ran) — and optionally
**plots** the key curves (including held-out eval) to a **PNG**:

```bash
python3 code/p4_slime_train/metrics.py \
    code/p4_slime_train/train_async_demoB.log \
    --csv code/p4_slime_train/train_async_demoB.csv \
    --eval-csv code/p4_slime_train/train_async_demoB.eval.csv \
    --plot code/p4_slime_train/train_async_demoB.png
```

The train CSV columns lead with the metrics that matter for RL health: `rollout/raw_reward` (the
learning signal), `train/grad_norm` (is the policy moving?), `train/kl_loss` (drift from the frozen
reference), `train/train_rollout_logprob_abs_diff` (weight-sync correctness), and the `perf/*`
timings. The eval CSV is `eval/step` + the held-out solve rate(s). The PNG is a small multi-panel
figure of those curves vs training step. The metric map across all stages (trainer → sandbox →
inference → weight sync) is documented in `docs/observability.md`.

---

## Repo map

```
code/
  agentcore_slime_demo/   # the library: sandbox, task, policy, harness, generate, gpu_pool, ...
  scripts/                # offline smokes + e2e_rl.py (unified Demo A/B driver)
  tests/                  # 19 offline tests
  p4_slime_train/         # real slime training in Docker: run_async.sh, 02_build_dataset.py,
                          #   metrics.py, ckpt/, and the training logs
docs/                     # integration reference, demos-guide, observability, gpu-density,
                          #   fault-recovery, sandbox-comparison, demo-plan, Agent Lightning notes
```

Full operator's manual: **`docs/demos-guide.md`**. Code walkthrough: **`code/README.md`**.
