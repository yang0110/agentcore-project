# Demos — the sandbox demo set

Every demo here is about **the sandbox**: the agent harness stays **in-process** (a warm
`run_episode` loop next to the trainer), and each demo varies the **task** and the **AgentCore
sandbox substrate** it talks to. The policy is always Qwen3-4B trained by slime (async/decoupled
GRPO), talking to the agent loop through slime's Anthropic adapter; only the task and the sandbox
change.

The demo directories are each **fully self-contained**. Every demo owns its `data/` (train +
held-out eval jsonl), its `ckpt/` output dir, its `run.sh`, a **vendored copy of the
`agentcore_slime_demo/` library** (it imports only this local copy — `PYTHONPATH` puts the demo dir
first and no longer includes `/work/code`), and its own **`deploy/`** with the exact Dockerfile(s)
its sandbox Runtime image is built from. So a client can read one folder end-to-end and see the
whole recipe: task → dataset → Dockerfile → launch → run → terminate. Each README carries the
AgentCore **launch / terminate / latency** basics inline.

Three things a demo *references* rather than copies (shared infra that would only drift if
duplicated): the base Megatron checkpoint at `code/p4_slime_train/ckpt/Qwen3-4B_torch_dist`, the
slime install at `/root/slime`, and the **shared deploy orchestrator** `code/deploy/deploy_two_runtime.py`
(+ its `redeploy_*`/`smoke_*` helpers) that creates the IAM roles, ECR repos, VPC endpoints, and the
GPU capacity provider. The top-level `code/agentcore_slime_demo/` still exists because `code/deploy/`
builds the Runtime images from it and `code/tests/` imports it.

## The demos

**12 demos** — 8 CPU + 4 GPU. One of them, `demo_cpu_tasks`, is a single folder that trains
**any of ~19 datasets, or any mix of them,** through a `DATASET=` selector (`DATASET=mbpp`,
`DATASET=gsm8k,gretelsql`, or the prebuilt `DATASET=mixed`). It replaces the ten former
per-dataset folders `demo_cpu_mbpp` … `demo_cpu_mcp` and the mixed-task `demo_cpu_mixtask`, which
differed only in their data — a single dataset is the one-element mix. Likewise `demo_gpu_farm` picks its GPU split through `FARM=fractional|whole` (it
replaces `demo_gpu_shared_single`, `demo_gpu_shared_multi` and `demo_gpu_local`, which differed
only in the farm knobs).

**Sandbox mode by substrate.** The **CPU** demos are **remote-only** — they run only on remote
managed serverless AgentCore sandbox sessions (`SANDBOX=agentcore-runtime` default, or `agentcore`
Code-Interpreter); their `run.sh` **rejects `SANDBOX=local`**, so a CPU training run can never
silently fall off the AgentCore substrate (the serverless CPU sandbox is always available). The
**GPU** demos keep a `local` fallback, because a remote g6 instance may be unavailable.

| Substrate | Demo | Task / point |
|---|---|---|
| **CPU serverless microVM** | [`demo_cpu_tasks/`](demo_cpu_tasks/) | the reference — ONE demo, many RLVR datasets via `DATASET=`: real HF `mbpp` (default) `gsm8k` `math500` `deepscaler` `humaneval` `gretelsql` `hotpotqa` `xlam` `mcp`, plus the synthetic task-family zoo (`math` `swe` `sql` `ctf` `rag` `tool_use` …); **mixed-task** batches via a comma list or `DATASET=mixed` (math+sql+ctf+datasci); 4 train + 4 infer + 0 env |
| | [`demo_cpu_imagepool/`](demo_cpu_imagepool/) | mixed-task RL over a pool of prebuilt images (one image → one Runtime per task family) |
| | [`demo_cpu_terminalbench/`](demo_cpu_terminalbench/) | a LARGE image pool (one image per Terminal-Bench task, ~229), one isolated microVM per episode, pre-opened to hide the cold start |
| | [`demo_cpu_branching/`](demo_cpu_branching/) | one episode → many training samples ("fan-out") |
| | [`demo_cpu_customruntime/`](demo_cpu_customruntime/) | real RL training on YOUR custom AgentCore Runtime (CPU) |
| | [`demo_cpu_heartbeat/`](demo_cpu_heartbeat/) | **resilience** — heartbeat that keeps the microVM warm across the weight-sync barrier |
| **CPU capacity-provider Instance** | [`demo_cpu_bigimage/`](demo_cpu_bigimage/) | a LARGE sandbox image (past the 2 GB microVM cap) on a warm CPU Instance |
| **rollout scaling** (scales harness + inference + sandbox together) | [`demo_cpu_scale_lanes/`](demo_cpu_scale_lanes/) | one task (DeepScaleR) on **4 × 4 × 4 lanes** — 4 warm mini-Claude harness workers : 4 SGLang engines : 4 persistent microVMs, each worker pinned 1:1:1; `LANES=N` scales them together → the async-RL scaling law `throughput = min(inference, sandbox, harness)` |
| **GPU Runtime instance** (g6 capacity provider) | [`demo_gpu_single/`](demo_gpu_single/) | KernelBench (level_1 CUDA); checker's GPU is the remote g6 sandbox Runtime, so the trainer box stays 4 train + 4 infer + 0 env |
| | [`demo_gpu_customruntime/`](demo_gpu_customruntime/) | real KernelBench RL on YOUR custom GPU Runtime |
| | [`demo_gpu_snapshot/`](demo_gpu_snapshot/) | **resilience** — application-level snapshot & resume across a GPU instance recycle |
| **local GPU Ray farm** (this box's GPUs 6,7) | [`demo_gpu_farm/`](demo_gpu_farm/) | KernelBench on a local Ray farm: `FARM=fractional` (default) packs two episodes onto ONE GPU (0.5 + 0.5); `FARM=whole` gives each episode a whole GPU across 6,7 |

`demo_cpu_tasks` is one in-process-harness loop; each `DATASET` differs only in its seed rows +
hidden RLVR checker, and gets its own `data/<name>_{train,eval}.jsonl`, `eval/<name>` metric and
`ckpt/Qwen3-4B_<name>_async` (see its README for the dataset table). Because every row names its own
task, a comma list (`DATASET=a,b,c`, interleaved by `run.sh`, `eval/mix`) or the prebuilt
`DATASET=mixed` (`eval/mixed`) trains one batch across families with no harness change.

The demos above vary the **sandbox**; `demo_cpu_scale_lanes` is the exception — it fixes the task
and scales the **rollout tiers themselves** as 1:1:1 lanes (default 4 harness workers : 4 SGLang
engines : 4 microVMs), to show the rollout scaling law and how the three tiers route/load-balance. See
[`../../docs/harness-scaling.md`](../../docs/harness-scaling.md).

In the other demos the **harness is in-process and single**; the sandbox is the AgentCore surface that changes (a
CPU Code-Interpreter microVM, or a GPU Runtime instance on a capacity provider). A prior variant that
hosted the harness on its *own* AgentCore Runtime (the "two-Runtime" demos) has been removed — see
[`../../docs/harness-sandbox-decoupling.md`](../../docs/harness-sandbox-decoupling.md) for why
in-process is the recommended path.

## Which to run

- **`demo_cpu_tasks`** — the reference. `DATASET=<name> bash run.sh` on remote AWS CPU microVMs
  (`SANDBOX=agentcore-runtime`, or `agentcore` Code Interpreter). Most battle-tested.
  `DATASET=a,b,c` or `DATASET=mixed` for a mixed-task batch.
- **`demo_gpu_single`** — GPU-sandbox path; the KernelBench checker's GPU is the remote g6 sandbox Runtime, so
  the trainer box stays 4 train + 4 infer + 0 env. Trains on
  the real KernelBench (level_1, 100 single-op CUDA problems; 20 held-out level_2 fusions for eval).
  Defaults `RB=4/NS=8/NR=20`; keep `NR × RB ≤ 100` (see its README).
- **`demo_gpu_farm`** — GPU sharing on a LOCAL Ray farm on this box's GPUs 6,7 (two-step launch:
  start the host farm via `run_local_gpu_runtime.sh fractional|whole`, then run
  `FARM=<same> bash run.sh` inside `agentcore_p4`). `fractional` packs two episodes onto ONE GPU
  (0.5 + 0.5); `whole` gives each episode a whole GPU across 6,7.
- **`demo_cpu_heartbeat` / `demo_gpu_snapshot`** — the two **resilience** demos (keeping a paused
  sandbox alive / recovering a lost one across the weight-sync barrier). Both run offline in seconds,
  no AWS or GPU. See [`../../docs/demos-program.md`](../../docs/demos-program.md).

The training demos share the training path: slime async/decoupled (`train_async.py`), Qwen3-4B,
GRPO, the checkpoint-deadlock fix (`--async-save` + `--custom-megatron-init-path …enable_persistent_async_ckpt`),
and save-only-at-end. They differ only in the **task** and the **sandbox**.

**`run.sh` vs `train.sh`.** Every demo has `run.sh`, the full runner
(all env knobs, defaults to a short smoke). Five also ship **`train.sh`** (`demo_cpu_tasks`,
`demo_cpu_imagepool`, `demo_cpu_terminalbench`, `demo_cpu_branching`, `demo_gpu_single`), a thin train-to-convergence wrapper
over it that exposes just the three top-level knobs a client tunes — training **steps** (`STEPS` →
`NR`), eval **cadence** (`EVAL_EVERY` → `EVAL_INTERVAL`), and the **sandbox** counts (`SANDBOX_CONC` →
`CONC`, and on CPU `SANDBOX_RUNTIMES` → the # of distinct sandbox Runtimes `K`) — passing everything
else through. Example: `STEPS=50 EVAL_EVERY=10 SANDBOX_CONC=4 ./train.sh`. See each README's
"Train to convergence" section.

Each directory has its own `README.md` with exact build/run commands and knobs.
