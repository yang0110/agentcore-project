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

Two sandbox substrates, plus two resilience demos:

| Substrate | Demo | Task / point |
|---|---|---|
| **CPU serverless microVM** (Code Interpreter) | [`demo_cpu_single/`](demo_cpu_single/) | the reference — MBPP on a CPU microVM, 4 train + 4 infer + 0 env |
| | [`demo_cpu_math/`](demo_cpu_math/) | math word problems (generation-bound: long reasoning, trivial check) |
| | [`demo_cpu_quiesce/`](demo_cpu_quiesce/) | **resilience** — heartbeat that keeps the microVM warm across the weight-sync barrier |
| **GPU Runtime instance** (g6 capacity provider) | [`demo_gpu_single/`](demo_gpu_single/) | KernelBench (level_1 CUDA); checker's GPU is the remote g6 sandbox Runtime, so the trainer box stays 4 train + 4 infer + 0 env |
| | [`demo_gpu_shared/`](demo_gpu_shared/) | KernelBench + a Ray GPU-sharing farm — many envs share one GPU on the remote g6 sandbox Runtime |
| | [`demo_gpu_snapshot/`](demo_gpu_snapshot/) | **resilience** — application-level snapshot & resume across a GPU instance recycle |

In every demo the **harness is in-process**; the sandbox is the AgentCore surface that changes (a
CPU Code-Interpreter microVM, or a GPU Runtime instance on a capacity provider). A prior variant that
hosted the harness on its *own* AgentCore Runtime (the "two-Runtime" demos) has been removed — see
[`../../docs/harness-sandbox-decoupling.md`](../../docs/harness-sandbox-decoupling.md) for why
in-process is the recommended path.

## Which to run

- **`demo_cpu_single`** — the reference. Runs offline-friendly (`SANDBOX=local`) or on real AWS Code
  Interpreter microVMs (`SANDBOX=agentcore`). Most battle-tested.
- **`demo_gpu_single`** — GPU-sandbox path; the KernelBench checker's GPU is the remote g6 sandbox Runtime, so
  the trainer box stays 4 train + 4 infer + 0 env. Trains on
  the real KernelBench (level_1, 100 single-op CUDA problems; 20 held-out level_2 fusions for eval).
  Defaults `RB=4/NS=8/NR=20`; keep `NR × RB ≤ 100` (see its README).
- **`demo_gpu_shared`** — extends the GPU-sandbox path with a Ray GPU-sharing farm on the remote g6
  sandbox Runtime (many envs packed onto one GPU).
- **`demo_cpu_quiesce` / `demo_gpu_snapshot`** — the two **resilience** demos (keeping a paused
  sandbox alive / recovering a lost one across the weight-sync barrier). Both run offline in seconds,
  no AWS or GPU. See [`../../docs/demos-program.md`](../../docs/demos-program.md).

The training demos share the training path: slime async/decoupled (`train_async.py`), Qwen3-4B,
GRPO, the checkpoint-deadlock fix (`--async-save` + `--custom-megatron-init-path …enable_persistent_async_ckpt`),
and save-only-at-end. They differ only in the **task** and the **sandbox**.

Each directory has its own `README.md` with exact build/run commands and knobs.
