# Demos — the 2×2 matrix

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

The matrix has two axes:

- **Task / sandbox** (columns): **A** = MBPP on a CPU microVM · **B** = KernelBench on a GPU sandbox.
- **Agent harness** (rows): **our** lightweight `run_episode` loop · a **mini Claude-Code-like**
  harness we wrote (no CLI) that runs on its **own AgentCore Runtime**. In every row the policy is
  still Qwen3-4B trained by slime; only the agent loop changes, talking to the policy through slime's
  Anthropic adapter.

|            | **A — MBPP / CPU microVM**            | **B — KernelBench / GPU sandbox**              |
|------------|----------------------------------------|-------------------------------------------------|
| **our harness**          | [`demo_cpu_single/`](demo_cpu_single/)   — 4 train + 4 infer + 0 env | [`demo_gpu_single/`](demo_gpu_single/)  — 4 train + 4 infer + 0 env (checker GPU on remote g6 sandbox) |
| **mini Claude-Code (own Runtime)** | [`demo_cpu_2runtime/`](demo_cpu_2runtime/) — Demo A + mini harness | [`demo_gpu_2runtime/`](demo_gpu_2runtime/) — Demo B + mini harness |

(A prior third row ran the **real Claude Code CLI** in a custom container; that path has been
removed in favor of the mini harness.) Demo [`demo_gpu_shared/`](demo_gpu_shared/) extends the B / mini-harness path
with a Ray GPU-sharing farm on the remote g6 sandbox Runtime.

All four matrix demos share the training path: slime async/decoupled (`train_async.py`), Qwen3-4B,
GRPO, the checkpoint-deadlock fix (`--async-save` + `--custom-megatron-init-path …enable_persistent_async_ckpt`),
and save-only-at-end. They differ only in the **task**, the **sandbox**, and (the mini-harness rows)
the **agent harness** (a **dedicated harness Runtime**).

## Which to run

- **`demo_cpu_single`** — the reference. Runs offline-friendly (`SANDBOX=local`) or on real AWS Code
  Interpreter microVMs (`SANDBOX=agentcore`). Most battle-tested.
- **`demo_gpu_single`** — GPU-sandbox path; the KernelBench checker's GPU is the remote g6 sandbox Runtime, so
  the trainer box stays 4 train + 4 infer + 0 env. Trains on
  the real KernelBench (level_1, 100 single-op CUDA problems; 20 held-out level_2 fusions for eval).
  Defaults `RB=4/NS=8/NR=20`; keep `NR × RB ≤ 100` (see its README).
- **`demo_cpu_2runtime` / `demo_gpu_2runtime`** — a **mini harness that mimics Claude Code** (call model → pick tool →
  run in sandbox → repeat) in ~1 file of Python. **No Node, no CLI** (it is not the Claude Code CLI),
  but it is deployed the **target** way: the harness runs on its **own AgentCore Runtime #1** and the
  sandbox is a **separate Runtime #2** (CPU Code-Interpreter-style for A3 / GPU for B3) — build and
  create both with `code/deploy/deploy_two_runtime.py`, then pass the two ARNs to `run.sh`. The
  earlier in-process shortcut has been removed; the harness logic still has an offline smoke
  (`code/scripts/mini_agent_smoke.py`). Start here if you want a real agentic loop without the CLI
  packaging. Built from the AgentCore
  [harness](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/harness.html) concept.

Each directory has its own `README.md` with exact build/run commands and knobs.
