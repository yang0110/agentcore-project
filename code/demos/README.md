# Demos — the 2×3 matrix

Six demo directories, each **independently runnable**. Each one owns its `data/` (train +
held-out eval jsonl), its `ckpt/` output dir, and its `run.sh`, so you can run any single demo
without touching the others or the shared `code/p4_slime_train/` tree. The `agentcore_slime_demo`
library is **imported from the shared copy** at `code/agentcore_slime_demo/` (put on `PYTHONPATH`
as `/work/code` inside the container) — one source of truth, no per-demo copies. The other shared
things a demo *references* (not copies) are the base Megatron checkpoint at
`code/p4_slime_train/ckpt/Qwen3-4B_torch_dist` and the slime install at `/root/slime` inside the
container.

The matrix has two axes:

- **Task / sandbox** (columns): **A** = MBPP on a CPU microVM · **B** = KernelBench on a GPU sandbox.
- **Agent harness** (rows): **our** lightweight `run_episode` loop · the **real Claude Code** CLI
  running in a custom Docker/AgentCore container · a **mini Claude-Code-like** harness we wrote
  (in-process, no CLI). In every row the policy is still Qwen3-4B trained by slime; only the agent
  loop changes, talking to the policy through slime's Anthropic adapter.

|            | **A — MBPP / CPU microVM**            | **B — KernelBench / GPU sandbox**              |
|------------|----------------------------------------|-------------------------------------------------|
| **our harness**          | [`demo_a/`](demo_a/)   — 4 train + 4 infer + 0 env | [`demo_b/`](demo_b/)  — 4 train + 2 infer + 2 env (GpuPool 6,7) |
| **real Claude Code (Docker)** | [`demo_a2/`](demo_a2/) — Demo A + Claude Code CLI | [`demo_b2/`](demo_b2/) — Demo B + Claude Code CLI |
| **mini Claude-Code (in-process)** | [`demo_a3/`](demo_a3/) — Demo A + mini harness | [`demo_b3/`](demo_b3/) — Demo B + mini harness |

All six share the training path: slime async/decoupled (`train_async.py`), Qwen3-4B, GRPO, the
checkpoint-deadlock fix (`--async-save` + `--custom-megatron-init-path …enable_persistent_async_ckpt`),
and save-only-at-end. They differ only in the **task**, the **sandbox**, and (rows 2/3) the
**agent harness** (a **custom container** for row 2; none for row 3).

## Which to run

- **`demo_a`** — the reference. Runs offline-friendly (`SANDBOX=local`) or on real AWS Code
  Interpreter microVMs (`SANDBOX=agentcore`). Most battle-tested.
- **`demo_b`** — GPU-in-sandbox path; envs dense-pack on physical GPUs 6,7 via `GpuPool`. Trains on
  the real KernelBench (level_1, 100 single-op CUDA problems; 20 held-out level_2 fusions for eval).
  Defaults `RB=4/NS=8/NR=20`; keep `NR × RB ≤ 100` (see its README).
- **`demo_a2` / `demo_b2`** — swap our loop for the **real Claude Code CLI**. These need a custom
  container image (Node 22 + Claude Code, built from `_agent_assets/`) and, on the GPU side, CUDA +
  torch too. They do **not** run offline. See each dir's README.
- **`demo_a3` / `demo_b3`** — a **mini harness that mimics Claude Code** (call model → pick tool →
  run in sandbox → repeat) in ~1 file of Python, running in the trainer process. **No Node, no CLI,
  no custom container**: the sandbox is the vanilla Code Interpreter (A3) / GPU Runtime (B3), and
  the harness logic has an offline smoke (`code/scripts/mini_agent_smoke.py`). Start here if you
  want a real agentic loop without the CLI packaging. Built from the AgentCore
  [harness](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/harness.html) concept.

Each directory has its own `README.md` with exact build/run commands and knobs.
