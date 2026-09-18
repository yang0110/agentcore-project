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

**16 demos**: two sandbox substrates (with a per-task CPU set + a GPU set), two resilience demos, a
harness-scaling demo, and a mixed-task demo.

**Sandbox mode by substrate.** The **CPU** demos are **remote-only** — they run only on remote
managed serverless AgentCore sandbox sessions (`SANDBOX=agentcore-runtime` default, or `agentcore`
Code-Interpreter); their `run.sh` **rejects `SANDBOX=local`**, so a CPU training run can never
silently fall off the AgentCore substrate (the serverless CPU sandbox is always available). The
**GPU** demos keep a `local` fallback, because a remote g6 instance may be unavailable.

| Substrate | Demo | Task / point |
|---|---|---|
| **CPU serverless microVM** (Code Interpreter) | [`demo_cpu_single/`](demo_cpu_single/) | the reference — MBPP coding on a CPU microVM, 4 train + 4 infer + 0 env; task-family zoo via `FAMILY` |
| | [`demo_cpu_math/`](demo_cpu_math/) | math word problems (multi-task: `gsm8k`/`math500`/`deepscaler` via `FAMILY`) |
| | [`demo_cpu_mixtask/`](demo_cpu_mixtask/) | **mixed-task** — every rollout step's batch spans several families at once (math+sql+ctf+tool_use); remote-only |
| _per-task, real HF datasets_ | [`demo_cpu_gsm8k/`](demo_cpu_gsm8k/) | GSM8K grade-school math (`openai/gsm8k`) |
| | [`demo_cpu_math500/`](demo_cpu_math500/) | MATH-500 competition math (`HuggingFaceH4/MATH-500`) |
| | [`demo_cpu_deepscaler/`](demo_cpu_deepscaler/) | DeepScaleR olympiad math (`agentica-org/DeepScaleR-Preview-Dataset`) |
| | [`demo_cpu_humaneval/`](demo_cpu_humaneval/) | HumanEval Python completion (`openai/openai_humaneval`) |
| | [`demo_cpu_gretelsql/`](demo_cpu_gretelsql/) | text-to-SQL (`gretelai/synthetic_text_to_sql`) |
| | [`demo_cpu_hotpotqa/`](demo_cpu_hotpotqa/) | multi-hop retrieval QA (`hotpotqa/hotpot_qa`) |
| | [`demo_cpu_xlam/`](demo_cpu_xlam/) | function calling (`minpeter/xlam-function-calling-60k-parsed`) |
| | [`demo_cpu_mcp/`](demo_cpu_mcp/) | MCP-style tool calling (`Team-ACE/ToolACE`) |
| | [`demo_cpu_quiesce/`](demo_cpu_quiesce/) | **resilience** — heartbeat that keeps the microVM warm across the weight-sync barrier |
| **GPU Runtime instance** (g6 capacity provider) | [`demo_gpu_single/`](demo_gpu_single/) | KernelBench (level_1 CUDA); checker's GPU is the remote g6 sandbox Runtime, so the trainer box stays 4 train + 4 infer + 0 env |
| | [`demo_gpu_shared/`](demo_gpu_shared/) | KernelBench + a Ray GPU-sharing farm — many envs share one GPU on the remote g6 sandbox Runtime |
| | [`demo_gpu_snapshot/`](demo_gpu_snapshot/) | **resilience** — application-level snapshot & resume across a GPU instance recycle |
| **harness-scaling** (varies the *harness* tier, not the sandbox) | [`demo_cpu_harness_scale/`](demo_cpu_harness_scale/) | one task (DeepScaleR), a pool of **W warm mini-Claude harness workers** → the async-RL scaling law `throughput = min(inference, sandbox, harness)` |

The 8 per-task CPU demos (`gsm8k` … `mcp`) are the same in-process-harness loop as `demo_cpu_single`,
each pinned to one real HF dataset + its hidden RLVR checker (see the task/dataset table in each
README); `demo_cpu_single` reaches the same tasks via its `FAMILY` selector.

The demos above vary the **sandbox**; `demo_cpu_harness_scale` is the exception — it fixes the task
and sandbox and makes the **harness the scalable tier** (a warm in-process worker pool), to show the
rollout scaling law and how the three tiers route/load-balance differently. See
[`../../docs/harness-scaling.md`](../../docs/harness-scaling.md).

In the other demos the **harness is in-process and single**; the sandbox is the AgentCore surface that changes (a
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

**`run.sh` vs `train.sh`.** Every training demo has two entry points: `run.sh` is the full runner
(all env knobs, defaults to a short smoke), and **`train.sh`** is a thin train-to-convergence wrapper
over it that exposes just the three top-level knobs a client tunes — training **steps** (`STEPS` →
`NR`), eval **cadence** (`EVAL_EVERY` → `EVAL_INTERVAL`), and the **sandbox** counts (`SANDBOX_CONC` →
`CONC`, and on CPU `SANDBOX_RUNTIMES` → the # of distinct sandbox Runtimes `K`) — passing everything
else through. Example: `STEPS=50 EVAL_EVERY=10 SANDBOX_CONC=4 ./train.sh`. See each README's
"Train to convergence" section.

Each directory has its own `README.md` with exact build/run commands and knobs.
