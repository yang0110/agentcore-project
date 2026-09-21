# demo_gpu_customruntime — real KernelBench RL on YOUR custom GPU Runtime

**What this teaches.** The managed AgentCore Code Interpreter (`SANDBOX=agentcore`) runs on a CPU
serverless microVM — it has **no GPU**. If your RL task needs CUDA (compiling/running a kernel,
`torch.cuda`, KernelBench-style rewards), the sandbox must be **your own custom Runtime deployed on
an AgentCore capacity-provider *Instance*** (a g6 GPU box). This demo turns that capability into a
**real 10-step slime training run**: Qwen3-4B learns KernelBench CUDA kernels where the reward comes
from **compiling and running the kernel on a real GPU inside the sandbox** — which only the custom
Runtime provides. The training simply **cannot run** on the Code Interpreter, because there is no
CUDA device there.

The sandbox exposes the **exact same** task-agnostic contract as every other backend —
`write_file` / `exec` / `read_file` — so the in-process `run_episode` harness and the slime trainer
are byte-identical to `demo_gpu_single`; only the *substrate* the checker lands on differs.

| `SANDBOX=` value      | substrate                                  | `torch.cuda.is_available()` | trains KernelBench? |
|-----------------------|--------------------------------------------|-----------------------------|---------------------|
| `agentcore-runtime`   | **your custom Runtime on a g6 Instance**   | **True** — a real CUDA device | **yes** (this demo's default) |
| `agentcore`           | managed Code Interpreter, CPU microVM      | **False** (no GPU)          | no — checker has no CUDA |
| `local`               | offline subprocess (host)                  | whatever the host has       | only for the offline probe |

> **1 AgentCore session = 1 whole Instance.** GPUs live on capacity-provider Instances, not on the
> serverless microVMs, and a runtime session leases a whole instance 1:1 — there is no cross-session
> GPU time-slicing. So opening one dedicated session per episode makes the **instance count scale with
> concurrency**, which is not cost-feasible. This demo therefore defaults to the **warm-instance**
> setting (`SANDBOX=agentcore-ray-farm`): all episodes invoke the custom Runtime under **one shared
> `runtimeSessionId`** (`B4_SHARED_RUNTIME_SESSION`) + a per-episode `sub_session`, so they reuse **one
> warm g6** and a self-managed in-instance Ray farm hands each episode a fresh isolated env on a whole
> GPU (`B4_GPU_FRACTION=1.0`) — the same mechanism as `demo_gpu_shared_multi` / `demo_gpu_local`. Set
> `SANDBOX=agentcore-runtime` for the legacy one-dedicated-instance-per-session path.

## What runs (real trainer)

KernelBench-style CUDA-kernel tasks: the policy writes `solution.py`, a GPU checker (`run_checks.py`)
imports torch, compiles/executes the kernel on `cuda`, and compares against a torch reference; the
harness reads the failure, the policy fixes it, multi-turn. Reward is the checker's pass/perf signal.
The harness runs **in-process**; the sandbox (`write_file`/`exec`/`read_file`) is a **remote custom
GPU Runtime** on a g6. slime trains async/decoupled (`train_async.py`, no `--colocate`): 4 train +
4 inference GPUs on this box, the checker GPU is the remote g6's.

- **10 train steps + 1 eval + one final checkpoint** — the shared smoke budget: `NR=11`
  ⇒ `range(1,NR)` = 10 real steps; `EVAL_INTERVAL=10` fires one held-out eval (20 level_2 fusion
  problems, disjoint from the 100 level_1 train problems); `SAVE_INTERVAL=1000000` + explicit
  `--num-rollout` ⇒ the only checkpoint written is slime's forced final-rollout one.

`gpu_torch_smoke.py` is kept as the seconds-long **offline CUDA-contract probe**: it
`write_file()`s a tiny torch-on-CUDA probe, `exec()`s it, `read_file()`s the result, and prints
whether a CUDA device was visible — the same `local` / `agentcore` / `agentcore-runtime` contrast as
the table above, **without training**. Use it to confirm a freshly-deployed GPU Runtime actually
sees a GPU before pointing the trainer at it.

## Reproduce (inside the `agentcore_p4` container)

```bash
# 0. Deploy the custom GPU Runtime (once) and export its ARN — see "Deploying" below.
export AGENTCORE_SANDBOX_RUNTIME_ARN=arn:aws:bedrock-agentcore:us-west-2:<acct>:runtime/<gpu-rt>

# 1. (optional) offline CUDA-contract probe — no training, proves write/exec/read + reports CUDA.
#    (run.sh is now the REAL trainer; the probe is its own script, invoked directly.)
python3 gpu_torch_smoke.py --backend agentcore-runtime   # expect cuda_available=True

# 2. REAL RL train + eval on the custom GPU Runtime (10 train + 1 eval, single final ckpt):
docker exec -e NR=11 -e EVAL_INTERVAL=10 -e EVAL_NS=2 \
    -e AGENTCORE_SANDBOX_RUNTIME_ARN="$AGENTCORE_SANDBOX_RUNTIME_ARN" agentcore_p4 \
    bash -lc 'cd /work && bash code/demos/demo_gpu_customruntime/run.sh 2>&1 | tee code/demos/demo_gpu_customruntime/smoke_run.log'
```

The KernelBench train/eval JSONL ships in `data/` (byte-identical to `demo_gpu_single/data`), so
there is no `prepare_data.sh` — the data is already present.

> **⚠️ Not verified on hardware.** This demo was converted from the offline contract probe to a real
> trainer, but the end-to-end training path (a live custom GPU Runtime + slime) has **not** been run
> on hardware here. Treat the first run as a bring-up: confirm `gpu_torch_smoke.py` reports
> `cuda_available=True` against the ARN, then run the trainer.

## Knobs (env vars; defaults in `run.sh`)

| Knob | Default | Meaning |
|---|---|---|
| `SANDBOX` | `agentcore-runtime` | GPU sandbox substrate. **`agentcore` (Code Interpreter) is CPU-only and cannot train this.** |
| `AGENTCORE_SANDBOX_RUNTIME_ARN` | *(required)* | your custom GPU Runtime ARN (g6 Instance with CUDA); hard error if unset |
| `RB` / `NS` | 4 / 8 | GRPO prompts-per-step / group size (`GBS = RB×NS`); keep `NR×RB ≤ 100` |
| `NR` | 11 | num-rollout ⇒ real steps `NR−1` = 10 |
| `CONC` | 4 | sandbox/episode concurrency; episodes in flight `S = CONC × num_engines` |
| `TEMP` | 1.0 | rollout sampling temperature |
| `EVAL_INTERVAL` / `EVAL_NS` | 10 / 2 | held-out eval cadence / samples |

## Deploying the custom GPU Runtime (the `agentcore-runtime` backend)

The image is `deploy/Dockerfile.sandbox.gpu` (`python:3.12-slim` + `bedrock-agentcore`,
`sandbox_server_agent.py`, torch cu124 installed in `CMD` — see the ~2 GB image-cap note below). It
must be created on a **GPU capacity-provider** so the Runtime lands on a g6 Instance. This repo's
tooling:

- **`code/deploy/deploy_two_runtime.py`** creates the GPU capacity provider + execution role and the
  base GPU Runtime `acs_sandbox_gpu`.
- **`code/deploy/redeploy_gpu_sandbox.py`** surgically rebuilds+redeploys **only** the GPU sandbox
  Runtime under the same name (mints a new ARN, printed at the end) — the fast path when you change
  the GPU Dockerfile.
- **`code/deploy/smoke_gpu_torch.py`**
  (`AGENTCORE_SANDBOX_GPU_RUNTIME_ARN=... python3 smoke_gpu_torch.py`) warms the cold g6 and runs
  `import torch; torch.cuda.is_available()` with a transient-5xx retry loop — a good sanity check
  before pointing this demo at the ARN.

The GPU image installs the CUDA torch stack in its `CMD` (at container startup) rather than baking it
into the image, because AgentCore caps the image at `maxImageSizeMb=2048` and CUDA torch alone is
~2.7 GB. Consequence: the first invocation of a cold g6 Runtime takes a couple of minutes while torch
installs — the harness `exec` timeout and the deploy smoke's retry loop both allow for it.

## Folder layout (self-contained)

- `run.sh` — the real KernelBench trainer on the custom GPU Runtime (this demo).
- `gpu_torch_smoke.py` — the offline CUDA-contract probe (write_file → exec → read_file of a
  `torch.cuda` check), kept for bring-up.
- `data/` — `kernelbench_train.jsonl` (100 level_1) + `kernelbench_eval.jsonl` (20 level_2).
- `deploy/Dockerfile.sandbox.gpu` — the custom GPU-Runtime image (torch cu124 installed at startup).
- `agentcore_slime_demo/` — vendored harness/sandbox/task lib (imported only from here).

## Takeaway

The managed Code Interpreter cannot give you a GPU; a custom Runtime on a g6 Instance can — through
the identical `write_file`/`exec`/`read_file` contract. That is what makes GPU RL tasks (KernelBench,
`torch.cuda` rewards) trainable on AgentCore **without changing a line of the harness or trainer** —
only the sandbox ARN points at a GPU substrate.
