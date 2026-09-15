# demo_gpu_2runtime — KernelBench CUDA task, TWO AgentCore Runtimes (harness Runtime + GPU sandbox Runtime on g6)

This demo (aka **Demo B3**) trains a Qwen3-4B policy on real
[KernelBench](https://huggingface.co/datasets/ScalingIntelligence/KernelBench) (write a `ModelNew`
that matches a reference `Model` numerically **and** compiles/runs on CUDA) with the rollout split
across **two AgentCore Runtimes**: a mini Claude-Code-like **harness Runtime #1** (arm64, managed,
VPC) whose tool calls and reward checker invoke a **remote GPU sandbox Runtime #2** (a g6
capacity-provider EC2 GPU, amd64) where the torch/CUDA checker actually runs.

## What this demo shows
- **Task:** real KernelBench CUDA-op repair — train on 100 level_1 single ops, eval on 20 held-out
  level_2 fusions. The reward checker imports torch and runs the candidate kernel on a GPU.
- **Two-Runtime split:** the agent loop (`agentcore_slime_demo.mini_agent.MiniClaudeHarness`,
  tools `write_file` / `bash` / `read_file`) lives on **Runtime #1**; every tool call *and* the
  KernelBench checker are dispatched to a **separate GPU sandbox Runtime #2** on a g6. No GPU on
  this trainer box is reserved for the checker — all 8 local GPUs go to slime (4 train + 4 infer).
- **Harness VPC dial-back:** Runtime #1 runs `networkMode=VPC` (no public port) and dials the
  slime `AnthropicAdapter` proxy on **this box's private IP** back over the VPC, gated by one narrow
  security-group ingress rule on `ADAPTER_PORT`.
- **GPU sandbox on a capacity provider:** Runtime #2 runs on a `g6.xlarge` capacity-provider
  instance. torch is **installed at container startup** (not baked, not on a persisted volume).

## AgentCore basics: launch / terminate / latency
- **Launch:** from `code/deploy/`, `export AWS_REGION=us-west-2` and
  `export ADAPTER_PUBLIC_HOST=<this box PRIVATE IP>`, then `python3 deploy_two_runtime.py all`.
  It builds all images, creates **Runtime #1** `acs_harness` (VPC) + **Runtime #2** `acs_sandbox_gpu`
  on capacity provider `acs_gpu_cp` (`g6.xlarge`), creates the private VPC endpoints the VPC-mode
  harness needs to start, opens the adapter port from the runtime security group, and runs the
  one-time `{"mode":"prepare"}` invoke. The **first** rollout on any cold g6 pays a **~90s torch
  install** (`torch==2.5.1` cu124), amortized over that instance's warm lifetime.
- **Terminate:** from `code/deploy/`, `python3 deploy_two_runtime.py cleanup` deletes `acs_harness`,
  `acs_sandbox_gpu`, and the capacity provider `acs_gpu_cp`. `run.sh` itself does **no** runtime
  teardown — remember to run cleanup so the g6 stops billing.
- **Latency (two-Runtime caveat):** because the harness runs on Runtime #1, the `[sbx-op]` timing
  records land in **Runtime #1's CloudWatch log**, not this trainer's `train.log` — download that log
  and run `metrics.py` on it for sandbox latency. There is **no `perf/sglang_gen_ms_*` split** here
  (completions are served by slime's in-tree `AnthropicAdapter`, which we do not patch). Expect a
  cold-g6 tail on the first ops (~90s torch install inherited from the GPU sandbox).

## Folder layout (self-contained)
- `agentcore_slime_demo/` — vendored shared library (this demo imports only this copy;
  `PYTHONPATH` no longer needs `/work/code`).
- `deploy/` — the two Dockerfiles: `Dockerfile.harness` (Runtime #1) + `Dockerfile.sandbox.gpu`
  (Runtime #2).
- `build_dataset.py`, `data/` (`kernelbench_train.jsonl` + `kernelbench_eval.jsonl`), `run.sh`,
  `metrics.py`.
- Shared infra referenced by path (not vendored): `code/deploy/deploy_two_runtime.py`,
  `redeploy_harness.py`, `redeploy_gpu_sandbox.py`, `_prepare_gpu_once.py`, `smoke_gpu_torch.py`.
  Base checkpoint `Qwen3-4B_torch_dist` is shared (override with `CKPT_BASE`).

## Step by step

### 0. Prereqs
- Run inside the persistent slime container `agentcore_p4` (repo mounted at `/work/code`).
- `export AWS_REGION=us-west-2` and `export AWS_BEARER_TOKEN_BEDROCK=<token>` (both Runtimes need it).
- Know **this box's private IP** (e.g. `PRIVATE-IP-f99864bc`) — the harness dials it back.
- Pick an adapter port and keep it pinned: `ADAPTER_PORT=18021` (default).

### 1. Build the dataset
```bash
cd /work/code/demos/demo_gpu_2runtime
python3 build_dataset.py --dataset kernelbench --level level_1 --limit 100 --out data/kernelbench_train.jsonl
python3 build_dataset.py --dataset kernelbench --level level_2 --limit 20  --out data/kernelbench_eval.jsonl
```

### 2. Build the two images (context = this demo dir)
Build with the **build context set to this demo dir** so `COPY agentcore_slime_demo` picks up the
**vendored** copy. Runtime #1 is arm64 (managed fleet), Runtime #2 is amd64 (g6 = x86_64):
```bash
cd /work/code/demos/demo_gpu_2runtime
# Runtime #1 — mini harness (linux/arm64), pushed as acs-harness:latest
docker buildx build --platform linux/arm64 -f deploy/Dockerfile.harness \
  -t <acct>.dkr.ecr.us-west-2.amazonaws.com/acs-harness:latest --push .
# Runtime #2 — GPU sandbox (linux/amd64), pushed as acs-sandbox-gpu:latest
docker buildx build --platform linux/amd64 -f deploy/Dockerfile.sandbox.gpu \
  -t <acct>.dkr.ecr.us-west-2.amazonaws.com/acs-sandbox-gpu:latest --push .
```
The harness image is a slim CPU Python container (aiohttp + boto3 + bedrock-agentcore, no
slime/torch) that runs `python -m agentcore_slime_demo.harness_runtime_agent`. The GPU sandbox image
stays small too — it installs `torch==2.5.1` (cu124) **at container startup** (~90s per cold g6),
because the CUDA torch stack (~2.7 GB) exceeds AgentCore's 2048 MB image quota.

> Alternatively, skip this step and let step 3 build+push: `deploy_two_runtime.py all` builds all
> images from `code/deploy/` itself. If you built here, pass `SKIP_BUILD=1` in step 3 to reuse them.

### 3. Launch both Runtimes
```bash
cd /work/code/deploy
export AWS_REGION=us-west-2
export ADAPTER_PUBLIC_HOST=PRIVATE-IP-f99864bc   # <-- this box PRIVATE IP
python3 deploy_two_runtime.py all
```
This creates `acs_harness` (VPC), `acs_sandbox_gpu` (on `acs_gpu_cp`, `g6.xlarge`), the VPC
endpoints, the adapter-port SG rule, and runs `{"mode":"prepare"}` once. It prints an **export
block** with the ARNs to copy into step 5. (`SKIP_PREPARE=1` skips the prepare invoke; run it later
with `python3 _prepare_gpu_once.py` — set `AGENTCORE_SANDBOX_GPU_RUNTIME_ARN` first.)

Iterate on a single Runtime without a full redeploy: harness → `python3 redeploy_harness.py`;
GPU sandbox → `python3 redeploy_gpu_sandbox.py` (each mints a **new** ARN — re-export it).

### 4. Smoke-test the GPU sandbox
Warms the cold g6, triggers the startup torch install, and checks `torch.cuda.is_available()`:
```bash
cd /work/code/deploy
AGENTCORE_SANDBOX_GPU_RUNTIME_ARN=<gpu ARN> python3 smoke_gpu_torch.py
```

### 5. Export the ARNs + adapter vars
Copy from the step-3 export block (B3 uses the **GPU** sandbox ARN):
```bash
export SANDBOX=agentcore-runtime AWS_REGION=us-west-2
export AGENTCORE_HARNESS_RUNTIME_ARN=arn:aws:bedrock-agentcore:us-west-2:...:runtime/acs_harness-...
export AGENTCORE_SANDBOX_RUNTIME_ARN=arn:aws:bedrock-agentcore:us-west-2:...:runtime/acs_sandbox_gpu-...
export ADAPTER_PUBLIC_HOST=PRIVATE-IP-f99864bc ADAPTER_PORT=18021
```
`run.sh` **requires**: `AGENTCORE_HARNESS_RUNTIME_ARN`, `AGENTCORE_SANDBOX_RUNTIME_ARN` (the GPU
ARN), `ADAPTER_PUBLIC_HOST` (private IP), and `ADAPTER_PORT` — plus `AWS_BEARER_TOKEN_BEDROCK`.

### 6. Run RL training
Run inside `agentcore_p4`, async/decoupled (`train_async.py`, no `--colocate`):
```bash
docker exec agentcore_p4 bash -lc 'cd /work \
  && SANDBOX=agentcore-runtime AWS_REGION=us-west-2 \
     AGENTCORE_HARNESS_RUNTIME_ARN=arn:aws:bedrock-agentcore:...:runtime/acs_harness-... \
     AGENTCORE_SANDBOX_RUNTIME_ARN=arn:aws:bedrock-agentcore:...:runtime/acs_sandbox_gpu-... \
     ADAPTER_PUBLIC_HOST=PRIVATE-IP-f99864bc ADAPTER_PORT=18021 \
     bash code/demos/demo_gpu_2runtime/run.sh'
```
Knobs (defaults from `run.sh`): `RB=4` (prompts/step), `NS=8` (GRPO group), `NR=11` (real steps =
`NR-1` = 10; keep **`NR × RB ≤ 100`** so save-only-at-end holds against the 100-problem level_1
pool), `TEMP=1.0`, `CONC=4` (rollout concurrency — keep small, each episode drives both Runtimes),
`EVAL_INTERVAL=10`, `EVAL_NS=2`, `MINI_AGENT_MAX_TURNS=3` / `EPISODE_MAX_TURNS=3`, `HARNESS_KIND=mini`.
Local GPU split is fixed at **4 train + 4 infer + 0 env** (`ACTOR_GPUS=4`, `ROLLOUT_GPUS=4`) — the
checker's GPU is remote on Runtime #2. `sglang-tool-call-parser qwen25` is required, else the
Hermes-style `<tool_call>` output is never parsed and reward stays 0.

### 7. Read metrics / latency
```bash
cd /work/code/demos/demo_gpu_2runtime
python3 metrics.py train.log --csv train.csv --eval-csv train.eval.csv --plot train.png
```
For sandbox latency (`perf/sandbox_ms_{mean,p50,p95}`), point `metrics.py` at the **downloaded
Runtime #1 CloudWatch log** where the `[sbx-op]` records land — not `train.log`. No SGLang decode
split is produced for this two-Runtime setup.

### 8. Terminate
```bash
cd /work/code/deploy
python3 deploy_two_runtime.py cleanup   # deletes acs_harness + acs_sandbox_gpu + acs_gpu_cp
```
