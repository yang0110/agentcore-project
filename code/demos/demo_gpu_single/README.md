# demo_gpu_single — KernelBench CUDA task, one GPU AgentCore sandbox on a g6 (in-process harness)

An agentic RL loop: the policy (Qwen3-4B) writes a CUDA `ModelNew`, a torch/CUDA checker runs it, the policy reads the failure and fixes it — GRPO rewards the fixes. The `run_episode` harness drives the multi-turn loop **in-process on this box**, but the checker's torch/CUDA runs **remotely** in a single GPU AgentCore Runtime backed by a `g6` capacity-provider instance.

## What this demo shows
- **Task:** the real **KernelBench** (`ScalingIntelligence/KernelBench`, 270 CUDA problems). Each problem gives a reference `Model` (nn.Module) + `get_inputs()`/`get_init_inputs()`; the policy must write an equivalent `ModelNew` that runs on CUDA and matches the reference.
- **Reward = GPU checker:** `python3 run_checks.py` rebuilds the reference on the GPU with matched seeds, asserts `allclose` on a `cuda` device, and reports a speedup. Pass → reward; failure text is fed back for the next turn.
- **Sandbox = one GPU AgentCore Runtime:** `sandbox_server_agent.py` on a `g6.xlarge` capacity-provider instance. `write_file` / `exec` / `read_file` — including the checker — run **there**, reached over `invoke_agent_runtime` (`SANDBOX=agentcore-runtime` → `AgentCoreRuntimeSandbox`). No local GPU is reserved for the checker.
- **In-process harness:** `generate.generate` drives SGLang directly on this box; only the sandbox is remote (no MCP / two-Runtime harness). GPU split on this 8×L40S box: `ACTOR_GPUS=4` train + `ROLLOUT_GPUS=4` inference + 0 env.

## AgentCore basics: launch / terminate / latency
- **Launch:** the sandbox is a GPU Runtime on the `acs_gpu_cp` capacity provider (a real `g6.xlarge`). **torch is installed at container startup**, not baked into the image and not on a persisted volume: the Dockerfile's `CMD` runs `pip install torch==2.5.1 numpy --index-url .../cu124` (~90s per cold g6) before the server binds. This dodges the 2048 MB `maxImageSizeMb` quota (CUDA torch is ~2.7 GB) and the fact that the capacity-provider EBS `models` volume is reprovisioned empty on every instance recycle — so `mode=prepare`/smoke now mainly **warm** the instance rather than persist anything.
- **Terminate:** `deploy_two_runtime.py cleanup` deletes the `acs_sandbox_gpu` Runtime **and** the `acs_gpu_cp` capacity provider. `run.sh` itself does no runtime teardown (it only `pkill`s local sglang / `ray stop`).
- **Latency:** multi-turn agentic rollout latency is long and variable (kernel repair on a GPU). Expect a cold g6 launch + the ~90s torch install on each cold instance; `mode=prepare` widens the invoke read timeout to 1200s so the cold launch doesn't time out. Per-step latency lands in the train CSV as `perf/sandbox_ms_{mean,p50,p95}` and `perf/sglang_gen_ms_{mean,p50,p95}` (via `metrics.py`).

## Folder layout (self-contained)
- `agentcore_slime_demo/` — **vendored** shared library (imports resolve here; `PYTHONPATH` is `${DEMO_DIR}`, no longer `/work/code`)
- `deploy/Dockerfile.sandbox.gpu` — the GPU sandbox image (ECR repo `acs-sandbox-gpu`; `COPY agentcore_slime_demo/sandbox_server_agent.py`)
- `build_dataset.py`, `data/` — KernelBench dataset (train `level_1` / eval `level_2`)
- `run.sh`, `metrics.py` — the training runner and the log→CSV/PNG parser

Not vendored (shared, referenced by path): `code/deploy/deploy_two_runtime.py`, `code/deploy/redeploy_gpu_sandbox.py`, `code/deploy/_prepare_gpu_once.py`, `code/deploy/smoke_gpu_torch.py`, and the base Megatron checkpoint (`/work/code/p4_slime_train/ckpt/Qwen3-4B_torch_dist`).

## Step by step

### 0. Prereqs
Run everything **inside** the persistent container `agentcore_p4` (repo at `/work/code`, slime at `/root/slime`, has `datasets` + HF cache). Set `AWS_REGION=us-west-2`, and `AWS_BEARER_TOKEN_BEDROCK` (for `invoke_agent_runtime`). The `deploy/` scripts reach AgentCore via IAM.

### 1. Build the dataset
Train on `level_1` (100 single-op problems); hold out 20 `level_2` fusions for eval (disjoint — no leakage):
```bash
docker exec agentcore_p4 python /work/code/demos/demo_gpu_single/build_dataset.py \
    --dataset kernelbench --level level_1 --limit 100 \
    --out /work/code/demos/demo_gpu_single/data/kernelbench_train.jsonl
docker exec agentcore_p4 python /work/code/demos/demo_gpu_single/build_dataset.py \
    --dataset kernelbench --level level_2 --limit 20 \
    --out /work/code/demos/demo_gpu_single/data/kernelbench_eval.jsonl
```
(`--dataset kernelbench-stub` builds the hand-written 3-op smoke set for a minimal offline check.)

### 2. Build the GPU sandbox image
Build with the **context = this demo dir** (so `COPY agentcore_slime_demo/...` resolves against the vendored copy), for `linux/amd64`:
```bash
cd /work/code/demos/demo_gpu_single
docker build --platform linux/amd64 \
    -f deploy/Dockerfile.sandbox.gpu \
    -t <ecr>/acs-sandbox-gpu:latest .
# then docker push <ecr>/acs-sandbox-gpu:latest
```

### 3. Launch the Runtime (g6 capacity provider)
From `code/deploy/`, create the capacity provider `acs_gpu_cp` + the `acs_sandbox_gpu` Runtime and run the one-time prepare invoke:
```bash
cd /work/code/deploy
python3 deploy_two_runtime.py all        # creates acs_gpu_cp + acs_sandbox_gpu, then mode=prepare
# SKIP_PREPARE=1 python3 deploy_two_runtime.py all   # skip the cold-launch warm-up
```
`all` runs the one-time `{"mode":"prepare"}` invoke (`prepare_gpu_volume`, read timeout widened to 1200s). To redeploy **just** the GPU image after a rebuild:
```bash
python3 redeploy_gpu_sandbox.py          # prints: OUT AGENTCORE_SANDBOX_GPU_RUNTIME_ARN=arn:aws:bedrock-agentcore:...
```
Standalone prepare (warm a cold instance) if needed:
```bash
GPU_SANDBOX_ARN=arn:aws:bedrock-agentcore:...:runtime/acs_sandbox_gpu-... python3 _prepare_gpu_once.py
```

### 4. Smoke-test the GPU sandbox
Confirm the remote g6 can `import torch` on CUDA (retries transient 5xx during cold launch):
```bash
AGENTCORE_SANDBOX_GPU_RUNTIME_ARN=arn:aws:bedrock-agentcore:...:runtime/acs_sandbox_gpu-... \
    python3 smoke_gpu_torch.py
```

### 5. Export the ARN
`run.sh` needs only the sandbox ARN (`SANDBOX` defaults to `agentcore-runtime`):
```bash
export AGENTCORE_SANDBOX_RUNTIME_ARN=arn:aws:bedrock-agentcore:...:runtime/acs_sandbox_gpu-...
```

### 6. Run RL training
From inside the container, run the standalone runner (REAL async slime training, `train_async.py`, no `--colocate`):
```bash
docker exec agentcore_p4 bash -lc \
  'cd /work && AWS_REGION=us-west-2 \
     AGENTCORE_SANDBOX_RUNTIME_ARN='"$AGENTCORE_SANDBOX_RUNTIME_ARN"' \
     NR=11 EVAL_INTERVAL=10 \
     bash code/demos/demo_gpu_single/run.sh'
```
Key knobs (defaults): `RB=4` (prompts/step), `NS=8` (GRPO group size), `NR=11` (real steps = `NR-1` = 10; **keep `NR × RB ≤ 100`** so one epoch spans the run and the only checkpoint is the final one), `TEMP=1.0`, `CONC=4` (sglang concurrency), `EVAL_INTERVAL=10`, `EVAL_NS=2`, `SAVE`, `CKPT_BASE`. `SAVE_INTERVAL` is huge (1000000): no mid-training snapshots; slime force-saves the final rollout under `--save`. The end-of-training save deadlock is fixed via `--async-save` + `--custom-megatron-init-path agentcore_slime_demo.megatron_init.enable_persistent_async_ckpt`.

**One-command train-to-convergence.** `train.sh` wraps this with top-level knobs — training
**steps**, eval **cadence**, sandbox **concurrency** — and passes `LOCAL_GPU`/`AGENTCORE_SANDBOX_RUNTIME_ARN`
through: `STEPS` → `NR=STEPS+1` (default 20; **keep `STEPS × RB ≤ 100`** for a single-epoch run),
`EVAL_EVERY` → `EVAL_INTERVAL`, `SANDBOX_CONC` → `CONC`. (There's no `K` knob here: the GPU sandbox is
a single g6 Runtime.)
```bash
docker exec agentcore_p4 bash -lc \
  'cd /work && AGENTCORE_SANDBOX_RUNTIME_ARN='"$AGENTCORE_SANDBOX_RUNTIME_ARN"' \
     STEPS=20 EVAL_EVERY=10 SANDBOX_CONC=4 \
     bash code/demos/demo_gpu_single/train.sh'
```

### 7. Read metrics / latency
Parse the tee'd training log into CSVs + a PNG:
```bash
python3 code/demos/demo_gpu_single/metrics.py <train.log> \
    --csv train.csv --eval-csv train.eval.csv --plot train.png
```
Per-step train CSV: `rollout/raw_reward` (the RL signal), `train/grad_norm`, `train/kl_loss`, `perf/step_time`, `perf/mfu`, `perf/wait_time_ratio`, and the latency columns `perf/sandbox_ms_{mean,p50,p95}` + `perf/sglang_gen_ms_{mean,p50,p95}`. The eval CSV is keyed by `eval/step` (held-out solve rate at each `EVAL_INTERVAL`).

### 8. Terminate
Tear the whole GPU stack back down (this is the only real cost — a live g6.xlarge + EBS):
```bash
cd /work/code/deploy
python3 deploy_two_runtime.py cleanup    # deletes acs_sandbox_gpu AND the acs_gpu_cp capacity provider
```
