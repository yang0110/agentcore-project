# demo_gpu_shared — KernelBench CUDA task, ONE g6 shared by a Ray fractional-GPU farm inside the sandbox Runtime

Multiple concurrent KernelBench episodes share ONE remote g6 GPU: they all invoke a single AgentCore
sandbox Runtime under one shared `runtimeSessionId`, and a Ray fractional-GPU farm running *inside*
that g6 leases one fractional-GPU actor per episode. One L4 + `B4_GPU_FRACTION=0.25` → 4 concurrent
sandboxes on that single GPU.

## What this demo shows
- **Task:** real [KernelBench](https://huggingface.co/datasets/ScalingIntelligence/KernelBench) —
  write a `ModelNew` that matches a reference `Model` numerically **and** runs on CUDA. The
  `run_episode` harness runs **in-process** (drives SGLang directly); the only remote piece is the
  shared-GPU sandbox Runtime.
- **GPU sharing via Ray fractional actors:** `ray.remote(num_gpus=0.25)(SandboxActor)` packs
  `floor(1/fraction)` long-lived, warm-CUDA actors onto the g6's one GPU. Ray does the packing; each
  actor caps its own GPU memory in software (`B4_PER_SANDBOX_MEM_FRACTION`, via an injected
  `sitecustomize.py`) because L4/L40S have no MIG.
- **One shared `runtimeSessionId` + per-episode `sub_session`:** AgentCore maps
  1 `runtimeSessionId` → 1 instance, so every episode across every rollout worker invokes the g6
  under the SAME session id (they land on the same g6) and carries a per-episode `sub_session` that
  keys its fractional actor.

## AgentCore basics: launch / terminate / latency
- **Launch:** this demo rides on the exec role + operator role + GPU capacity provider `acs_gpu_cp`
  that `deploy_two_runtime.py all` creates. Run `deploy_two_runtime.py all` **first** (once), then from
  `code/deploy/` run `python3 redeploy_gpu_sandbox_shared.py` (build+push `acs-sandbox-gpu-b4` +
  create the `acs_sandbox_gpu_b4` runtime), or `SKIP_BUILD=1 python3 redeploy_gpu_sandbox_shared.py`
  to reuse the already-pushed image. Farm knobs are injected as **runtime env** (retunable WITHOUT a
  rebuild): `B4_GPU_FRACTION` (0.25), `B4_FARM_GPUS` (1, **MUST be set** — the slim image has no
  nvidia-smi for Ray to autodetect the L4), `B4_PER_SANDBOX_MEM_FRACTION` (0.25), `B4_WORKSPACE_ROOT`
  (`/workspace/b4-sbx`), `B4_WARM_CUDA` (1), optional `B4_MAX_SANDBOXES` (unset ⇒ `floor(gpus/frac)`).
  The script prints `OUT AGENTCORE_SANDBOX_GPU_B4_RUNTIME_ARN=…` and `export AGENTCORE_SANDBOX_RUNTIME_ARN=<arn>`.
- **Terminate:** delete `acs_sandbox_gpu_b4` **directly** — rerun the delete block in
  `redeploy_gpu_sandbox_shared.py` (control-plane `delete_agent_runtime`). **Caveat:**
  `deploy_two_runtime.py cleanup` does NOT delete `acs_sandbox_gpu_b4` (it only removes
  `harness`/`sandbox_cpu`/`sandbox_gpu`) — but it WILL delete the shared capacity provider `acs_gpu_cp`
  that b4 rides on, so tear b4 down before (or independently of) that cleanup.
- **Latency:** the first invoke cold-launches the g6 + ~90s torch install + `ray.init` + CUDA warm —
  **minutes, well past botocore's 60s default** (the client uses `read_timeout=900`). So run.sh sets a
  high AgentCore-invoke retry budget: the default 8 attempts (~150s) is too short and would kill the
  job on a single transient blip; b4 uses `INVOKE_MAX_RETRIES=20` (~10 min patience). Sharing hinges
  on AgentCore dispatching concurrent invokes under ONE `runtimeSessionId` onto the one g6
  concurrently; if it serializes per-session, episodes still run correctly but one-at-a-time. Non-MIG
  L4 can wedge under sustained nvcc load — raise `B4_GPU_FRACTION` toward 0.5 to mitigate.

## Folder layout (self-contained)
- `agentcore_slime_demo/` — vendored shared library (this demo imports ONLY this copy;
  `/work/code` is no longer on `PYTHONPATH`). Includes `sandbox_ray_farm.py`,
  `sandbox_server_agent.py`, `sandbox.py`.
- `deploy/Dockerfile.sandbox.gpu.shared` — the dedicated shared-GPU sandbox image (bakes `ray`, ships
  the package files with an empty `__init__`, runs `python -m agentcore_slime_demo.sandbox_server_agent`).
- `build_dataset.py`, `data/`, `run.sh`, `metrics.py`.

Shared infra referenced by path (NOT vendored): `code/deploy/deploy_two_runtime.py` (roles +
`acs_gpu_cp`), `code/deploy/redeploy_gpu_sandbox_shared.py`, `code/deploy/smoke_gpu_shared_g6.py`.
Base checkpoint is shared.

## Step by step

### 0. Prereqs
Work inside the persistent slime container `agentcore_p4`; region `us-west-2` (scripts
`setdefault("AWS_REGION", "us-west-2")`). Run **`python3 deploy_two_runtime.py all` once** to create
the exec/operator roles + GPU capacity provider `acs_gpu_cp` that this demo reuses.

### 1. Build the dataset
Already shipped in `data/` (identical KernelBench slices). To rebuild:
```bash
python3 build_dataset.py --dataset kernelbench --level level_1 --limit 100 --out data/kernelbench_train.jsonl
python3 build_dataset.py --dataset kernelbench --level level_2 --limit 20  --out data/kernelbench_eval.jsonl
```

### 2. Build the shared-GPU sandbox image
`redeploy_gpu_sandbox_shared.py` builds this for you (step 3). To build by hand, use **this demo dir
as the build context** (so the `COPY agentcore_slime_demo/…` lines resolve against the vendored copy)
and target **linux/amd64** (the g6 is x86_64):
```bash
docker build --platform linux/amd64 \
  -f deploy/Dockerfile.sandbox.gpu.shared \
  -t <ecr>/acs-sandbox-gpu-b4:latest .
```
> **File-name drift to ignore:** the actual dedicated Dockerfile is
> `deploy/Dockerfile.sandbox.gpu.shared`. Its own header comment still says
> `Dockerfile.sandbox.gpu.b4` and refers to the b2/b3 base as `Dockerfile.sandbox.gpu` — those names
> are stale; build the `.shared` file with the context above.

### 3. Launch the shared Runtime
From `code/deploy/`:
```bash
python3 redeploy_gpu_sandbox_shared.py            # build+push acs-sandbox-gpu-b4 + create acs_sandbox_gpu_b4
# or, to reuse the already-pushed image:
SKIP_BUILD=1 python3 redeploy_gpu_sandbox_shared.py
```
Override any farm knob inline to retune without a rebuild, e.g. `B4_GPU_FRACTION=0.5 B4_FARM_GPUS=1 python3 redeploy_gpu_sandbox_shared.py`.

### 4. Smoke-test the farm
Validates package import, `ray.init`, fractional actors, and concurrent `sub_session`s coexisting on
one g6 — before spending a slime run:
```bash
AGENTCORE_SANDBOX_RUNTIME_ARN=<b4 arn> python3 smoke_gpu_shared_g6.py --episodes 4
```

### 5. Export the ARN
Copy the printed line:
```bash
export AGENTCORE_SANDBOX_RUNTIME_ARN=<b4 arn>
```
run.sh selects the shared-GPU backend with `SANDBOX=agentcore-ray-farm` (its default) and needs only
`AGENTCORE_SANDBOX_RUNTIME_ARN` — the client just picks the shared session + per-episode sub_session;
the `B4_*` farm knobs live on the g6 image, not here.

### 6. Run RL training
Launch inside the slime container with the ARN exported:
```bash
docker exec agentcore_p4 bash -lc 'cd /work \
  && AWS_REGION=us-west-2 \
     AGENTCORE_SANDBOX_RUNTIME_ARN=<b4 arn> \
     bash code/demos/demo_gpu_shared/run.sh'
```
Knobs (defaults in run.sh): `RB` (4), `NS` (8), `NR` (11 ⇒ real steps = NR−1 = 10 — keep
**`NR × RB ≤ 100`**), `TEMP` (1.0), `CONC` (4 ⇒ `--sglang-server-concurrency`; keep
**`CONC ≤ B4_MAX_SANDBOXES`** so episodes don't block on the farm's checkout gate), `EVAL_INTERVAL`
(10), `EVAL_NS` (2), `INVOKE_MAX_RETRIES` (**20** ≈ 10 min patience — do not lower), and optionally
`B4_SHARED_RUNTIME_SESSION` (pin the shared session id to reuse a warm g6 across runs; auto-minted as
`b4farm-<32 hex>` if unset). All 8 local GPUs go to slime (`ACTOR_GPUS=4` + `ROLLOUT_GPUS=4`); the
shared GPU is the remote g6's.

**One-command train-to-convergence.** `train.sh` wraps this with top-level knobs — training
**steps**, eval **cadence**, sandbox **concurrency** — passing `AGENTCORE_SANDBOX_RUNTIME_ARN`
through: `STEPS` → `NR=STEPS+1` (default 20; **keep `STEPS × RB ≤ 100`**), `EVAL_EVERY` →
`EVAL_INTERVAL`, `SANDBOX_CONC` → `CONC` (keep `SANDBOX_CONC ≤ B4_MAX_SANDBOXES`). (No `K` knob: the
farm packs many envs onto the single g6's GPU, not across multiple Runtimes.)
```bash
docker exec agentcore_p4 bash -lc 'cd /work \
  && AGENTCORE_SANDBOX_RUNTIME_ARN=<b4 arn> STEPS=20 EVAL_EVERY=10 SANDBOX_CONC=4 \
     bash code/demos/demo_gpu_shared/train.sh'
```

### 7. Read metrics / latency
```bash
python3 metrics.py train.log --csv train.csv --eval-csv train.eval.csv --plot train.png
```
Watch that concurrent episodes actually overlap on the one g6 (per-episode `sub_session`s active at
once) — that is the sharing this demo is proving.

### 8. Terminate
Delete `acs_sandbox_gpu_b4` **directly** (rerun the delete block in
`redeploy_gpu_sandbox_shared.py`, or call control-plane `delete_agent_runtime`). Remember:
`deploy_two_runtime.py cleanup` will NOT remove `acs_sandbox_gpu_b4`, but WILL delete the shared
capacity provider `acs_gpu_cp` — so terminate b4 first.
