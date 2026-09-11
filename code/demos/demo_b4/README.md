# Demo B4 — KernelBench (GPU) with many sandboxes sharing ONE remote g6 GPU via Ray fractions

**Demo B4 = Demo B3's task + mini harness, but a different sandbox-GPU story: pack SEVERAL
concurrent KernelBench sandboxes onto ONE remote g6 instance and let Ray split that g6's single GPU
between them.** Same task (real [KernelBench](https://huggingface.co/datasets/ScalingIntelligence/KernelBench):
write a `ModelNew` that matches a reference `Model` numerically **and** runs on CUDA), same
[`MiniClaudeHarness`](../../agentcore_slime_demo/mini_agent.py), same slime async/decoupled training,
same **Qwen3-4B policy**. The one thing that changes is **how the checker's remote GPU is shared**.

## Why B4 exists (the problem with B3's sandbox)

[Demo B3](../demo_b3/) makes the sandbox its own **AgentCore Runtime #2** (a remote g6), and
AgentCore maps **1 `runtimeSessionId` → 1 isolated instance**. That is clean isolation but it does
**not share a GPU**: every concurrent rollout wants its own g6 instance (a **g6 per session**), so
*N* in-flight episodes demand *N* g6 instances. On a capacity-starved g6 fleet that is exactly what
stalled B3 (see the project's `agentcore-gpu-instances-sharing` note: GPU is only on
capacity-provider instances, and AgentCore exposes **no** cross-session GPU-sharing config).

B4 keeps the sandbox on a **remote g6 AgentCore Runtime #2** (like all of series B — "all of series
B uses the remote g6 instance"), but instead of one-session-per-g6 it packs **several concurrent
sandboxes onto ONE g6 instance** and multiplexes that g6's single GPU with **Ray fractional GPU
allocation running inside the g6**.

## How B4 shares the remote g6 GPU

AgentCore maps **1 `runtimeSessionId` → 1 instance**, so to make N concurrent episodes share ONE g6
they all invoke it under the **same `runtimeSessionId`** (`B4_SHARED_RUNTIME_SESSION` — one per
rollout process) and carry a **per-episode `sub_session`** in every op payload. Inside that one g6, a
Ray farm leases one **fractional-GPU actor** per `sub_session`:

```python
ray.remote(num_gpus=0.5)(SandboxActor)   # 2 of these fit on the g6's one L4 (frac=0.5)
```

Ray schedules `floor(1/fraction)` sandbox actors onto the g6's GPU and sets each actor's
`CUDA_VISIBLE_DEVICES` accordingly. **One g6 with one L4 + `frac=0.25` → 4 concurrent KernelBench
sandboxes on that single GPU.** The moving parts:

- [`sandbox_ray_farm.py`](../../agentcore_slime_demo/sandbox_ray_farm.py) — **runs INSIDE the g6.**
  `RayGpuFarm` is a fixed pool of fractional-GPU Ray actors + a checkout gate, driven by
  [`sandbox_server_agent.py`](../../agentcore_slime_demo/sandbox_server_agent.py) (`open`/`close`/
  `exec`/`write_file`/`read_file`, keyed by `sub_session`). Each actor is **long-lived** and holds a
  **warm CUDA context**, reused across episodes — the *pool of long-lived sessions* design.
- [`agentcore_runtime_sandbox.py`](../../agentcore_slime_demo/agentcore_runtime_sandbox.py) —
  `RayFarmRuntimeSandbox`, the **client** half: a drop-in `Sandbox` that turns each harness op into an
  `invoke_agent_runtime` under the shared `runtimeSessionId` + this episode's `sub_session`. The
  harness never knows it isn't a dedicated microVM.
- [`ray_farm_generate.py`](../../agentcore_slime_demo/ray_farm_generate.py) — the B4 slime hook. It
  reuses B3's mini harness and KernelBench reward verbatim; only the sandbox backend differs.

One image serves **all of series B**: a request **without** a `sub_session` takes the legacy B2/B3
single-session path; a request **with** one takes the farm path. B4 just also needs `ray` installed
on the g6 image.

### Ray's fraction is scheduling-only — so we add a software memory cap

`num_gpus=0.25` tells Ray *"at most 4 of these per GPU"* and nothing more: it enforces **no** memory
or compute limit, and L4 / L40S (Ada Lovelace) have **no MIG** to fall back on, so the co-resident
sandboxes would otherwise race for the whole device and one runaway kernel could OOM its neighbours.
So each sandbox **caps its own GPU memory in software** (`torch.cuda.set_per_process_memory_fraction`
with `B4_PER_SANDBOX_MEM_FRACTION`). Because the checker runs as a *subprocess* (`python3
run_checks.py`), the cap is applied by a tiny **`sitecustomize.py`** the actor drops into the
sandbox working dir (Python auto-imports `sitecustomize` from the script's own directory), so it
takes effect **inside the checker process**. **Ray does the packing; the sitecustomize hook does the
isolation.**

## B4 vs the other demos

|                | sandbox backend                          | GPU for the checker                | AgentCore runtimes |
|----------------|------------------------------------------|------------------------------------|--------------------|
| **Demo B**     | remote AgentCore Runtime #2 (g6)         | 1 instance/session (no sharing)    | 1 (sandbox)        |
| **Demo B3**    | remote AgentCore Runtime #2 (g6)         | 1 instance/session (no sharing)    | 2 (harness+sandbox)|
| **Demo B4**    | **remote g6 Runtime #2 + Ray farm inside** | **N sandboxes share the g6's 1 GPU** | **1 (sandbox)**  |

Like B3, **this box gives ALL 8 of its GPUs to slime** (`ACTOR_GPUS=4` train + `ROLLOUT_GPUS=4`
infer) — the shared GPU is the **remote g6's**, not a local one. B4's simplification over B3 is that
the mini harness runs **in-process** (it dials the `AnthropicAdapter` proxy over `localhost`, so no
VPC dial-back, no `ADAPTER_PUBLIC_HOST`, no harness Runtime #1). So B4 has exactly **one** remote
piece — the shared-GPU sandbox Runtime #2 — and adds exactly **one** thing over Demo B: the Ray
GPU-sharing farm inside that g6.

## GPU split on this box

Identical to B2/B3 — all 8 to slime, none held back (the checker's GPU is remote):

```
4 train (Megatron)  +  4 inference (SGLang)   →  slime gets all 8 GPUs
```

The g6-side sharing is tuned entirely by the **`B4_*` env vars baked into the g6 sandbox image**
(not set here): at `B4_GPU_FRACTION=0.5` on one L4 → `floor(1 / 0.5) = 2` concurrent sandboxes; drop
to `0.25` for **4** on the same GPU (and raise `CONC` to match).

## Offline smoke (no slime, no AWS; proves the GPU sharing)

The offline smoke exercises the **same** server-side `RayGpuFarm` on a **local** Ray cluster (in
place of the g6). Fan out N concurrent sandboxes sharing one GPU and watch them coexist:

```bash
PYTHONPATH=. python3 ../../scripts/ray_farm_smoke.py --episodes 8 --gpu-fraction 0.25 --farm-gpus 1
```

## Build the datasets

Already shipped in `data/` (copied from B3 — identical KernelBench slices). To rebuild:

```bash
python3 build_dataset.py --dataset kernelbench --level level_1 --limit 100 --out data/kernelbench_train.jsonl
python3 build_dataset.py --dataset kernelbench --level level_2 --limit 20  --out data/kernelbench_eval.jsonl
```

## Deploy the shared-GPU sandbox (required before a real run)

Build the g6 sandbox image with **`ray` installed** and the farm config baked as ENV
(`B4_GPU_FRACTION`, `B4_MAX_SANDBOXES`, `B4_PER_SANDBOX_MEM_FRACTION`, `B4_FARM_GPUS`,
`B4_WORKSPACE_ROOT`, plus `SANDBOX_PYLIB`/`LD_LIBRARY_PATH=/usr/lib64` so the fractional actors'
volume-installed torch resolves), place it on a **GPU capacity provider**, and export its ARN as
`AGENTCORE_SANDBOX_RUNTIME_ARN`. (Base is [`code/deploy/Dockerfile.sandbox.gpu`](../../deploy/Dockerfile.sandbox.gpu)
— the CUDA torch stack; B4 adds `ray` + the `B4_*` ENV on top.)

## Run

Only the sandbox Runtime is remote; the harness is in-process. Launch inside the shared slime
container with the sandbox ARN exported:

```bash
docker exec agentcore_p4 bash -lc 'cd /work \
  && AWS_REGION=us-west-2 \
     AGENTCORE_SANDBOX_RUNTIME_ARN=<b4 GPU-shared sandbox Runtime ARN> \
     bash code/demos/demo_b4/run.sh'
```

Knobs: `RB`, `NS`, `NR` (real steps = NR−1 — keep **`NR × RB ≤ 100`**), `TEMP`, `CONC` (keep
**`CONC ≤ B4_MAX_SANDBOXES`** so episodes don't block on the farm's checkout gate), `EVAL_INTERVAL`,
`EVAL_NS`, `EPISODE_MAX_TURNS`, `INVOKE_MAX_RETRIES`, and optionally `B4_SHARED_RUNTIME_SESSION` (pin
the shared session id). The farm knobs (`B4_GPU_FRACTION`, `B4_FARM_GPUS`, `B4_MAX_SANDBOXES`,
`B4_PER_SANDBOX_MEM_FRACTION`, `B4_WORKSPACE_ROOT`) live on the **g6 image**, not here — the client
only picks the shared session + per-episode sub_session.

> **Open assumption to validate on the first real run:** the GPU sharing needs AgentCore to dispatch
> **concurrent** invokes under one `runtimeSessionId` onto the one g6 instance concurrently. If it
> serializes per-session instead, episodes still run correctly but one-at-a-time (no sharing).

## Metrics

```bash
python3 metrics.py train_b4.log --csv train_b4.csv --eval-csv train_b4.eval.csv --plot train_b4.png
```
