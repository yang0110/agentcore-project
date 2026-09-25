# demo_gpu_farm — KernelBench on a LOCAL Ray-farm GPU sandbox (`FARM=fractional | whole`)

One folder for the Ray-farm GPU demos. A Ray farm inside a LOCAL AgentCore sandbox Runtime leases one
long-lived, warm-CUDA actor per concurrent KernelBench episode; **`FARM=` picks how it partitions its
GPUs**. It replaces three former folders that ran the same harness, trainer and farm and differed only
in these knobs:

| `FARM=` | farm GPUs | farm knobs | slots | replaces |
|---|---|---|---|---|
| **`fractional`** (default) | GPU 6 | `B4_FARM_GPUS=1`, `B4_GPU_FRACTION=0.5`, `B4_PER_SANDBOX_MEM_FRACTION=0.5` | `floor(1/0.5)=2` actors **sharing one GPU** (0.5 + 0.5) | `demo_gpu_shared_single` |
| `whole` | GPUs 6,7 | `B4_FARM_GPUS=2`, `B4_GPU_FRACTION=1.0`, `B4_PER_SANDBOX_MEM_FRACTION=0.95` | `floor(2/1.0)=2` actors, **one whole GPU each** | `demo_gpu_shared_multi`, `demo_gpu_local` (identical config) |

`fractional` is the real GPU-**sharing** case: each actor caps itself to ~half the card's VRAM in
software (an injected `sitecustomize.py`), because L40S has no MIG. `whole` needs no memory guard,
which sidesteps the non-MIG nvcc-wedge that fractional sharing can hit under sustained compile load.

## What this demo shows
- **Task:** real [KernelBench](https://huggingface.co/datasets/ScalingIntelligence/KernelBench) —
  write a `ModelNew` that matches a reference `Model` numerically **and** runs on CUDA. The
  `run_episode` harness runs **in-process** (drives SGLang directly); the only "remote" piece is the
  local sandbox Runtime container, reached over `http://127.0.0.1:8080`.
- **GPU packing via Ray:** `ray.remote(num_gpus=B4_GPU_FRACTION)(SandboxActor)` — Ray places the
  actors (two on GPU 6 at 0.5, or one per GPU at 1.0) and pins each via `CUDA_VISIBLE_DEVICES`.
- **One shared `runtimeSessionId` + per-episode `sub_session`:** AgentCore maps
  **1 `runtimeSessionId` → 1 instance**, so this is **not** "2 AgentCore sessions on one box." Every
  episode invokes the one local runtime under the SAME session id (`B4_SHARED_RUNTIME_SESSION`) and
  carries a per-episode `sub_session` that keys its GPU actor.

## GPU split
- **slime:** GPUs 0–5 — `ACTOR_GPUS=4` (train) + `ROLLOUT_GPUS=2` (infer) = 6. `CUDA_VISIBLE_DEVICES`
  is pinned to `0,1,2,3,4,5` **before** `ray start` so slime never touches the farm's GPUs.
- **local farm:** GPU 6 (`fractional`; GPU 7 idle) or GPUs 6,7 (`whole`), held by the sandbox runtime
  container (`docker run` on the host).

## Relationship to the other GPU demos
| demo | sandbox | GPU story |
|---|---|---|
| `demo_gpu_single` | one dedicated **remote g6** (`SANDBOX=agentcore-runtime`) | 1 episode ↔ 1 whole remote GPU |
| **`demo_gpu_farm FARM=fractional`** | one **local** runtime + Ray **fractional** farm (`agentcore-ray-farm`) | 2 episodes share ONE GPU |
| **`demo_gpu_farm FARM=whole`** | one **local** runtime + Ray **whole-GPU** farm (`agentcore-ray-farm`) | N episodes ↔ N whole local GPUs |
| `demo_gpu_snapshot`, `demo_gpu_customruntime` | the `whole` farm (with `LOCAL_GPU=1`) or a remote g6 | + snapshot/resume, + custom Runtime |

## Step by step

### 0. Prereqs
The trainer runs inside the persistent slime container `agentcore_p4` (repo at `/work/code`, slime at
`/root/slime`); region `us-west-2`. The **local runtime container** is launched on the **HOST**
(needs docker + the nvidia container runtime), not inside `agentcore_p4`. No AWS control-plane calls
and no capacity provider are needed — everything is on-box.

### 1. Build the dataset
Already shipped in `data/` (train = 100 KernelBench level_1, eval = 20 level_2). To rebuild:
```bash
python3 build_dataset.py --dataset kernelbench --level level_1 --limit 100 --out data/kernelbench_train.jsonl
python3 build_dataset.py --dataset kernelbench --level level_2 --limit 20  --out data/kernelbench_eval.jsonl
```

### 2. Start the local farm (on the HOST) — same mode as `FARM`
This builds the shared-GPU sandbox image (context = this demo dir, so the `COPY agentcore_slime_demo/…`
lines resolve against the vendored copy), reserves the farm GPU(s), and serves the farm at
`http://127.0.0.1:8080` with that mode's knobs baked in:
```bash
bash code/deploy/run_local_gpu_runtime.sh fractional   # GPU 6, 0.5 + 0.5
bash code/deploy/run_local_gpu_runtime.sh whole        # GPUs 6,7, one GPU each
# reuse an already-built image:
SKIP_BUILD=1 bash code/deploy/run_local_gpu_runtime.sh whole
```
(The old mode names still work: `shared_single` = `fractional`; `shared_multi` / `local` = `whole`.)
The image pip-installs the CUDA torch stack on first boot (kept out of the image for the 2 GB
AgentCore cap), so the first `/ping` can take a couple of minutes; the launcher polls until healthy.

### 3. Run RL training (inside agentcore_p4)
```bash
docker exec agentcore_p4 bash -lc 'cd /work && AWS_REGION=us-west-2 FARM=fractional \
    bash code/demos/demo_gpu_farm/run.sh'
```
Knobs (defaults in `run.sh`): `FARM` (`fractional`), `RB` (4), `NS` (8), `NR` (11 ⇒ real steps =
NR−1 = 10 — keep **`NR × RB ≤ 100`**), `TEMP` (1.0), `CONC` (2 ⇒ `--sglang-server-concurrency`; keep
**`CONC ≤ B4_MAX_SANDBOXES = 2`** so episodes don't block on the farm's checkout gate), `EVAL_INTERVAL`
(10), `EVAL_NS` (2), `INVOKE_MAX_RETRIES` (**20** ≈ 10 min patience — do not lower), and optionally
`B4_SHARED_RUNTIME_SESSION` (pin the shared session id to reuse a warm farm across runs; auto-minted
as `b4farm-<32 hex>` if unset). Checkpoints go to `ckpt/Qwen3-4B_kernelbench_farm_<FARM>_async`.

> **Keep the client OPEN gate == the server slot count.** `B4_FARM_MAX_CONCURRENT` (client) and
> `B4_MAX_SANDBOXES = floor(B4_FARM_GPUS / B4_GPU_FRACTION)` (server) must agree — both are 2 under
> either `FARM` default. If you retune one, retune the other.

### 4. Scale the farm
Reserve more GPUs for the farm and raise `B4_FARM_GPUS` to match. E.g. a 3-GPU whole farm (5,6,7),
slime on 5:
```bash
# HOST:
LOCAL_RT_GPUS=5,6,7 B4_FARM_GPUS=3 bash code/deploy/run_local_gpu_runtime.sh whole
# agentcore_p4:
FARM=whole SLIME_CUDA_VISIBLE=0,1,2,3,4 SLIME_NUM_GPUS=5 ROLLOUT_GPUS=1 \
  B4_FARM_GPUS=3 CONC=3 bash code/demos/demo_gpu_farm/run.sh
```
`B4_FARM_MAX_CONCURRENT` recomputes to `floor(3/1.0)=3`; keep `CONC ≤ 3`. For more fractional
packing, lower `B4_GPU_FRACTION` on both sides (e.g. 0.25 ⇒ 4 slots on one GPU) — watch for the
non-MIG wedge.

### 5. Read metrics / latency
```bash
python3 metrics.py train.log --csv train.csv --eval-csv train.eval.csv --plot train.png
```
Watch that concurrent episodes actually overlap (per-episode `sub_session`s active at once) — on
one GPU for `fractional`, on distinct GPUs for `whole`.

### 6. Terminate
```bash
bash code/deploy/run_local_gpu_runtime.sh stop
```

## Results
Both modes passed the 2026-09-21 GPU smoke sweep (`NR=11`: 10 train steps + 1 held-out eval,
`Job succeeded`, final ckpt iter 10; artifacts in `code/demos/_smoke_logs/` under the old demo names):
`fractional` (`demo_gpu_shared_single`) `eval/kernelbench` 0.775; `whole` 0.65
(`demo_gpu_shared_multi`) and 0.70 (`demo_gpu_local`) — two runs of the same config.

## Folder layout (self-contained)
- `agentcore_slime_demo/` — vendored shared library (this demo imports ONLY this copy;
  `/work/code` is not on `PYTHONPATH`). Includes `sandbox_ray_farm.py`, `sandbox_server_agent.py`,
  `sandbox.py`.
- `deploy/Dockerfile.sandbox.gpu.shared` — the shared-GPU sandbox image (bakes `ray`, runs
  `python -m agentcore_slime_demo.sandbox_server_agent`); used by the local farm in both modes.
- `build_dataset.py`, `data/`, `run.sh`, `metrics.py`, `logs/`.

Shared infra referenced by path (NOT vendored): `code/deploy/run_local_gpu_runtime.sh` (serves the
local farm). Base checkpoint (`/work/code/p4_slime_train/ckpt/Qwen3-4B_torch_dist`), the
`agentcore_p4` container and `/root/slime` are shared.
