# demo_gpu_local — KernelBench CUDA task, a LOCAL whole-GPU Ray farm on this box's reserved GPUs (6,7)

The standalone **whole-GPU-per-session** demo. It serves the SAME Ray-farm sandbox Runtime that
`demo_gpu_shared` uses, but **locally on this 8×L40S box** and with the farm fraction set to **1.0**,
so each concurrent episode gets a **whole dedicated GPU** instead of a fractional slice. Reserve GPUs
6,7 for the farm (`B4_FARM_GPUS=2`, `B4_GPU_FRACTION=1.0` → `floor(2/1.0)=2` slots), give slime the
other 6. Two GPUs → **two whole-GPU sub-session slots, one GPU each**.

## What this demo shows
- **Task:** real [KernelBench](https://huggingface.co/datasets/ScalingIntelligence/KernelBench) —
  write a `ModelNew` that matches a reference `Model` numerically **and** runs on CUDA. The
  `run_episode` harness runs **in-process** (drives SGLang directly); the only "remote" piece is the
  local sandbox runtime container, reached over `http://127.0.0.1:8080`.
- **Whole-GPU packing via Ray:** `ray.remote(num_gpus=1.0)(SandboxActor)` places one long-lived,
  warm-CUDA actor **per GPU**. With 2 reserved GPUs that is 2 actors, and Ray pins each actor to a
  distinct physical GPU via `CUDA_VISIBLE_DEVICES`. Because each episode owns a whole GPU, there is
  **no fractional memory guard to tune and no MIG** — which sidesteps the non-MIG nvcc-wedge that
  fractional sharing (`demo_gpu_shared`) can hit under sustained compile load.
- **One shared `runtimeSessionId` + per-episode `sub_session`:** AgentCore maps
  **1 `runtimeSessionId` → 1 instance**, so this is **not** "2 AgentCore sessions on one box." Every
  episode across every rollout worker invokes the one local runtime under the SAME session id
  (`B4_SHARED_RUNTIME_SESSION`) and carries a per-episode `sub_session` that keys its whole-GPU actor.

## GPU split
- **slime:** GPUs 0–5 — `ACTOR_GPUS=4` (train) + `ROLLOUT_GPUS=2` (infer) = 6. `CUDA_VISIBLE_DEVICES`
  is pinned to `0,1,2,3,4,5` **before** `ray start` so slime never touches the farm's GPUs.
- **local farm:** GPUs 6,7 — held by the sandbox runtime container (`docker run` on the host), one
  whole-GPU Ray actor each.

## Relationship to the other GPU demos
| demo | sandbox | GPU story |
|---|---|---|
| `demo_gpu_single` | one dedicated **remote g6** (`SANDBOX=agentcore-runtime`) | 1 episode ↔ 1 whole remote GPU |
| `demo_gpu_shared` | one shared **remote g6** + Ray **fractional** farm (`agentcore-ray-farm`) | many episodes share ONE GPU (`B4_GPU_FRACTION<1`) |
| **`demo_gpu_local`** | one **local** runtime + Ray **whole-GPU** farm (`agentcore-ray-farm`) | N episodes ↔ N whole **local** GPUs (`B4_GPU_FRACTION=1.0`) |

## Step by step

### 0. Prereqs
The trainer runs inside the persistent slime container `agentcore_p4` (repo at `/work/code`, slime at
`/root/slime`); region `us-west-2`. The **local runtime container** is launched on the **HOST**
(needs docker + the nvidia container runtime), not inside `agentcore_p4`. No AWS control-plane calls
and no capacity provider are needed — everything is on-box.

### 1. Build the dataset
Already shipped in `data/` (identical KernelBench slices). To rebuild:
```bash
python3 build_dataset.py --dataset kernelbench --level level_1 --limit 100 --out data/kernelbench_train.jsonl
python3 build_dataset.py --dataset kernelbench --level level_2 --limit 20  --out data/kernelbench_eval.jsonl
```

### 2. Start the local whole-GPU farm (on the HOST)
This builds the shared-GPU sandbox image (context = this demo dir, so the `COPY agentcore_slime_demo/…`
lines resolve against the vendored copy), reserves GPUs 6,7, and serves the farm at
`http://127.0.0.1:8080` with the whole-GPU defaults baked in (`B4_FARM_GPUS=2`, `B4_GPU_FRACTION=1.0`,
`B4_PER_SANDBOX_MEM_FRACTION=0.95`):
```bash
bash code/deploy/run_local_gpu_runtime.sh local
# reuse an already-built image:
SKIP_BUILD=1 bash code/deploy/run_local_gpu_runtime.sh local
```
The image pip-installs the CUDA torch stack on first boot (kept out of the image for the 2 GB
AgentCore cap), so the first `/ping` can take a couple of minutes; the launcher polls until healthy.

### 3. Run RL training (inside agentcore_p4)
No `LOCAL_GPU` flag — this demo is always local:
```bash
docker exec agentcore_p4 bash -lc 'cd /work && AWS_REGION=us-west-2 \
    bash code/demos/demo_gpu_local/run.sh'
```
Knobs (defaults in `run.sh`): `RB` (4), `NS` (8), `NR` (11 ⇒ real steps = NR−1 = 10 — keep
**`NR × RB ≤ 100`**), `TEMP` (1.0), `CONC` (2 ⇒ `--sglang-server-concurrency`; keep
**`CONC ≤ B4_MAX_SANDBOXES = 2`** so episodes don't block on the farm's checkout gate), `EVAL_INTERVAL`
(10), `EVAL_NS` (2), `INVOKE_MAX_RETRIES` (**20** ≈ 10 min patience — do not lower), and optionally
`B4_SHARED_RUNTIME_SESSION` (pin the shared session id to reuse a warm farm across runs; auto-minted
as `b4farm-<32 hex>` if unset). slime runs on GPUs 0–5 (`ACTOR_GPUS=4` + `ROLLOUT_GPUS=2`); the farm
owns GPUs 6,7.

> **Keep the client OPEN gate == the server slot count.** `B4_FARM_MAX_CONCURRENT` (client) and
> `B4_MAX_SANDBOXES = floor(B4_FARM_GPUS / B4_GPU_FRACTION)` (server) must agree — both default to
> `floor(2/1.0) = 2` here. If you retune one, retune the other.

### 4. Scale to N whole GPUs
Whole-GPU packing scales linearly: reserve more GPUs for the farm and raise `B4_FARM_GPUS` to match,
keeping `B4_GPU_FRACTION=1.0`. E.g. to give the farm 3 GPUs (5,6,7) and slime 5:
```bash
# HOST:
LOCAL_RT_GPUS=5,6,7 B4_FARM_GPUS=3 bash code/deploy/run_local_gpu_runtime.sh local
# agentcore_p4:
SLIME_CUDA_VISIBLE=0,1,2,3,4 SLIME_NUM_GPUS=5 ROLLOUT_GPUS=1 \
  B4_FARM_GPUS=3 CONC=3 bash code/demos/demo_gpu_local/run.sh
```
`B4_FARM_MAX_CONCURRENT` recomputes to `floor(3/1.0)=3`; keep `CONC ≤ 3`.

### 5. Read metrics / latency
```bash
python3 metrics.py train.log --csv train.csv --eval-csv train.eval.csv --plot train.png
```
Watch that concurrent episodes actually overlap on distinct GPUs (per-episode `sub_session`s active
at once, each pinned to its own GPU) — that is the whole-GPU packing this demo proves.

### 6. Terminate
```bash
bash code/deploy/run_local_gpu_runtime.sh stop
```

## Folder layout (self-contained)
- `agentcore_slime_demo/` — vendored shared library (this demo imports ONLY this copy;
  `/work/code` is not on `PYTHONPATH`). Includes `sandbox_ray_farm.py`, `sandbox_server_agent.py`,
  `sandbox.py`.
- `deploy/Dockerfile.sandbox.gpu.shared` — the shared-GPU sandbox image (bakes `ray`, runs
  `python -m agentcore_slime_demo.sandbox_server_agent`); reused as-is for the local farm.
- `build_dataset.py`, `data/`, `run.sh`, `metrics.py`.

Shared infra referenced by path (NOT vendored): `code/deploy/run_local_gpu_runtime.sh` (serves the
local farm on GPUs 6,7). Base checkpoint is shared.
