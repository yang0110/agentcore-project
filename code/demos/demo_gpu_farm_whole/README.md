# demo_gpu_farm_whole — KernelBench with ONE WHOLE GPU per episode (local Ray farm on GPUs 6,7)

The scale-out-on-one-box demo. A Ray farm inside a LOCAL AgentCore sandbox Runtime container owns host
GPUs 6,7 and runs one long-lived, warm-CUDA actor per GPU (`B4_FARM_GPUS=2`, `B4_GPU_FRACTION=1.0` ⇒
`floor(2/1.0)=2` slots). Each concurrent KernelBench episode leases a whole card, so no two episodes
share GPU memory or SMs. That removes the need for a software VRAM guard and sidesteps the non-MIG nvcc
wedge that fractional sharing can hit under sustained compile load. The episodes still share the host
(RAM, CPU, kernel, NVIDIA driver, network): see Isolation guarantees.

The mode is **hard-set** in `run.sh` and `deploy/demo.env` (no `FARM=` selector). Sibling demos:
`demo_gpu_farm_fractional` (same farm, two episodes on ONE GPU) and `demo_gpu_single` (the same image
on a remote g6 Runtime). Formerly `demo_gpu_farm FARM=whole`, before that `demo_gpu_shared_multi`
and `demo_gpu_local` (identical config).

## What this demo shows
- **Task:** real [KernelBench](https://huggingface.co/datasets/ScalingIntelligence/KernelBench) —
  write a `ModelNew` that matches a reference `Model` numerically **and** runs on CUDA. The
  `run_episode` harness runs **in-process** (drives SGLang directly); the only "remote" piece is the
  local sandbox Runtime container, reached over `http://127.0.0.1:8080`.
- **GPU placement via Ray:** `ray.remote(num_gpus=1.0)(SandboxActor)` — Ray puts one actor on each
  of GPUs 6,7 and pins it via `CUDA_VISIBLE_DEVICES`.
- **One shared `runtimeSessionId` + per-episode `sub_session`:** AgentCore maps
  **1 `runtimeSessionId` → 1 instance**, so this is **not** "2 AgentCore sessions on one box." Every
  episode invokes the one local runtime under the SAME session id (`B4_SHARED_RUNTIME_SESSION`) and
  carries a per-episode `sub_session` that keys its GPU actor.

## GPU split
- **slime:** GPUs 0–5 — `ACTOR_GPUS=4` (train) + `ROLLOUT_GPUS=2` (infer) = 6. `CUDA_VISIBLE_DEVICES`
  is pinned to `0,1,2,3,4,5` **before** `ray start` so slime never touches the farm's GPUs.
- **local farm:** GPUs 6,7, held by the sandbox runtime container (`docker run` on the host).

## Relationship to the other GPU demos
| demo | sandbox | GPU story |
|---|---|---|
| `demo_gpu_single` | one warm **remote g6** Runtime (+ `LOCAL_GPU=1`, `SNAPSHOT_EVERY=1`) | episodes on one remote L4 |
| `demo_gpu_farm_fractional` | one **local** runtime + Ray **fractional** farm | 2 episodes share ONE GPU |
| **`demo_gpu_farm_whole`** (this) | one **local** runtime + Ray **whole-GPU** farm | N episodes ↔ N whole local GPUs |

## Step by step (numbered scripts; run from this dir on the HOST — container steps `docker exec` themselves)

Everything is on-box: no ECR push, no AWS control-plane calls, no capacity provider. The farm
container runs on the HOST (docker + the nvidia container runtime); the trainer runs in `agentcore_p4`.

```bash
bash 1_prepare_data.sh               # 1. container: verify shipped KernelBench data (100 level_1 / 20 level_2; REBUILD=1 re-downloads)
bash 2_build_image.sh                # 2. amd64 image acs-gpufarmwhole-gpu from deploy/Dockerfile.sandbox.gpu.shared
bash 3_push_ecr.sh                   # 3. skipped (local-gpu)
bash 4_deploy_runtime.sh             # 4. tag -> acs-local-gpu-sbx:whole, start the farm on GPUs 6,7 at 127.0.0.1:8080
bash 5_run.sh                        # 5. container: 10 train steps + 1 eval; logs/<tag>.{log,csv,eval.csv,png}
bash 6_cleanup.sh                    # 6. stop the local farm
```

Step 4 is `deploy/local_gpu_runtime.sh whole` (the same as `code/deploy/run_local_gpu_runtime.sh
whole`). The image pip-installs the CUDA torch stack on first boot (kept out of the image for the 2 GB
cap), so the first `/ping` can take a couple of minutes; step 4 polls until healthy.

Knobs (defaults in `run.sh`): `RB` (4), `NS` (8), `NR` (11 ⇒ real steps =
NR−1 = 10 — keep **`NR × RB ≤ 100`**), `TEMP` (1.0), `CONC` (2 ⇒ `--sglang-server-concurrency`; keep
**`CONC ≤ B4_MAX_SANDBOXES = 2`** so episodes don't block on the farm's checkout gate), `EVAL_INTERVAL`
(10), `EVAL_NS` (2), `INVOKE_MAX_RETRIES` (**20** ≈ 10 min patience — do not lower), and optionally
`B4_SHARED_RUNTIME_SESSION` (pin the shared session id to reuse a warm farm across runs; auto-minted
as `b4farm-<32 hex>` if unset), and `HEARTBEAT` (0; `1` = a `SessionKeeper` pings any sandbox
session idle past `HEARTBEAT_PING_FRACTION × HEARTBEAT_IDLE` = 0.5 × 600 s, so it survives the
weight-sync stall — the former `demo_cpu_heartbeat`, now an opt-in knob of every demo). The heartbeat
only helps sessions that can idle out, i.e. our own AgentCore Runtimes; it does **not** extend a
managed Code Interpreter session, and the local farm container here has no idle timeout at all, so
it is a no-op in this demo unless you point it at a remote Runtime. Checkpoints go to
`ckpt/Qwen3-4B_kernelbench_farm_whole_async`.

> **Keep the client OPEN gate == the server slot count.** `B4_FARM_MAX_CONCURRENT` (client) and
> `B4_MAX_SANDBOXES = floor(B4_FARM_GPUS / B4_GPU_FRACTION)` (server) must agree — both are 2 here.
> If you retune one, retune the other.

### Scale the farm
Reserve more GPUs for the farm and raise `B4_FARM_GPUS` to match. E.g. a 3-GPU whole farm (5,6,7),
slime on 5:
```bash
LOCAL_RT_GPUS=5,6,7 B4_FARM_GPUS=3 bash 4_deploy_runtime.sh
SLIME_CUDA_VISIBLE=0,1,2,3,4 SLIME_NUM_GPUS=5 ROLLOUT_GPUS=1 B4_FARM_GPUS=3 CONC=3 bash 5_run.sh
```
`B4_FARM_MAX_CONCURRENT` recomputes to `floor(3/1.0)=3`; keep `CONC ≤ 3`. To pack several episodes
onto one card instead, use `demo_gpu_farm_fractional`.

### Key metrics
`rollout/raw_reward`, `eval/kernelbench`, and that concurrent episodes actually overlap (per-episode
`sub_session`s active at once, on distinct GPUs). `logs/<tag>.demo.csv` records per episode which
actor served it and its `cuda_visible`.

## Results
| run | config | outcome | `eval/kernelbench` | notes |
|---|---|---|---|---|
| 2026-09-19 `logs/run_20260919_033018.whole-gpu_local.log` (as `demo_gpu_local`) | NR=11, slime 0–5, farm 6,7 | **Job failed** after step 1 (raw_reward 0.69) | — | the farm container died mid-run (`Connection refused` on 127.0.0.1:8080, 84×): likely the Ray-GCS failure mode that `local_gpu_runtime.sh` now absorbs with `--shm-size` + `--restart=on-failure` |
| 2026-09-21 smoke (as `demo_gpu_shared_multi`) | NR=11, slime 0–5, farm 6,7 | Job succeeded, ckpt iter 10 | **0.65** | artifacts in `code/demos/_smoke_logs/` under the old name |
| 2026-09-21 smoke (as `demo_gpu_local`) | same config | Job succeeded, ckpt iter 10 | **0.70** | second run of the identical config |

Not re-run since the split into its own folder or the isolation hardening (the farm code is identical
to `demo_gpu_farm_fractional`, where the hardened code was probed live).

## Folder layout (self-contained)
- `agentcore_slime_demo/` — vendored shared library (this demo imports ONLY this copy;
  `/work/code` is not on `PYTHONPATH`). Includes `sandbox_ray_farm.py`, `sandbox_farm_env.py`,
  `sandbox_server_agent.py`, `sandbox.py`.
- `deploy/Dockerfile.sandbox.gpu.shared` — the shared-GPU sandbox image (bakes `ray`, runs
  `python -m agentcore_slime_demo.sandbox_server_agent`); used by the local farm (GPUs 6,7, whole).
- `1_…6_*.sh` + `deploy/` (`demo.env`, `steps_common.sh`, `local_gpu_runtime.sh` = the farm launcher).
- `build_dataset.py`, `prepare_data.sh`, `data/`, `run.sh`, `metrics.py`, `logs/`.

Shared by path (NOT vendored): the base checkpoint (`/work/code/p4_slime_train/ckpt/Qwen3-4B_torch_dist`),
the `agentcore_p4` container and `/root/slime`.
