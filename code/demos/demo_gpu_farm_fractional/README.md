# demo_gpu_farm_fractional — KernelBench with TWO episodes sharing ONE GPU (local Ray farm, 0.5 + 0.5)

The GPU-**sharing** demo. A Ray farm inside a LOCAL AgentCore sandbox Runtime container owns host GPU 6
and splits it into two fractional, long-lived, warm-CUDA actors (`B4_FARM_GPUS=1`,
`B4_GPU_FRACTION=0.5` ⇒ `floor(1/0.5)=2` slots). Two concurrent KernelBench episodes then compile and
run their CUDA kernels on the same card at the same time. L40S (and the g6's L4) has no MIG, so each
actor caps itself to ~half the VRAM in software (`B4_PER_SANDBOX_MEM_FRACTION=0.5`, via an injected
`sitecustomize.py`).

The mode is **hard-set** in `run.sh` and `deploy/demo.env` (no `FARM=` selector). Sibling demos:
`demo_gpu_farm_whole` (same farm, one whole GPU per episode on GPUs 6,7) and `demo_gpu_single` (the
same image on a remote g6 Runtime). Formerly `demo_gpu_farm FARM=fractional`, before that
`demo_gpu_shared_single`.

## What this demo shows
- **Task:** real [KernelBench](https://huggingface.co/datasets/ScalingIntelligence/KernelBench) —
  write a `ModelNew` that matches a reference `Model` numerically **and** runs on CUDA. The
  `run_episode` harness runs **in-process** (drives SGLang directly); the only "remote" piece is the
  local sandbox Runtime container, reached over `http://127.0.0.1:8080`.
- **GPU packing via Ray:** `ray.remote(num_gpus=0.5)(SandboxActor)` — Ray places both actors on
  GPU 6 and pins each via `CUDA_VISIBLE_DEVICES`. Each actor has its own CUDA context.
- **One shared `runtimeSessionId` + per-episode `sub_session`:** AgentCore maps
  **1 `runtimeSessionId` → 1 instance**, so this is **not** "2 AgentCore sessions on one box." Every
  episode invokes the one local runtime under the SAME session id (`B4_SHARED_RUNTIME_SESSION`) and
  carries a per-episode `sub_session` that keys its GPU actor.

## GPU split
- **slime:** GPUs 0–5 — `ACTOR_GPUS=4` (train) + `ROLLOUT_GPUS=2` (infer) = 6. `CUDA_VISIBLE_DEVICES`
  is pinned to `0,1,2,3,4,5` **before** `ray start` so slime never touches the farm's GPUs.
- **local farm:** GPU 6, held by the sandbox runtime container (`docker run` on the host). GPU 7 is idle.

## Relationship to the other GPU demos
| demo | sandbox | GPU story |
|---|---|---|
| `demo_gpu_single` | one warm **remote g6** Runtime (+ `LOCAL_GPU=1`, `SNAPSHOT_EVERY=1`) | episodes on one remote L4 |
| **`demo_gpu_farm_fractional`** (this) | one **local** runtime + Ray **fractional** farm | 2 episodes share ONE GPU |
| `demo_gpu_farm_whole` | one **local** runtime + Ray **whole-GPU** farm | N episodes ↔ N whole local GPUs |

## Step by step (numbered scripts; run from this dir on the HOST — container steps `docker exec` themselves)

Everything is on-box: no ECR push, no AWS control-plane calls, no capacity provider. The farm
container runs on the HOST (docker + the nvidia container runtime); the trainer runs in `agentcore_p4`.

```bash
bash 1_prepare_data.sh               # 1. container: verify shipped KernelBench data (100 level_1 / 20 level_2; REBUILD=1 re-downloads)
bash 2_build_image.sh                # 2. amd64 image acs-gpufarm-gpu from deploy/Dockerfile.sandbox.gpu.shared
bash 3_push_ecr.sh                   # 3. skipped (local-gpu)
bash 4_deploy_runtime.sh             # 4. tag -> acs-local-gpu-sbx:fractional, start the farm on GPU 6 at 127.0.0.1:8080
bash 5_run.sh                        # 5. container: 10 train steps + 1 eval; logs/<tag>.{log,csv,eval.csv,png}
bash 6_cleanup.sh                    # 6. stop the local farm
```

Step 4 is `deploy/local_gpu_runtime.sh fractional` (the same as
`code/deploy/run_local_gpu_runtime.sh fractional`). The image pip-installs the CUDA torch stack on
first boot (kept out of the image for the 2 GB cap), so the first `/ping` can take a couple of
minutes; step 4 polls until healthy.

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
`ckpt/Qwen3-4B_kernelbench_farm_fractional_async`.

> **Keep the client OPEN gate == the server slot count.** `B4_FARM_MAX_CONCURRENT` (client) and
> `B4_MAX_SANDBOXES = floor(B4_FARM_GPUS / B4_GPU_FRACTION)` (server) must agree — both are 2 here.
> If you retune one, retune the other.

### Pack tighter
Lower `B4_GPU_FRACTION` (and `B4_PER_SANDBOX_MEM_FRACTION`) on BOTH sides, e.g. 0.25 ⇒ 4 slots on
GPU 6:
```bash
B4_GPU_FRACTION=0.25 B4_PER_SANDBOX_MEM_FRACTION=0.25 bash 4_deploy_runtime.sh
B4_GPU_FRACTION=0.25 CONC=4 bash 5_run.sh
```
Watch for the non-MIG wedge (more co-tenants ⇒ more nvcc load on one card) and the shared host RAM
(see Isolation guarantees). To give each episode its own card instead, use `demo_gpu_farm_whole`.

### Key metrics
`rollout/raw_reward`, `eval/kernelbench`, and that concurrent episodes actually overlap (per-episode
`sub_session`s active at once on GPU 6). `logs/<tag>.demo.csv` records per episode which actor
(`b4-sbx-0`/`b4-sbx-1`) served it, its `cuda_visible`, `mem_fraction` and the lease wait.

## Results
| run | config | outcome | `eval/kernelbench` | raw_reward by step | sandbox latency | notes |
|---|---|---|---|---|---|---|
| 2026-09-21 smoke (as `demo_gpu_shared_single`) | NR=11, slime 0–5 | Job succeeded, ckpt iter 10 | **0.775** | — | — | artifacts in `code/demos/_smoke_logs/` under the old name |
| 2026-09-25 smoke `logs/smoke-local-fractional_20260925_095218.*` | NR=11, env override `SLIME_CUDA_VISIBLE=2,3,4,5 ACTOR_GPUS=2 ROLLOUT_GPUS=2` (GPUs 0,1 were busy); farm on GPU 6 | Job succeeded, 10 steps + eval, ~83 min | **0.65** | 0.72 0.59 0.22 0.69 0.72 0.19 0.72 0.75 0.63 0.44 | exec p50 9.0 s / p95 19.8 s; write 4.3 ms; open 4.2 ms; close 3.6 ms (gen p50 2.2 s / p95 5.8 s) | 359 episodes, split 181/178 across `b4-sbx-1`/`b4-sbx-0`, both on host GPU 6 at `mem_fraction=0.5`, lease wait always 0; checker PASSED 325 (mean reward 0.64), IMPORT_ERROR 18, NO_OUTPUT 16 |

The two 0.775 / 0.65 evals are 1 eval × 20 problems × 2 samples each on a smoke-sized run — noise, not
a regression. Both runs predate the isolation hardening below (see the live probe for that).

## Folder layout (self-contained)
- `agentcore_slime_demo/` — vendored shared library (this demo imports ONLY this copy;
  `/work/code` is not on `PYTHONPATH`). Includes `sandbox_ray_farm.py`, `sandbox_farm_env.py`,
  `sandbox_server_agent.py`, `sandbox.py`.
- `deploy/Dockerfile.sandbox.gpu.shared` — the shared-GPU sandbox image (bakes `ray`, runs
  `python -m agentcore_slime_demo.sandbox_server_agent`); used by the local farm (GPU 6, fractional).
- `1_…6_*.sh` + `deploy/` (`demo.env`, `steps_common.sh`, `local_gpu_runtime.sh` = the farm launcher).
- `build_dataset.py`, `prepare_data.sh`, `data/`, `run.sh`, `metrics.py`, `logs/`.

Shared by path (NOT vendored): the base checkpoint (`/work/code/p4_slime_train/ckpt/Qwen3-4B_torch_dist`),
the `agentcore_p4` container and `/root/slime`.
