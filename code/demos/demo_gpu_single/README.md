# demo_gpu_single — KernelBench CUDA RL on ONE warm custom GPU Runtime (g6), + snapshot/resume and custom-Runtime angles

An agentic RL loop: the policy (Qwen3-4B) writes a CUDA `ModelNew`, a torch/CUDA checker runs it, and
the policy reads the failure and fixes it. GRPO rewards the fixes. The `run_episode` harness drives
the multi-turn loop **in-process on this box**. The checker's torch/CUDA runs **remotely** in your
custom GPU AgentCore Runtime on a `g6` capacity-provider Instance. All episodes share **one warm
g6**, so the instance count does not scale with concurrency.

> **Merged 2026-09-25.** This one demo now covers what used to be three:
> `demo_gpu_single` (the default path), `demo_gpu_snapshot` (now `SNAPSHOT_EVERY=1`) and
> `demo_gpu_customruntime` (the "why a custom Runtime" angle plus `gpu_torch_smoke.py`). All three had
> the same trainer, the same KernelBench data and the same g6 image.

| Angle | How | What it shows |
|---|---|---|
| **Warm-shared GPU sandbox** (default) | `bash 5_run.sh` | One g6 + an in-instance Ray farm. Every episode gets a fresh env on a whole GPU, and cold-boot is paid once per run |
| **Custom Runtime = the only GPU** | `python3 gpu_torch_smoke.py` | The managed Code Interpreter is a CPU microVM (`cuda_available=False`). Only your custom Runtime on a g6 has CUDA, through the same `write_file`/`exec`/`read_file` contract |
| **App-level snapshot/resume** | `SNAPSHOT_EVERY=1 bash 5_run.sh` | AgentCore has no VM/GPU snapshot. The harness restores each task's workspace from a store you own at episode ENTER and snapshots it at EXIT |

## What this demo shows
- **Task:** real **KernelBench** (`ScalingIntelligence/KernelBench`). Train = `level_1` (100
  single-op problems); held-out eval = 20 `level_2` fusions, disjoint from train.
- **Reward = GPU checker:** `run_checks.py` rebuilds the reference on the GPU with matched seeds,
  asserts `allclose` on `cuda` and reports a speedup. The failure text feeds the next turn.
- **Sandbox:** `sandbox_server_agent.py` (image `deploy/Dockerfile.sandbox.gpu.shared`) on `g6.xlarge`.
  The default is `SANDBOX=agentcore-ray-farm`: one shared `runtimeSessionId` (`B4_SHARED_RUNTIME_SESSION`)
  plus a per-episode `sub_session`, served by the in-instance Ray farm. `SANDBOX=agentcore-runtime` is the
  legacy one-dedicated-instance-per-episode path.
- **GPU split on this 8×L40S box:** 4 train + 4 inference. The checker's GPU is remote.
  `LOCAL_GPU=1` runs the same image as a local whole-GPU farm on GPUs 6,7 instead, with slime on 0–5
  and a synchronous checkpoint save.

| `SANDBOX=` | substrate | `torch.cuda.is_available()` | trains KernelBench? |
|---|---|---|---|
| `agentcore-ray-farm` (default) / `agentcore-runtime` | **your custom Runtime on a g6 Instance** | **True** | **yes** |
| `agentcore` | managed Code Interpreter, CPU microVM | False | no |
| `local` | offline subprocess (host) | whatever the host has | offline probe only |

### Snapshot/resume (`SNAPSHOT_EVERY=1`)
Every episode opens a fresh sandbox sub_session, so every episode boundary is effectively a recycle.
`agentcore_slime_demo/generate.py` wraps the episode with two hooks from `app_snapshot.py`:
- `maybe_restore(sb, "ep-<task_id>")` at ENTER writes the task's saved workspace back (resume).
- `maybe_snapshot(sb, "ep-<task_id>")` at EXIT captures it. EXIT is the quiescent point, right
  before the weight-sync barrier, so the snapshot is consistent.

The store is `snapshot_store.py`: `LocalSnapshotStore` under `SNAPSHOT_DIR`, or `S3SnapshotStore`
when `SNAPSHOT_S3_BUCKET` is set. Watch the `[app-snapshot] snapshot key=… / restore key=…` log lines.
`RECYCLE_AFTER=N` logs a simulated recycle. `python3 snapshot_demo.py` is the one-second offline
animation of the same mechanism. Why there is no real GPU-instance snapshot:
[`docs/gpu-snapshot.md`](../../../docs/gpu-snapshot.md). **Hardware status:** the snapshot glue has only
been validated offline. The live g6 + store round-trip is unverified.

## Step by step (numbered scripts; run from this dir on the HOST — container steps `docker exec` themselves)

```bash
python3 gpu_torch_smoke.py       # 0. offline: CUDA contract probe (write -> exec -> read of torch.cuda)
python3 snapshot_demo.py         #    offline: snapshot/resume animation
bash 1_prepare_data.sh           # 1. verify shipped KernelBench data (REBUILD=1 re-downloads from HF)
bash 2_build_image.sh            # 2. amd64 image from deploy/Dockerfile.sandbox.gpu.shared
bash 3_push_ecr.sh               # 3. -> ECR acs-gpusingle-gpu
bash 4_deploy_runtime.sh         # 4. Runtime acs_gpusingle_gpu on the g6 capacity provider acs_gpu_cp -> deploy/runtime.env
bash 5_run.sh                    # 5. 10 train steps + 1 eval (container); SNAPSHOT_EVERY=1 for the snapshot angle
bash 6_cleanup.sh                # 6. delete the Runtime (DELETE_CP=1 also deletes acs_gpu_cp; the g6 is the real cost)
```

**On-box, no g6:** `LOCAL_GPU=1 bash 4_deploy_runtime.sh` starts the local whole-GPU farm on GPUs 6,7.
Then run `LOCAL_GPU=1 bash 5_run.sh` and finish with `LOCAL_GPU=1 bash 6_cleanup.sh`. Skip steps 3–4's
ECR/AgentCore parts; step 4 tags the local image.

**Train to convergence:** `STEPS=20 EVAL_EVERY=10 SANDBOX_CONC=4 bash train.sh`, run inside the
container. `STEPS` maps to `NR=STEPS+1`. Keep `STEPS × RB ≤ 100`.

## Knobs (defaults in `run.sh`)

| Knob | Default | Meaning |
|---|---|---|
| `SANDBOX` | `agentcore-ray-farm` | warm-shared g6. `agentcore-runtime` = one instance per episode |
| `AGENTCORE_SANDBOX_RUNTIME_ARN` | from `deploy/runtime.env` | the g6 Runtime (required unless `LOCAL_GPU=1`) |
| `LOCAL_GPU` | `0` | `1` = local whole-GPU farm on 6,7 (slime on 0–5, sync save) |
| `B4_FARM_MAX_CONCURRENT` | `1` | client open-gate. Must not exceed the server farm slots |
| `SNAPSHOT_EVERY` | `0` | snapshot cadence in episodes. `0` = off |
| `SNAPSHOT_DIR` / `SNAPSHOT_S3_BUCKET` | `snapshots/` / unset | local store root / S3 store (production) |
| `SNAPSHOT_PATHS` | `progress.json,work.log,solution.py` | workspace files to keep |
| `RECYCLE_AFTER` | `0` | `>0` = log a simulated recycle after N episodes |
| `NR` / `RB` / `NS` / `TEMP` | 11 / 4 / 8 / 1.0 | 10 train + 1 eval; keep `NR × RB ≤ 100` |
| `EVAL_INTERVAL` / `EVAL_NS` | 10 / 2 | held-out eval cadence / samples |
| `INVOKE_MAX_RETRIES` | 8 | retries on transient invoke 5xx (cold g6) |
| `HEARTBEAT` / `HEARTBEAT_IDLE` / `HEARTBEAT_PING_FRACTION` | 0 / 600 / 0.5 | `1` = a `SessionKeeper` pings a sandbox session idle past `fraction × idle` s, so the warm g6 session survives the weight-sync stall (was `demo_cpu_heartbeat`; opt-in knob of every demo) |

## Key metrics
`rollout/raw_reward` (the RL signal) and `eval/kernelbench` (held-out solve rate). Latency:
`perf/sandbox_ms_{mean,p50,p95}` (remote checker round-trips) vs `perf/sglang_gen_ms_{mean,p50,p95}`,
plus the cold/warm open split. `5_run.sh` writes `logs/<tag>.{csv,eval.csv,png}` via `metrics.py`.

## Folder layout (self-contained)
- `1_…6_*.sh` + `deploy/` (`steps_common.sh`, `demo.env`, `agentcore_deploy.py`, `local_gpu_runtime.sh`,
  `Dockerfile.sandbox.gpu.shared`; `Dockerfile.sandbox.gpu` = the older flat server, non-farm)
- `run.sh` (trainer), `train.sh` (convergence wrapper), `metrics.py`, `prepare_data.sh`, `build_dataset.py`, `data/`
- `gpu_torch_smoke.py`, `snapshot_demo.py` (+ `agentcore_idle_sim.py`): offline probes
- `agentcore_slime_demo/`: vendored library (incl. `app_snapshot.py`, `snapshot_store.py`, `session_keeper.py`)
- `logs/snapshot/`: smoke log from the former `demo_gpu_snapshot`
