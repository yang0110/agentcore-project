# demo_gpu_single — KernelBench CUDA RL on ONE warm custom GPU Runtime (g6), + snapshot/resume and custom-Runtime angles

**Sandbox tier 3: own Runtime, custom image.** KernelBench needs CUDA, which only a custom image on a GPU capacity-provider Instance (g6) has. See the tier table in [`../README.md`](../README.md#which-sandbox-does-a-task-need-three-tiers).

An agentic RL loop: the policy (Qwen3-4B) writes a CUDA `ModelNew`, a torch/CUDA checker runs it, and
the policy reads the failure and fixes it. GRPO rewards the fixes. The `run_episode` harness drives
the multi-turn loop **in-process on this box**. The checker's torch/CUDA runs **remotely** in your
custom GPU AgentCore Runtime on a `g6` capacity-provider Instance. All episodes share **one warm
g6**, so the instance count does not scale with concurrency.

> **Merged 2026-09-25.** This one demo now covers what used to be three:
> `demo_gpu_single` (the default path), `demo_gpu_snapshot` (now `SNAPSHOT_EVERY=1`) and
> `demo_gpu_customruntime` (the "why a custom Runtime" angle plus `gpu_torch_smoke.py`). All three had
> the same trainer, the same KernelBench data and the same g6 image. The on-box Ray-farm demos are
> separate: `demo_gpu_farm` (`FARM=fractional`: two episodes share one local GPU; `FARM=whole`: one
> whole local GPU per episode) runs the same image locally, as one instance serving many sessions.

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
| `HEARTBEAT` / `HEARTBEAT_IDLE` / `HEARTBEAT_PING_FRACTION` | auto (1 on the remote g6) / 600 / 0.5 | `1` = a `SessionKeeper` pings a sandbox session idle past `fraction × idle` s, so the warm g6 session survives the weight-sync stall (was `demo_cpu_heartbeat`; on by default for remote Runtimes in every demo, off with `LOCAL_GPU=1`). Helps only our own Runtimes (the g6 here); it does **not** extend a managed Code Interpreter session, and the `LOCAL_GPU=1` farm has no idle timeout |

## Key metrics
`rollout/raw_reward` (the RL signal) and `eval/kernelbench` (held-out solve rate). Latency:
`perf/sandbox_ms_{mean,p50,p95}` (remote checker round-trips) vs `perf/sglang_gen_ms_{mean,p50,p95}`,
plus the cold/warm open split. `5_run.sh` writes `logs/<tag>.{csv,eval.csv,png}` via `metrics.py`.

## Results
| run | config | outcome | `eval/kernelbench` | raw_reward by step | latency | notes |
|---|---|---|---|---|---|---|
| 2026-09-18 `smoke_run.log` | earlier smoke (config in the log) | Job succeeded | 0.15 | — | — | pre-merge artifact |
| 2026-09-25 remote smoke `logs/smoke-remote_20260925_113622.*` | NR=11, g6 Runtime (image baked `B4_GPU_FRACTION=0.5` ⇒ 2 actors on the L4, client gate 1), env override `SLIME_CUDA_VISIBLE=2..7 ACTOR_GPUS=4 ROLLOUT_GPUS=2` (GPUs 0,1 busy) | **Job failed at step 5** (rc=1), no eval | — | 0.53 0.47 0.25 0.56 | cold open 207 s (one `Insufficient EC2 capacity g6.xlarge`, retried); warm open p50 153 ms / p95 224 ms; write p50 119 ms; exec p50 5.4 s / p95 17.0 s; close 110 ms (gen p50 2.1 s / p95 6.6 s) | 136 episodes (69/68 across `b4-sbx-0/1`); checker PASSED 93 (mean 0.71), SETUP_ERROR 27, IMPORT_ERROR 16. Cause: host-RAM OOM, one `run_checks.py` at 12.09 GB on the 16 GB g6.xlarge, Ray killed both actors, next exec HTTP 500 (see Isolation guarantees) |

The 09-25 Runtime and its ECR repo were deleted after the run; the capacity provider `acs_gpu_cp` was
kept. Note the mismatch this run exposed: `run.sh` describes one WHOLE L4 per episode, but
`deploy/demo.env` bakes `B4_GPU_FRACTION=0.5`, so the g6 holds two warm actors (two CUDA contexts and
two sets of Ray worker RAM) while the client gate only ever uses one at a time.

## Isolation guarantees

The default `SANDBOX=agentcore-ray-farm` puts every episode in ONE warm g6 (one shared `runtimeSessionId`, per-episode `sub_session`). `SANDBOX=agentcore-runtime` gives each episode its own instance, i.e. the left column. Be clear about what the right column does and does not give you:

| | session per instance (VM boundary) | shared sub_session (this demo's default: Ray farm in one g6) |
|---|---|---|
| **Filesystem** | Separate instance and disk; nothing shared. | Fresh `…/actor-<slot>/<sub_session>/` per episode, realpath-confined `read_file`/`write_file`, `rmtree` on close. **On the g6 all slots share one uid** (see next row), so a command run inside the sandbox can read a co-resident episode's files; the 0700/0711 modes only bite on the local farm. |
| **Process / user** | Separate kernel, PID space and users. | **Not available on the g6.** The AgentCore Runtime container runs as uid 0 with **zero capabilities** (CapEff/CapPrm/CapBnd = 0, NoNewPrivs=1, seccomp filter; probed live 2026-09-25), so setuid is impossible and the farm falls back to one shared uid (`ping`/`open` report `uid_isolation: false`). Left: path confinement for `read_file`/`write_file`, a fresh dir per episode, and a kill of the exec's process group. A `setsid` escapee survives close and can read the next episode's files. The per-slot uid works on the local farm (`LOCAL_GPU=1`), where docker root keeps CHOWN/KILL/SETGID/SETUID. |
| **GPU memory** | Whole GPU per instance. | Separate CUDA contexts per slot with **driver-enforced address spaces** (one slot cannot read another's device memory). The per-process **fraction cap is cooperative**: `sitecustomize.py` calls `torch.cuda.set_per_process_memory_fraction`, which only limits torch's caching allocator. A checker that raises the fraction, or allocates through raw CUDA / another library, can take the whole card and OOM its co-tenant. |
| **GPU fault blast radius** | One GPU per VM; an Xid or wedge stays in that instance. | **Shared.** There is no MIG on L4/L40S, so an Xid, a hung kernel or a wedged context hits every co-tenant on that GPU. We saw the g6's L4 wedge under sustained nvcc load with fractional sharing; a lower packing (FRAC=0.5) made it rarer but does not remove it. |
| **Compute interference** | None (dedicated). | Time-sliced SMs with no MPS or priorities: a heavy kernel slows the other slot's kernels and skews its measured speedup (reward noise). CPU for nvcc is shared too. |
| **RAM / CPU** | Per instance (16 GB on g6.xlarge). | **Shared, no per-slot limit.** Observed 2026-09-25 on the g6 (`logs/smoke-remote_20260925_113622.log`): one checker (`run_checks.py`) grew to 12.09 GB of the g6.xlarge's 16 GB, Ray's memory monitor killed BOTH farm actors, the next exec returned HTTP 500 and the job failed at step 5. A per-slot `RLIMIT_AS` or cgroup `memory.max` would contain it; not implemented (and cgroup writes are unavailable inside the zero-capability Runtime). |
| **Network** | Own instance network stack; VPC/SG policy is set per Runtime. | **Shared network namespace.** Slots can connect to each other's listening ports and to the farm server itself on `localhost:8080`. Sub_session ids are 128-bit random and not listable, so driving another episode needs a leaked id, but nothing else stops it. Blocking uids `20000+` from `:8080` (iptables owner match) would close this; not done. |
| **Kernel** | Separate guest kernel (the VM boundary). | **Shared kernel** (and, on the g6, one container with one seccomp profile). A kernel or NVIDIA-driver exploit crosses slots. uid separation is ordinary Unix DAC, not a security boundary against a determined attacker. |

**Caches and poisoning.** By default each slot uid has its own HOME under `/var/tmp/b4-home/uid-<N>` (0700), with `XDG_CACHE_HOME` and `TORCH_EXTENSIONS_DIR` inside it, so compiled torch extensions are reused warm across episodes. The risk: episodes that later run on the SAME slot share that cache, so one episode can plant a poisoned `.so` or pip/pycache entry that a later episode loads (on the g6 every slot shares one HOME, so the risk spans all episodes). `SANDBOX_ISOLATE_HOME=1` removes it: HOME and every cache move into the per-episode dir (`sandbox_farm_env.fresh_env(isolate_caches=True)`) and are wiped on close, at the cost of recompiling every extension per episode. The nvcc serialization lock `/dev/shm/b4-nvcc.lock` is created 0644 by the actor and opened read-only by the checker (flock works on a read-only fd); if it cannot be opened the build just runs unserialized.

**What was verified.** On the g6 (2026-09-25, a plain exec on the deployed image): uid 0 with
CapInh/CapPrm/CapEff/CapBnd/CapAmb all 0, NoNewPrivs=1, Seccomp=2, 4 vCPU, one L4. So on this demo's
default path the farm runs in its shared-uid fallback. The per-slot-uid mode was probed live on the
local farm (`demo_gpu_farm/logs/isolation_probe_20260925.log`), which is what
`LOCAL_GPU=1` gets. **g6 image rebuild: pending.** The hardened farm code has not been deployed to a
g6; the next `2_build_image.sh` + `4_deploy_runtime.sh` picks it up. It was not rebuilt now because,
given the measured capabilities, it could only show the fallback.


## Folder layout (self-contained)
- `1_…6_*.sh` + `deploy/` (`steps_common.sh`, `demo.env`, `agentcore_deploy.py`, `local_gpu_runtime.sh`,
  `Dockerfile.sandbox.gpu.shared`; `Dockerfile.sandbox.gpu` = the older flat server, non-farm)
- `run.sh` (trainer), `train.sh` (convergence wrapper), `metrics.py`, `prepare_data.sh`, `build_dataset.py`, `data/`
- `gpu_torch_smoke.py`, `snapshot_demo.py` (+ `agentcore_idle_sim.py`): offline probes
- `agentcore_slime_demo/`: vendored library (incl. `sandbox_ray_farm.py`, `sandbox_farm_env.py`, `app_snapshot.py`, `snapshot_store.py`, `session_keeper.py`)
- `logs/snapshot/`: smoke log from the former `demo_gpu_snapshot`
