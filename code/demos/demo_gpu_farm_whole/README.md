# demo_gpu_farm_whole — KernelBench with ONE WHOLE GPU per episode (local Ray farm on GPUs 6,7)

**Sandbox tier 3: own Runtime, custom image.** The CUDA sandbox image of the g6 GPU Runtime, run here as a local farm on GPUs 6,7. See the tier table in [`../README.md`](../README.md#which-sandbox-does-a-task-need-three-tiers).

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
weight-sync stall — the former `demo_cpu_heartbeat`, now a knob of every demo, default on only for remote Runtimes — so off here). The heartbeat
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

## Isolation guarantees

All episodes run in ONE local runtime container (one shared `runtimeSessionId`, per-episode `sub_session`). The left column is what you would get with one AgentCore session per episode (the VM boundary; `demo_gpu_single SANDBOX=agentcore-runtime` on a g6):

| | session per instance (VM boundary) | shared sub_session (this demo: 1 GPU per slot on GPUs 6,7) |
|---|---|---|
| **Filesystem** | Separate instance and disk; nothing shared. | Each episode gets `…/actor-<slot>/<sub_session>/`, owned by the slot's uid, mode 0700; the farm and actor roots are 0711, so a neighbour can neither read nor list it. The server's `read_file`/`write_file` are realpath-confined to the episode dir (`../`, absolute and symlink escapes are refused). The dir is `rmtree`d on close. Still shared: the image FS, world-writable `/tmp` and `/dev/shm`, and the per-uid HOME/caches (below). |
| **Process / user** | Separate kernel, PID space and users. | Each slot's commands run as their own non-root uid `20000+slot` (subprocess `user=`/`group=`/`extra_groups=[]`, umask 077; no Python `preexec_fn` in the threaded Ray actor), so a neighbour's processes cannot be signalled or ptraced. Same PID namespace: `ps` shows every slot's command lines. On close the farm SIGKILLs the exec's process group and then every process of the uid, which also catches `setsid`/`nohup` escapees. Episodes that reuse a slot later reuse its uid. |
| **GPU memory** | Whole GPU per instance. | **No GPU-memory sharing:** one whole GPU per actor. Separation is by `CUDA_VISIBLE_DEVICES`, which is an env var, not a device cgroup: the container sees both GPUs 6 and 7, so a checker that resets the variable can open the other episode's card. Contexts are still separate and driver-enforced (no reading another context's memory). |
| **GPU fault blast radius** | One GPU per VM; an Xid stays in that instance. | Mostly per-GPU: an Xid on GPU 6 does not take down GPU 7. But both sit behind one host NVIDIA driver and one container, so a driver hang or a container crash (the 2026-09-19 run lost the whole farm mid-run) hits both episodes. |
| **Compute interference** | None (dedicated). | No SM sharing. Host CPU (nvcc, Python) is shared, so compile-heavy neighbours still slow each other and add noise to the speedup reward. |
| **RAM / CPU** | Per instance. | **Shared, no per-slot limit.** One checker can exhaust the container's RAM and get every actor killed; this happened on the g6 (a 12 GB `run_checks.py` on a 16 GB g6.xlarge killed both actors, 2026-09-25, see `demo_gpu_single`). A per-slot `RLIMIT_AS` / cgroup `memory.max` would contain it; not implemented. |
| **Network** | Own instance network stack; VPC/SG policy is set per Runtime. | **Shared network namespace.** Slots can connect to each other's listening ports and to the farm server itself on `localhost:8080`. Sub_session ids are 128-bit random and not listable, so driving another episode needs a leaked id, but nothing else stops it. Blocking uids `20000+` from `:8080` (iptables owner match) would close this; not done. |
| **Kernel** | Separate guest kernel (the VM boundary). | **Shared kernel** (and, on the g6, one container with one seccomp profile). A kernel or NVIDIA-driver exploit crosses slots. uid separation is ordinary Unix DAC, not a security boundary against a determined attacker. |

**Caches and poisoning.** By default each slot uid has its own HOME under `/var/tmp/b4-home/uid-<N>` (0700), with `XDG_CACHE_HOME` and `TORCH_EXTENSIONS_DIR` inside it, so compiled torch extensions are reused warm across episodes. The risk: episodes that later run on the SAME slot share that cache, so one episode can plant a poisoned `.so` or pip/pycache entry that a later episode loads. `SANDBOX_ISOLATE_HOME=1` removes it: HOME and every cache move into the per-episode dir (`sandbox_farm_env.fresh_env(isolate_caches=True)`) and are wiped on close, at the cost of recompiling every extension per episode. The nvcc serialization lock `/dev/shm/b4-nvcc.lock` is created 0644 by the actor and opened read-only by the checker (flock works on a read-only fd); if it cannot be opened the build just runs unserialized.

**Verification.** The farm code is byte-identical to `demo_gpu_farm_fractional`, where it was probed
live on GPU 6 on 2026-09-25 (per-slot uids 20000/20001, cross-read denied, confinement, separate CUDA
contexts, clean close; see `demo_gpu_farm_fractional/logs/isolation_probe_20260925.log`). The whole-GPU
layout itself was **not** re-probed: the one-GPU-per-slot claim rests on Ray's `CUDA_VISIBLE_DEVICES`
placement, which the 2026-09-21 smoke logs show, not on a new test.


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
