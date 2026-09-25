# demo_gpu_farm_fractional — KernelBench with TWO episodes sharing ONE GPU (local Ray farm, 0.5 + 0.5)

**Sandbox tier 3: own Runtime, custom image.** The CUDA sandbox image of the g6 GPU Runtime, run here as a local farm on GPU 6. See the tier table in [`../README.md`](../README.md#which-sandbox-does-a-task-need-three-tiers).

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

## Isolation guarantees

All episodes run in ONE local runtime container (one shared `runtimeSessionId`, per-episode `sub_session`). The left column is what you would get with one AgentCore session per episode (the VM boundary; `demo_gpu_single SANDBOX=agentcore-runtime` on a g6):

| | session per instance (VM boundary) | shared sub_session (this demo: 2 slots on GPU 6) |
|---|---|---|
| **Filesystem** | Separate instance and disk; nothing shared. | Each episode gets `…/actor-<slot>/<sub_session>/`, owned by the slot's uid, mode 0700; the farm and actor roots are 0711, so a neighbour can neither read nor list it. The server's `read_file`/`write_file` are realpath-confined to the episode dir (`../`, absolute and symlink escapes are refused). The dir is `rmtree`d on close. Still shared: the image FS, world-writable `/tmp` and `/dev/shm`, and the per-uid HOME/caches (below). |
| **Process / user** | Separate kernel, PID space and users. | Each slot's commands run as their own non-root uid `20000+slot` (subprocess `user=`/`group=`/`extra_groups=[]`, umask 077; no Python `preexec_fn` in the threaded Ray actor), so a neighbour's processes cannot be signalled or ptraced. Same PID namespace: `ps` shows every slot's command lines. On close the farm SIGKILLs the exec's process group and then every process of the uid, which also catches `setsid`/`nohup` escapees. Episodes that reuse a slot later reuse its uid. |
| **GPU memory** | Whole GPU per instance. | Separate CUDA contexts per slot with **driver-enforced address spaces** (one slot cannot read another's device memory). The per-process **fraction cap is cooperative**: `sitecustomize.py` calls `torch.cuda.set_per_process_memory_fraction`, which only limits torch's caching allocator. A checker that raises the fraction, or allocates through raw CUDA / another library, can take the whole card and OOM its co-tenant. |
| **GPU fault blast radius** | One GPU per VM; an Xid or wedge stays in that instance. | **Shared.** There is no MIG on L4/L40S, so an Xid, a hung kernel or a wedged context hits every co-tenant on that GPU. We saw the g6's L4 wedge under sustained nvcc load with fractional sharing; a lower packing (FRAC=0.5) made it rarer but does not remove it. |
| **Compute interference** | None (dedicated). | Time-sliced SMs with no MPS or priorities: a heavy kernel slows the other slot's kernels and skews its measured speedup (reward noise). CPU for nvcc is shared too. |
| **RAM / CPU** | Per instance. | **Shared, no per-slot limit.** One checker can exhaust the container's RAM and get every actor killed; this happened on the g6 (a 12 GB `run_checks.py` on a 16 GB g6.xlarge killed both actors, 2026-09-25, see `demo_gpu_single`). A per-slot `RLIMIT_AS` / cgroup `memory.max` would contain it; not implemented. |
| **Network** | Own instance network stack; VPC/SG policy is set per Runtime. | **Shared network namespace.** Slots can connect to each other's listening ports and to the farm server itself on `localhost:8080`. Sub_session ids are 128-bit random and not listable, so driving another episode needs a leaked id, but nothing else stops it. Blocking uids `20000+` from `:8080` (iptables owner match) would close this; not done. |
| **Kernel** | Separate guest kernel (the VM boundary). | **Shared kernel** (and, on the g6, one container with one seccomp profile). A kernel or NVIDIA-driver exploit crosses slots. uid separation is ordinary Unix DAC, not a security boundary against a determined attacker. |

**Caches and poisoning.** By default each slot uid has its own HOME under `/var/tmp/b4-home/uid-<N>` (0700), with `XDG_CACHE_HOME` and `TORCH_EXTENSIONS_DIR` inside it, so compiled torch extensions are reused warm across episodes. The risk: episodes that later run on the SAME slot share that cache, so one episode can plant a poisoned `.so` or pip/pycache entry that a later episode loads. `SANDBOX_ISOLATE_HOME=1` removes it: HOME and every cache move into the per-episode dir (`sandbox_farm_env.fresh_env(isolate_caches=True)`) and are wiped on close, at the cost of recompiling every extension per episode. The nvcc serialization lock `/dev/shm/b4-nvcc.lock` is created 0644 by the actor and opened read-only by the checker (flock works on a read-only fd); if it cannot be opened the build just runs unserialized.

**Verified live** on this demo's local farm (2026-09-25, `bash 4_deploy_runtime.sh` = fractional on
GPU 6, then `python3 isolation_probe.py`; output `logs/isolation_probe_20260925.log`):
- slots run as uid 20000 / 20001 (`groups=` only their own gid); A's dir is `drwx------ 20000`, the
  actor/farm roots `drwx--x--x root`.
- B's `cat` / `ls` of A's file, A's dir, the actor root, the farm root and A's HOME all give
  *Permission denied*; a symlink B plants to A's dir is also denied.
- B's `read_file` of `../../actor-0/…`, the absolute path and the symlink are refused. (In that run the
  refusal surfaced as an HTTP 500; it now returns `{"text": "", "error": "… outside this sub_session"}`
  so the client does not retry it. That change was re-checked offline as root, not re-run live.)
- CUDA, both slots at once: each process sees the L40S as `cuda:0`; host `nvidia-smi` on GPU 6 shows
  **4 separate compute contexts** (2 warm actors at 496 MiB + the 2 checkers at 14 GiB each).
  Allocating 0.3 × total succeeds and a further 0.3 × total raises `CUDA out of memory` in **both**
  slots, so the 0.5 cap is applied per process (cooperatively, see the table).
- nvcc lock: `/dev/shm/b4-nvcc.lock` is `-rw-r--r-- root` and readable by uid 20000; the
  `load_inline` wrapper in `sitecustomize.py` ran under the uid. The actual build then stopped at
  *Ninja is required*: this image ships neither ninja nor nvcc, so a real compile under the uid was
  **not** exercised.
- close: a timed-out exec is killed (exit 124); close A reports `killed=3, left_procs=0`, A's dir is
  gone, and the next episode on slot 0 (same uid 20000) starts with only `sitecustomize.py` + `tmp/`.
  A `setsid nohup sleep` escapee is killed by the uid sweep (checked offline as root; the live run
  could not list processes because the image has no `ps`).

**g6 image rebuild: pending.** The hardened farm is not in any deployed g6 image (the 09-25 Runtime was
deleted). It was not rebuilt because the g6 container has zero capabilities, so it would only exercise
the shared-uid fallback that `demo_gpu_single` documents.


## Folder layout (self-contained)
- `agentcore_slime_demo/` — vendored shared library (this demo imports ONLY this copy;
  `/work/code` is not on `PYTHONPATH`). Includes `sandbox_ray_farm.py`, `sandbox_farm_env.py`,
  `sandbox_server_agent.py`, `sandbox.py`.
- `deploy/Dockerfile.sandbox.gpu.shared` — the shared-GPU sandbox image (bakes `ray`, runs
  `python -m agentcore_slime_demo.sandbox_server_agent`); used by the local farm (GPU 6, fractional).
- `1_…6_*.sh` + `deploy/` (`demo.env`, `steps_common.sh`, `local_gpu_runtime.sh` = the farm launcher).
- `build_dataset.py`, `prepare_data.sh`, `data/`, `run.sh`, `metrics.py`, `isolation_probe.py`, `logs/`.

Shared by path (NOT vendored): the base checkpoint (`/work/code/p4_slime_train/ckpt/Qwen3-4B_torch_dist`),
the `agentcore_p4` container and `/root/slime`.
