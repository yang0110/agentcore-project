# demo_gpu_farm — ONE GPU instance serving MANY sessions (KernelBench CUDA RL on a Ray GPU farm, local or on a g6)

**Sandbox tier 3: own Runtime, custom image.** KernelBench needs CUDA, which only a custom image has: on
AgentCore that means a GPU capacity-provider Instance (g6); on this box, a local farm on GPU 6 (or 6,7).
See the tier table in [`../README.md`](../README.md#which-sandbox-does-a-task-need-three-tiers).

An agentic RL loop: the policy (Qwen3-4B) writes a CUDA `ModelNew`, a torch/CUDA checker runs it, and
the policy reads the failure and fixes it. GRPO rewards the fixes. A Ray farm inside ONE sandbox
runtime owns the farm GPU(s) and leases one long-lived, warm-CUDA actor per episode, so one GPU
instance serves several concurrent sessions. Two knobs pick where the farm lives and how it splits:

| | `FARM=fractional` (default) | `FARM=whole` |
|---|---|---|
| **`LOCAL_GPU=1`** (default): local runtime container, slime on GPUs 0–5 (4 train + 2 infer), sync save | GPU 6 at `B4_GPU_FRACTION=0.5` ⇒ **2 slots** on one L40S | GPUs 6,7 at 1.0 ⇒ **2 slots**, one card each |
| **`LOCAL_GPU=0`**: ONE warm **g6.xlarge** capacity-provider Instance (your custom Runtime), slime on all 8 GPUs (4 + 4), async save | its L4 at 0.5 ⇒ **2 slots** | the whole L4 ⇒ **1 slot** (raise `B4_FARM_GPUS` on a bigger g6) |

Slots = `floor(B4_FARM_GPUS / B4_GPU_FRACTION)`; `run.sh` sets the client open gate
(`B4_FARM_MAX_CONCURRENT`) and `CONC` to it. With `fractional`, two KernelBench episodes compile and
run their kernels on the same card at once. L40S/L4 have no MIG, so each actor caps itself to its share
of VRAM in software (`B4_PER_SANDBOX_MEM_FRACTION`, via an injected `sitecustomize.py`). `whole` removes
GPU memory and fault sharing but still shares the host.

> **Not full isolation, in either mode or site.** All episodes share one kernel, network and host (and,
> with `fractional`, one physical GPU). Trusted checkers only. Full isolation on AgentCore is one session
> per episode — one g6 each (`LOCAL_GPU=0 SANDBOX=agentcore-runtime`). See [Isolation guarantees](#isolation-guarantees).

**History.** Merged 2026-09-28 with `demo_gpu_single` (the remote g6 path, now `LOCAL_GPU=0`, which
had itself absorbed `demo_gpu_snapshot` → `SNAPSHOT_EVERY=1` and `demo_gpu_customruntime` →
`gpu_torch_smoke.py` on 2026-09-25). Merged 2026-09-26 from `demo_gpu_farm_fractional` +
`demo_gpu_farm_whole`; originally `demo_gpu_shared_single` / `demo_gpu_shared_multi` / `demo_gpu_local`.
All ran the same harness, trainer, KernelBench data and image.

| Angle | How | What it shows |
|---|---|---|
| **GPU sharing** (default) | `bash 5_run.sh` (`FARM=fractional`) | two episodes on one card; `FARM=whole` for a card each |
| **Warm-shared remote g6** | `LOCAL_GPU=0` on every step | one g6 + the same in-instance farm; the capacity-provider cold boot is paid once per run, not per episode |
| **Custom Runtime = the only GPU** | `python3 gpu_torch_smoke.py` | the managed Code Interpreter is a CPU microVM (`cuda_available=False`); only your custom Runtime on a g6 has CUDA, through the same `write_file`/`exec`/`read_file` contract |
| **App-level snapshot/resume** | `SNAPSHOT_EVERY=1 bash 5_run.sh` | AgentCore has no VM/GPU snapshot; the harness restores each task's workspace from a store you own at ENTER and snapshots it at EXIT |

## What this demo shows
- **Task:** real [KernelBench](https://huggingface.co/datasets/ScalingIntelligence/KernelBench).
  Train = `level_1` (100 single-op problems); held-out eval = 20 `level_2` fusions, disjoint from train.
- **Reward = GPU checker:** `run_checks.py` rebuilds the reference on the GPU with matched seeds,
  asserts `allclose` on `cuda` and reports a speedup. The failure text feeds the next turn.
- **Agent:** the tool-calling mini_claude loop (`HARNESS=mini_claude`, the only harness) runs **inside
  the episode's sub_session**, next to the checker: `in_session_harness.py` writes `session_runner.py`
  into the session and the loop dials this box's per-engine `AnthropicAdapter` back
  (`ADAPTER_PORT_BASE` 18100..; SG→SG rule on the remote path). A hidden grader scores the result after
  the loop stops. No in-trainer fallback: a session that can't host the loop raises
  `HarnessPlacementError`. `run.sh` adds `--sglang-tool-call-parser qwen25 --sglang-reasoning-parser qwen3`.
- **Harness tier (the default in every demo):** the trainer keeps only `harness_pool.py`:
  `HARNESS_WORKERS` dispatch workers (default RB×NS = 32) pull episodes off one queue; each episode
  takes its farm sub_session first, then the least-loaded SGLang engine, sticky for all its turns.
  With 1–2 farm slots the run is sandbox-bound by design (`[demo] sandbox_wait_ms` > 0, `qdepth` ~0).
- **GPU packing via Ray:** `ray.remote(num_gpus=B4_GPU_FRACTION)(SandboxActor)`; Ray places the
  actors and pins each via `CUDA_VISIBLE_DEVICES`. Each actor has its own CUDA context.
- **One shared `runtimeSessionId` + per-episode `sub_session`:** AgentCore maps
  **1 `runtimeSessionId` → 1 instance**, so this is **not** "N AgentCore sessions on one box." Every
  episode invokes the one runtime under the SAME session id (`B4_SHARED_RUNTIME_SESSION`) and carries
  a per-episode `sub_session` that keys its GPU actor.

| `SANDBOX=` | substrate | `torch.cuda.is_available()` | trains KernelBench? |
|---|---|---|---|
| `agentcore-ray-farm` (default) | the farm: local container, or your custom Runtime on a g6 | **True** | **yes** |
| `agentcore-runtime` (`LOCAL_GPU=0`) | one dedicated g6 per episode (legacy; instance count scales with concurrency) | **True** | yes |
| `agentcore` | managed Code Interpreter, CPU microVM | False | no |

## Snapshot/resume (`SNAPSHOT_EVERY=1`)
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

**Local (default).** Everything on-box: no ECR push, no AWS control-plane calls, no capacity provider.
The farm container runs on the HOST (docker + the nvidia container runtime); the trainer in `agentcore_p4`.

```bash
python3 gpu_torch_smoke.py           # 0. offline: CUDA contract probe;  python3 snapshot_demo.py: snapshot animation
bash 1_prepare_data.sh               # 1. container: verify shipped KernelBench data (100 level_1 / 20 level_2; REBUILD=1 re-downloads)
bash 2_build_image.sh                # 2. amd64 image acs-gpufarm-gpu from deploy/Dockerfile.sandbox.gpu.shared
bash 3_push_ecr.sh                   # 3. skipped (local-gpu)
bash 4_deploy_runtime.sh             # 4. tag -> acs-local-gpu-sbx:fractional, start the farm on GPU 6 at 127.0.0.1:8080
                                     #    (FARM=whole bash 4_deploy_runtime.sh -> GPUs 6,7; then FARM=whole bash 5_run.sh)
bash 5_run.sh                        # 5. container: 10 train steps + 1 eval; logs/<tag>.{log,csv,eval.csv,png}
bash 6_cleanup.sh                    # 6. stop the local farm
```

Step 4 is `deploy/local_gpu_runtime.sh $FARM` (the same as `code/deploy/run_local_gpu_runtime.sh
$FARM`). The image pip-installs the CUDA torch stack on first boot (kept out of the image for the 2 GB
cap), so the first `/ping` can take a couple of minutes; step 4 polls until healthy.

**Remote g6 (`LOCAL_GPU=0`, real AWS $).** Pass `LOCAL_GPU=0` (and the same `FARM`) to every step:

```bash
LOCAL_GPU=0 bash 2_build_image.sh    # 2. same image
LOCAL_GPU=0 bash 3_push_ecr.sh       # 3. -> ECR acs-gpufarm-gpu (deploy/images.env)
LOCAL_GPU=0 bash 4_deploy_runtime.sh # 4. Runtime acs_gpufarm_gpu on the g6 capacity provider acs_gpu_cp -> deploy/runtime.env
LOCAL_GPU=0 bash 5_run.sh            # 5. 10 train steps + 1 eval (container); slime on all 8 GPUs
LOCAL_GPU=0 bash 6_cleanup.sh        # 6. delete the Runtime (DELETE_CP=1 also deletes acs_gpu_cp; the g6 is the real cost)
```

The farm split is baked into the Runtime's env at deploy (`deploy/demo.env` `RUNTIME_ARGS`), so
switching `FARM` on the remote path needs `FORCE_RECREATE=1 LOCAL_GPU=0 FARM=… bash 4_deploy_runtime.sh`.

**Train to convergence:** `STEPS=20 EVAL_EVERY=10 bash train.sh` inside the container (`STEPS` maps
to `NR=STEPS+1`; keep `STEPS × RB ≤ 100`). Convergence wrappers: `../convergence/12_gpu_farm_remote.sh`,
`13_gpu_farm_fractional.sh`, `14_gpu_farm_whole.sh`, `18_gpu_farm_snapshot.sh`.

## Knobs (defaults in `run.sh`)

| Knob | Default | Meaning |
|---|---|---|
| `LOCAL_GPU` | `1` | `1` = local farm (slime 0–5, sync save); `0` = remote g6 (slime 0–7, async save + persistent ckpt worker) |
| `FARM` | `fractional` | `fractional` / `whole`; must match the farm (local container mode, or the Runtime's deploy-time env) |
| `SANDBOX` | `agentcore-ray-farm` | the farm; `agentcore-runtime` (remote only) = one dedicated g6 per episode |
| `AGENTCORE_SANDBOX_RUNTIME_ARN` | from `deploy/runtime.env` | the g6 Runtime; required with `LOCAL_GPU=0`, ignored with `LOCAL_GPU=1` |
| `B4_FARM_GPUS` / `B4_GPU_FRACTION` | from `LOCAL_GPU` × `FARM` (table above) | farm math; retune BOTH sides together |
| `B4_FARM_MAX_CONCURRENT` / `CONC` | the slot count | client open gate / `--sglang-server-concurrency`; must not exceed the server's slots |
| `NR` / `RB` / `NS` / `TEMP` | 11 / 4 / 8 / 1.0 | 10 train + 1 eval; keep `NR × RB ≤ 100` |
| `EVAL_INTERVAL` / `EVAL_NS` / `EVAL_TEMP` | 10 / 2 / 0.2 | held-out eval cadence / samples / temperature |
| `HARNESS_WORKERS` | empty → RB×NS | harness tier: dispatch workers; each episode picks the least-loaded engine |
| `INVOKE_MAX_RETRIES` | 20 | ≈ 10 min patience on transient invoke errors (one invoke per tool op) — do not lower |
| `B4_SHARED_RUNTIME_SESSION` | auto `b4farm-<32 hex>` | pin it to reuse a warm farm / g6 across runs |
| `SNAPSHOT_EVERY` / `SNAPSHOT_DIR` / `SNAPSHOT_S3_BUCKET` / `SNAPSHOT_PATHS` / `RECYCLE_AFTER` | 0 / `snapshots/` / unset / `progress.json,work.log,solution.py` / 0 | the snapshot/resume angle (above) |
| `HEARTBEAT` / `HEARTBEAT_IDLE` / `HEARTBEAT_PING_FRACTION` | auto / 600 / 0.5 | auto = **on for `LOCAL_GPU=0`** (a `SessionKeeper` pings the warm g6 session idle past `fraction × idle` s so it survives the weight-sync stall), **off locally** (the local farm has no idle timeout) |

Checkpoints go to `ckpt/Qwen3-4B_kernelbench_{farm|remote}_${FARM}[_snapshot]_async`.

> **Keep the client OPEN gate == the server slot count.** `B4_FARM_MAX_CONCURRENT` (client) and
> `B4_MAX_SANDBOXES = floor(B4_FARM_GPUS / B4_GPU_FRACTION)` (server) must agree. If you retune one,
> retune the other.

### Pack tighter
Lower `B4_GPU_FRACTION` (and `B4_PER_SANDBOX_MEM_FRACTION`) on BOTH sides, e.g. 0.25 ⇒ 4 slots on
GPU 6:
```bash
B4_GPU_FRACTION=0.25 B4_PER_SANDBOX_MEM_FRACTION=0.25 bash 4_deploy_runtime.sh
B4_GPU_FRACTION=0.25 bash 5_run.sh
```
Watch for the non-MIG wedge (more co-tenants ⇒ more nvcc load on one card) and the shared host RAM
(the g6.xlarge has only 16 GB; see Isolation guarantees). To give each episode its own card, use `FARM=whole`.

## Key metrics
`rollout/raw_reward` (the RL signal) and `eval/kernelbench` (held-out solve rate). Latency:
`perf/sandbox_ms_{mean,p50,p95}` (checker round-trips) vs `perf/sglang_gen_ms_{mean,p50,p95}`, plus the
cold/warm open split (remote). `logs/<tag>.demo.csv` records per episode which actor
(`b4-sbx-0`/`b4-sbx-1`) served it, its `cuda_visible`, `mem_fraction` and the lease wait.

## Results
Local runs (`LOCAL_GPU=1`):

| run | config | outcome | `eval/kernelbench` | raw_reward by step | sandbox latency | notes |
|---|---|---|---|---|---|---|
| `FARM=fractional` 2026-09-21 smoke (as `demo_gpu_shared_single`) | NR=11, slime 0–5 | Job succeeded, ckpt iter 10 | **0.775** | — | — | artifacts in `code/demos/_smoke_logs/` under the old name |
| `FARM=fractional` 2026-09-25 smoke `logs/smoke-local-fractional_20260925_095218.*` | NR=11, env override `SLIME_CUDA_VISIBLE=2,3,4,5 ACTOR_GPUS=2 ROLLOUT_GPUS=2` (GPUs 0,1 were busy); farm on GPU 6 | Job succeeded, 10 steps + eval, ~83 min | **0.65** | 0.72 0.59 0.22 0.69 0.72 0.19 0.72 0.75 0.63 0.44 | exec p50 9.0 s / p95 19.8 s; write 4.3 ms; open 4.2 ms; close 3.6 ms (gen p50 2.2 s / p95 5.8 s) | 359 episodes, split 181/178 across `b4-sbx-1`/`b4-sbx-0`, both on host GPU 6 at `mem_fraction=0.5`, lease wait always 0; checker PASSED 325 (mean reward 0.64), IMPORT_ERROR 18, NO_OUTPUT 16 |
| `FARM=whole` 2026-09-19 `logs/whole/run_20260919_033018.whole-gpu_local.log` (as `demo_gpu_local`) | NR=11, slime 0–5, farm 6,7 | **Job failed** after step 1 (raw_reward 0.69) | — | — | — | the farm container died mid-run (`Connection refused` on 127.0.0.1:8080, 84×): likely the Ray-GCS failure mode that `local_gpu_runtime.sh` now absorbs with `--shm-size` + `--restart=on-failure` |
| `FARM=whole` 2026-09-21 smoke (as `demo_gpu_shared_multi`) | NR=11, slime 0–5, farm 6,7 | Job succeeded, ckpt iter 10 | **0.65** | — | — | artifacts in `code/demos/_smoke_logs/` under the old name |
| `FARM=whole` 2026-09-21 smoke (as `demo_gpu_local`) | same config | Job succeeded, ckpt iter 10 | **0.70** | — | — | second run of the identical config |

Remote runs (`LOCAL_GPU=0`, recorded as `demo_gpu_single`; artifacts in `logs/remote/`):

| run | config | outcome | `eval/kernelbench` | raw_reward by step | latency | notes |
|---|---|---|---|---|---|---|
| 2026-09-18 `logs/remote/smoke_run.log` | earlier smoke (config in the log) | Job succeeded | 0.15 | — | — | pre-merge artifact |
| 2026-09-25 remote smoke `logs/remote/smoke-remote_20260925_113622.*` | NR=11, g6 Runtime (image baked `B4_GPU_FRACTION=0.5` ⇒ 2 actors on the L4, client gate 1), env override `SLIME_CUDA_VISIBLE=2..7 ACTOR_GPUS=4 ROLLOUT_GPUS=2` (GPUs 0,1 busy) | **Job failed at step 5** (rc=1), no eval | — | 0.53 0.47 0.25 0.56 | cold open 207 s (one `Insufficient EC2 capacity g6.xlarge`, retried); warm open p50 153 ms / p95 224 ms; write p50 119 ms; exec p50 5.4 s / p95 17.0 s; close 110 ms (gen p50 2.1 s / p95 6.6 s) | 136 episodes (69/68 across `b4-sbx-0/1`); checker PASSED 93 (mean 0.71), SETUP_ERROR 27, IMPORT_ERROR 16. Cause: host-RAM OOM, one `run_checks.py` at 12.09 GB on the 16 GB g6.xlarge, Ray killed both actors, next exec HTTP 500 (see Isolation guarantees) |
| 2026-09-26 smoke-all `logs/remote/smoke_all_20260926_144840.*` | smoke-all driver, g6 Runtime, slime on GPUs 2–7 | Job succeeded | **0.725** | 0.66 0.66 | — | |
| 2026-09-28 smoke-all `logs/remote/smoke_all_20260928_034006.*` | same | Job succeeded | **0.70** | 0.63 0.69 | — | |

The evals are 1 eval × 20 problems × 2 samples on smoke-sized runs — noise, not a trend. The 09-25
failure exposed a mismatch that the merge fixed: the remote `run.sh` gated the client to 1 slot while
the Runtime baked `B4_GPU_FRACTION=0.5` (2 actors). The gate is now derived from the same `FARM` math
on both sides.

## Isolation guarantees

All episodes run in ONE runtime (one shared `runtimeSessionId`, per-episode `sub_session`). The left
column is what you would get with one AgentCore session per episode (the VM boundary;
`LOCAL_GPU=0 SANDBOX=agentcore-runtime`). Where the two sites differ, the row says so: the local
container runs as docker root with CHOWN/KILL/SETGID/SETUID, while the AgentCore g6 container runs as
uid 0 with **zero capabilities** (CapEff/CapPrm/CapBnd = 0, NoNewPrivs=1, seccomp; probed live 2026-09-25).

| | session per instance (VM boundary) | shared sub_session (this demo) |
|---|---|---|
| **Filesystem** | Separate instance and disk; nothing shared. | Each episode gets `…/actor-<slot>/<sub_session>/`; the server's `read_file`/`write_file` are realpath-confined to it (`../`, absolute and symlink escapes are refused); `rmtree`d on close. **Local:** owned by the slot's uid, mode 0700 under 0711 roots, so a neighbour can neither read nor list it. **g6:** all slots share one uid (next row), so a command in the sandbox can read a co-resident episode's files. Still shared everywhere: the image FS, world-writable `/tmp` and `/dev/shm`, and the HOME/caches (below). |
| **Process / user** | Separate kernel, PID space and users. | **Local:** each slot runs as its own non-root uid `20000+slot` (umask 077), so neighbours can't be signalled or ptraced; on close the farm SIGKILLs the exec's process group and every process of the uid (catches `setsid`/`nohup` escapees). Same PID namespace: `ps` shows every slot's command lines. **g6:** setuid is impossible, so the farm falls back to one shared uid (`uid_isolation: false`); only the process-group kill remains, and a `setsid` escapee survives close. |
| **GPU memory** | Whole GPU per instance. | Separate CUDA contexts per slot with **driver-enforced address spaces** (one slot cannot read another's device memory). The per-process **fraction cap is cooperative**: `sitecustomize.py` calls `torch.cuda.set_per_process_memory_fraction`, which only limits torch's caching allocator. A checker that raises the fraction, or allocates through raw CUDA / another library, can take the whole card and OOM its co-tenant. |
| **GPU fault blast radius** | One GPU per VM; an Xid or wedge stays in that instance. | **Shared.** There is no MIG on L4/L40S, so an Xid, a hung kernel or a wedged context hits every co-tenant on that GPU. We saw the g6's L4 wedge under sustained nvcc load with fractional sharing; a lower packing (FRAC=0.5) made it rarer but does not remove it. |
| **Compute interference** | None (dedicated). | Time-sliced SMs with no MPS or priorities: a heavy kernel slows the other slot's kernels and skews its measured speedup (reward noise). CPU for nvcc is shared too. |
| **RAM / CPU** | Per instance (16 GB on g6.xlarge). | **Shared, no per-slot limit.** Observed 2026-09-25 on the g6 (`logs/remote/smoke-remote_20260925_113622.log`): one `run_checks.py` grew to 12.09 GB of 16 GB, Ray's memory monitor killed BOTH farm actors, the next exec returned HTTP 500 and the job failed at step 5. A per-slot `RLIMIT_AS` / cgroup `memory.max` would contain it; not implemented (and cgroup writes are unavailable inside the zero-capability Runtime). |
| **Network** | Own instance network stack; VPC/SG policy is set per Runtime. | **Shared network namespace.** Slots can connect to each other's listening ports and to the farm server itself on `localhost:8080`. Sub_session ids are 128-bit random and not listable, so driving another episode needs a leaked id, but nothing else stops it. Blocking uids `20000+` from `:8080` (iptables owner match) would close this; not done. |
| **Kernel** | Separate guest kernel (the VM boundary). | **Shared kernel** (and, on the g6, one container with one seccomp profile). A kernel or NVIDIA-driver exploit crosses slots. uid separation is ordinary Unix DAC, not a security boundary against a determined attacker. |

**Caches and poisoning.** By default each slot uid has its own HOME under `/var/tmp/b4-home/uid-<N>` (0700), with `XDG_CACHE_HOME` and `TORCH_EXTENSIONS_DIR` inside it, so compiled torch extensions are reused warm across episodes. The risk: episodes that later run on the SAME slot share that cache, so one episode can plant a poisoned `.so` or pip/pycache entry that a later episode loads. `SANDBOX_ISOLATE_HOME=1` removes it: HOME and every cache move into the per-episode dir (`sandbox_farm_env.fresh_env(isolate_caches=True)`) and are wiped on close, at the cost of recompiling every extension per episode. The nvcc serialization lock `/dev/shm/b4-nvcc.lock` is created 0644 by the actor and opened read-only by the checker (flock works on a read-only fd); if it cannot be opened the build just runs unserialized. On the g6 every slot shares one HOME, so the risk spans all episodes.

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
deleted). The next `LOCAL_GPU=0 bash 2_build_image.sh` + `4_deploy_runtime.sh` picks it up; given the
measured zero capabilities it can only show the shared-uid fallback there.

## Folder layout (self-contained)
- `agentcore_slime_demo/` — vendored shared library (this demo imports ONLY this copy;
  `/work/code` is not on `PYTHONPATH`). Includes `sandbox_ray_farm.py`, `sandbox_farm_env.py`,
  `sandbox_server_agent.py`, `sandbox.py`, `harness_pool.py`, `app_snapshot.py`, `snapshot_store.py`,
  `session_keeper.py`, `megatron_init.py`.
- `deploy/Dockerfile.sandbox.gpu.shared` — the farm-capable GPU sandbox image (bakes `ray`, runs
  `python -m agentcore_slime_demo.sandbox_server_agent`), used at both sites; `Dockerfile.sandbox.gpu` =
  the older flat server (non-farm).
- `1_…6_*.sh` + `deploy/` (`demo.env`, `steps_common.sh`, `agentcore_deploy.py`, `local_gpu_runtime.sh` = the local farm launcher).
- `run.sh` (trainer), `train.sh` (convergence wrapper), `build_dataset.py`, `prepare_data.sh`, `data/`, `metrics.py`.
- Offline probes: `gpu_torch_smoke.py`, `snapshot_demo.py` (+ `agentcore_idle_sim.py`), `isolation_probe.py`.
- `logs/` (local runs), `logs/whole/`, `logs/remote/` (the former `demo_gpu_single` runs), `logs/snapshot/`
  (the former `demo_gpu_snapshot`).

Shared by path (NOT vendored): the base checkpoint (`/work/code/p4_slime_train/ckpt/Qwen3-4B_torch_dist`),
the `agentcore_p4` container and `/root/slime`.
