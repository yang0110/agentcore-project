# demo_cpu_instance_pool: a large sandbox image on CPU Instances, run as a warm pool (`SANDBOX_POOL=N`) shared by per-episode isolated sub-sessions

**The point of this demo:** a sandbox image too big for a microVM runs on AgentCore **capacity-provider
Instances** (EC2 m5). To avoid paying the Instance cold boot on every episode, a small **warm pool** of
Instances is shared by all episodes. Each episode still gets a fresh, isolated environment on the
Instance it lands on.

The harness, reward path and slime args match `demo_cpu_tasks`. This is purely a deployment change.
The default task is `DATASET=deepscaler`, and any `demo_cpu_tasks` dataset works (see `prepare_data.sh`).
This folder was `demo_cpu_tasks VARIANT=bigimage`, and before that `demo_cpu_bigimage`.

## 1. Why an Instance

| | Code Interpreter / Runtime microVM | Runtime on a CPU Instance (this demo) |
|---|---|---|
| Image cap | ~2 GB | instance root disk (`ROOT_FREE_GIB`, default 40 GB) |
| Heavy deps (torch, scipy, sklearn, …) | won't fit in a big stack | baked in (`deploy/Dockerfile.sandbox.bigcpu`, ~4 GB) |
| Compute | AWS-managed microVM | your EC2 family (`CPU_INSTANCE`, default `m5.xlarge`) |
| Session length | ~8 h | 14 days |
| Cold boot | a few s | **~60–83 s** (EC2 launch + 4 GB image pull) |
| Session ↔ compute | one microVM per session | **one Instance per `runtimeSessionId`** |

That last row is the catch. With a fresh `runtimeSessionId` per episode (`SANDBOX=agentcore-runtime`),
every episode boots its own m5. The old `VARIANT=bigimage` smoke in `logs/bigimage/` ran this way:

- about **32 cold boots per step**, one per episode, each around 83 s;
- 375 cold boots over 10 steps;
- sandbox wait ratio above 0.97.

## 2. The warm pool

```
                       ┌─ runtimeSessionId cpupool-<run>-m00 ─► warm m5 #0 ─┬─ sub_session ep-a  (fresh env)
 episodes ─► _pool_acquire ─┤                                               └─ sub_session ep-c
 (least-loaded,        └─ runtimeSessionId cpupool-<run>-m01 ─► warm m5 #1 ─┬─ sub_session ep-b
  round-robin on ties)                                                      └─ sub_session ep-d
```

- **Pool sessions.** `run.sh` mints `SANDBOX_POOL=N` (default **2**) shared runtimeSessionIds,
  `cpupool-<runid>-m00 … m0(N-1)`, and exports them as `B4_SHARED_RUNTIME_SESSIONS`. AgentCore pins
  each id to its own Instance, so the pool is **N warm m5s** and the cold boot is paid **N times per
  run**, not once per episode.
- **Choosing an Instance.** `RayFarmRuntimeSandbox` (`SANDBOX=agentcore-ray-farm`, in
  `agentcore_slime_demo/agentcore_runtime_sandbox.py`) opens each episode on the **least-loaded**
  pool member (`_pool_acquire`, round-robin on ties) and releases the member on close.
- **Per-episode sub_session.** Every op carries a per-episode `sub_session`.
- **The server side.** The server-side farm is `SANDBOX_FARM=cpu` (`sandbox_cpu_farm.py`), baked into
  the Runtime env. The server runs as root in the image. For each `sub_session` it:
  - leases a **uid slot** and runs everything as uid/gid `20000+slot`, with no supplementary groups;
  - creates `/workspace/_farm/ep-<id>` owned by that uid, mode `0700`, under a `0711` farm root;
  - runs every exec as a **fresh subprocess** under `prlimit` and `umask 077`, with `PWD`, `HOME`,
    `TMPDIR` and pip/XDG/HF caches all pointing into that dir;
  - on close, **kills every process of the uid**, deletes the dir and the uid's files in
    `/tmp` / `/var/tmp` / `/dev/shm`, and frees the slot.

**Why `N=2` by default.** One Instance puts every episode of a step on a single m5 (4 vCPU). Two
Instances halve that CPU contention, and the cold boot is still paid only twice per run. Both boots
happen in parallel during the first step. Raise N when checker CPU time binds, for example
`perf/sandbox_ms_p95` climbing with `CONC`. Lower it to 1 to minimise idle cost.

A singular `B4_SHARED_RUNTIME_SESSION`, or an explicit `shared_session_id`, turns the pool off: every
episode then goes to that one Instance, as in the old warm-shared bigimage behaviour.

### Isolation guarantees: per-episode uid on a shared kernel, NOT a VM boundary

The warm pool trades the VM boundary for warm reuse. The farm puts back as much isolation as one kernel
allows: every sub_session is its **own unprivileged uid** (`20000+slot`) with a private `0700` dir.

| | AgentCore microVM per session (VM boundary, guaranteed by AgentCore) | Shared m5 sub_session (what the farm enforces + what is still shared) |
|---|---|---|
| **Filesystem** | own root fs per session, discarded with the microVM | **enforced:** own dir `0700` owned by the episode uid, farm root `0711` (not listable), `umask 077`, HOME / TMPDIR / caches inside the dir; the root server's `read_file` / `write_file` refuse `..`, absolute and symlink escapes; close deletes the dir and the uid's files in `/tmp`, `/var/tmp`, `/dev/shm`. **Shared:** the image's read-only system dirs, and any world-writable path an episode chmods open itself |
| **Process / user** | own process tree and PID namespace | **enforced:** distinct non-root uid/gid per episode, no supplementary groups, so no ptrace, no signals, no `/proc/<pid>/environ` or `cwd` of another episode; the sub_session id stays off every command line; close SIGKILLs **every** process of the uid (setsid'd daemons included) and reaps the zombies. **Shared:** one PID namespace, so `ps` lists other episodes' command lines |
| **RAM / CPU** | own vCPU / memory allocation | **enforced:** `prlimit` `NPROC=256` per uid (a fork bomb stops at its own cap), `FSIZE=1 GiB`, `CORE=0`; per-op wall-clock timeout, which kills the uid's tree. **Shared:** no per-episode cgroup and no address-space cap, so one episode can take the m5's CPU and RAM; an OOM can kill a neighbour or the server |
| **Network** | own network interface per microVM | **enforced:** the server's `AWS_*` env is not inherited. **Shared:** one network namespace. An episode can reach the server's own port `8080`, but acting on another episode there needs that episode's random 64-bit sub_session id, which nothing exposes. It can also reach the instance metadata endpoint and anything the instance can reach |
| **Kernel** | own guest kernel (Firecracker) | **shared:** one kernel for all sub_sessions on the instance. A kernel exploit or local root escalation from one episode breaks **all** of the above |
| **Fault blast radius** | one session | **one instance:** a crash, OOM or disk-full of the server or instance hits every episode on that pool member (`SANDBOX_POOL` of them in parallel) |

**What this means:**

- **Good enough for RLVR, including adversarial-ish policies.** The farm stops the reward hacks a
  policy could stumble into: reading a sibling's `solution.py` or answer file, killing a neighbour's
  checker, or leaving a daemon that tampers with the next episode.
- **Not good enough against a kernel exploit**, or anything that needs its own network stack or
  guaranteed CPU/RAM. For those, use a **microVM per session**: `SANDBOX=agentcore` (Code Interpreter)
  or `SANDBOX=agentcore-runtime` on a microVM Runtime. This demo's image is > 2 GB, so for it that
  means a fresh Instance per episode, paying the cold boot every time.
- **Tasks that mutate the whole OS** (Terminal-Bench style) also need a fresh session per episode.

**Uid isolation needs a root server,** which is the case in the image (no `USER` line). If the server
ran unprivileged, `describe()` reports `uid_isolation: false` and only the dir + env freshness remains.

Check it with `isolation_probe.py`. A plants a secret file, a `/tmp` file and a detached process; B
tries to reach them.

```bash
python3 isolation_probe.py                                     # offline, as root (the agentcore_p4 container)
AGENTCORE_SANDBOX_RUNTIME_ARN=<arn> python3 isolation_probe.py --live   # live, two sub_sessions on ONE warm m5
```

Offline (2026-09-25): **16/16 checks pass**. The checks:

- **uids:** A and B run as uids 20000 and 20001.
- **Listing:** `ls` of the farm root and of A's dir both fail with Permission denied.
- **Reading A's secret:** B gets Permission denied by explicit path and via `../`.
- **Server file ops:** the server's `read_file` / `write_file` refuse `../<A>/marker.txt`.
- **`/tmp`:** A's `/tmp` file is unreadable to B.
- **Signals:** `kill -0` of A's pid is "Operation not permitted".
- **Env and limits:** `AWS_*` is not inherited, and nproc=256 / fsize=1 GiB are in force.
- **After A closes:** A's setsid'd `sleep` is gone ("No such process"), and so is its `/tmp` file.

Also verified offline:

- A 400-way fork loop stops at "Resource temporarily unavailable".
- A 2 GB write stops at 1 GiB ("File size limit exceeded").
- A timed-out exec leaves no process of the uid alive.
- numpy and torch checkers run unchanged as the uid.

<!-- LIVE_ISOLATION -->

Before this hardening (the first live probe, same day), both sub_sessions ran as root: B could `ls ..`
and `cat ../<A>/marker.txt`.

See also [`docs/sandbox-fresh-env-on-persistent-instance.md`](../../../docs/sandbox-fresh-env-on-persistent-instance.md).

## 3. Run it

```bash
python3 probe_bigimage.py                  # offline: image size vs the microVM cap + LocalSandbox contract parity
python3 netcheck.py                        # offline: safe-networking checklist (docs/agentcore-networking.md)
bash 1_prepare_data.sh                     # container: data/deepscaler_{train,eval}.jsonl (DATASET=<name> for another)
bash 2_build_image.sh                      # host: ~4 GB amd64 image
bash 3_push_ecr.sh                         # host: -> ECR acs-sandbox-bigcpu
bash 4_deploy_runtime.sh                   # host: CPU capacity provider acs_cpu_cp_bigimage + Runtime (SANDBOX_FARM=cpu) -> deploy/runtime.env
SANDBOX_POOL=2 bash 5_run.sh               # container: 10 train steps + 1 eval on a 2-Instance warm pool
python3 pool_report.py logs/<tag>.log      # episodes per Instance, cold boots, warm-op p50/p95, reward per member
bash 6_cleanup.sh                          # host: delete the Runtime + the CP (DELETE_CP=0 keeps the CP)
```

- **One Runtime serves the whole pool.** The pool size is purely client-side, so changing
  `SANDBOX_POOL` needs no redeploy.
- **Networking stays pull-only.** The Runtime uses `networkMode=PUBLIC` and IAM
  `invoke_agent_runtime`, with no inbound port and no dial-back.
- **Idle Instances.** Idle pool Instances are reclaimed after the Runtime idle timeout (600 s). The
  next run pays the cold boot again.

## Knobs

| Knob | Default | Meaning |
|---|---|---|
| `SANDBOX_POOL` | `2` | number of warm Instances (shared runtimeSessionIds). `POOL` is an alias **only when `run.sh` is called directly**, because `steps_common.sh` forwards only `SANDBOX_*`-style names into the container |
| `B4_SHARED_RUNTIME_SESSIONS` | minted per run | explicit comma list of pool session ids (reuse Instances across runs) |
| `SANDBOX_POOL_PROBE` (alias `POOL_PROBE_INSTANCE` when calling `run.sh` directly) | `1` | on open, read the Instance `boot_id` → `[demo] instance=i-xxxxxxxx` (one extra ~150 ms exec per episode; `0` = off) |
| `SANDBOX_FARM_UID_BASE` / `SANDBOX_FARM_MAX_SLOTS` / `SANDBOX_FARM_NPROC` / `SANDBOX_FARM_FSIZE_MB` / `SANDBOX_FARM_UID_ISOLATION` | 20000 / 256 / 256 / 1024 / 1 | **server-side** (Runtime env): uid range, concurrent sub_sessions per instance, per-uid process cap, max file size; `0` disables uid switching |
| `SANDBOX` | `agentcore-ray-farm` | the only accepted value (the warm pool) |
| `DATASET` | `deepscaler` | any `demo_cpu_tasks` dataset, or a comma-list mix |
| `HEARTBEAT` | `0` | `1` = ping idle sessions so pool Instances survive the weight-sync barrier on long runs |
| `CPU_INSTANCE` / `ROOT_FREE_GIB` / `FORCE_RECREATE` | `m5.xlarge` / 40 / 0 | deploy: capacity-provider instance type, root disk, recreate CP + Runtime |
| `RB` / `NS` / `NR` / `CONC` / `MODE` | as in `demo_cpu_tasks` | batch, group, rollouts, concurrency, async mode |

## Metrics

On top of the general set (`perf/sandbox_ms_*`, `perf/sandbox_open_ms_*`, `perf/sandbox_coldboot_count`,
`perf/sglang_gen_ms_*`, `rollout/raw_reward`, `eval/<dataset>`), every episode's `[demo]` line carries
the pool fields:

- `pool_size`;
- `pool_slot` (`m0`, `m1`, …);
- `instance` (`i-` + the first 8 hex chars of the boot_id);
- `open_ms`;
- `cold` (1 if the open took ≥ 5 s).

`metrics.py` aggregates the numeric fields per step as `demo/*`. `pool_report.py` prints the
per-Instance split, the cold boots and the warm-op p50/p95.

## Results

**Pool probe** (2026-09-25, `SANDBOX_POOL=2`, MockPolicy, 6 episodes, live Runtime
`acs_sandbox_bigcpu`):

- 2 distinct Instances;
- **2 cold boots**, 83.1 s and 83.6 s (in parallel);
- warm opens 121–328 ms;
- reward 1.0 on all 6.

**RL smoke** (`SANDBOX_POOL=2`, NR=11): see below.

`logs/smoke_pool2_20260925_113657.*`, 2026-09-25, `SANDBOX_POOL=2`, NR=11, the pre-hardening image. The
ray job succeeded (10 steps).

| | this run (pool of 2 warm Instances) | old run (Instance per episode) |
|---|---|---|
| Instances used | **2** (m0 → `i-827f10d4`, 210 ep; m1 → `i-4bb897a8`, 196 ep) | ~375 |
| cold boots | **2** (4 episodes waited 63.1 s, all in step 1); `perf/sandbox_coldboot_count` = 4, 0, 0, … | ~32 per step |
| warm ops | 3169, p50 **117 ms**, p95 261 ms (per-step p50 110–120 ms, p95 227–608 ms) | p50 ~150 ms |
| `perf/step_time` | 38–219 s (step 1: 159 s, including the boot) | ~715–753 s |
| `perf/wait_time_ratio` | 0.50–0.93 | 0.97 |
| `raw_reward` (steps 1–10) | 1.0, 0.81, 0.72, 0.94, 0.91, 0.75, 0.91, 0.72, 0.94, 0.91 | 0.66–0.97 |
| `eval/deepscaler` (step 9) | **0.7875** | 0.7625 |

Per-instance mean reward: m0 0.843, m1 0.832, so neither Instance is a bad host. Checkers: PASSED
n=400 (mean 0.8475) and IMPORT_ERROR n=5 (reward 0). `python3 pool_report.py logs/<tag>.log`
prints the per-instance split.

**Old per-episode-Instance run** (`logs/bigimage/`, 2026-09-21, `SANDBOX=agentcore-runtime`, before the
pool):

- about 32 cold boots per step (375 total);
- warm-op p50 about 150 ms;
- `raw_reward` 0.66–0.97;
- `eval/deepscaler` 0.7625.

## Folder layout

- `agentcore_slime_demo/`: the vendored lib. The pool logic is in `agentcore_runtime_sandbox.py`
  (`_pool_acquire`, `_open_episode`); `harness.py` merges `sandbox.demo_fields` into `[demo]`.
- `1_…6_*.sh`, `deploy/demo.env`, `deploy/steps_common.sh`: the numbered steps (`steps_common.sh`
  is identical across demos).
- `deploy/`: `Dockerfile.sandbox.bigcpu`, `deploy_bigimage.py` (CP + Runtime), `agentcore_deploy.py`.
- `run.sh` (RL run), `metrics.py`, `pool_report.py`, `isolation_probe.py` (cross-episode check), `probe_bigimage.py`, `netcheck.py` (+ samples),
  `prepare_data.sh`, `build_dataset.py`, `data/`.
- `logs/bigimage/`: the pre-pool results (moved from `demo_cpu_tasks`).

Prereqs: the `agentcore_p4` container (repo at `/work/code`), `AWS_REGION=us-west-2` and AWS creds,
the base checkpoint `code/p4_slime_train/ckpt/Qwen3-4B_torch_dist`, and the Qwen3-4B HF snapshot in the
container cache.
