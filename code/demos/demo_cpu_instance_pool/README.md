# demo_cpu_instance_pool: a large sandbox image on CPU Instances, run as a warm pool (`SANDBOX_POOL=N`) shared by per-episode isolated sub-sessions

**Sandbox tier 3: own Runtime, custom image.** The ≈ 4 GB torch + scientific image is over the 2 GB microVM cap, so it runs on a CPU capacity-provider Instance (m5). See the tier table in [`../README.md`](../README.md#which-sandbox-does-a-task-need-three-tiers).

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
  the Runtime env. For each `sub_session` it:
  - creates `/workspace/_farm/ep-<id>` (mode `0700`, under a `0711` farm root), with `PWD`, `HOME`,
    `TMPDIR` and the pip/XDG/HF caches all pointing into it;
  - runs every exec as a fresh subprocess inside **`sandbox_farm_jail.py`**. That is a **Landlock**
    domain which may write only this dir, open no TCP connection and signal nobody else. A
    subreaper wrapper kills the command's whole process tree when it exits, times out, goes above
    256 processes or goes above 4 GiB RSS;
  - on close, stops its running execs and deletes the dir.

  Where the container keeps `CAP_SETUID`/`CAP_CHOWN` (local docker), it **also** runs each sub_session
  as its own uid (`20000+slot`). **AgentCore drops every capability, so on AgentCore Landlock is the
  boundary** (see below).

**Why `N=2` by default.** One Instance puts every episode of a step on a single m5 (4 vCPU). Two
Instances halve that CPU contention, and the cold boot is still paid only twice per run. Both boots
happen in parallel during the first step. Raise N when checker CPU time binds, for example
`perf/sandbox_ms_p95` climbing with `CONC`. Lower it to 1 to minimise idle cost.

A singular `B4_SHARED_RUNTIME_SESSION`, or an explicit `shared_session_id`, turns the pool off: every
episode then goes to that one Instance, as in the old warm-shared bigimage behaviour.

### Isolation guarantees: a Landlock jail per episode on a shared kernel, NOT a VM boundary

The warm pool trades the VM boundary for warm reuse. The farm puts back as much isolation as one
unprivileged kernel user can.

**What the container allows.** Measured live on the m5, 2026-09-25:

- The server is "root" only inside a user namespace (uid_map `0 100000 65536`).
- `CapEff` = `CapBnd` = 0, `NoNewPrivs=1`, and seccomp is on.
- So `setuid`, `chown`, `unshare` (user/mount/pid/net namespaces) and cgroup writes are all `EPERM`.

**Per-episode uids are therefore impossible on AgentCore.** The first hardened image tried them, and
every farm `open` failed with `chown` `EPERM` (HTTP 500). What the kernel (6.18) *does* allow an
unprivileged process is **Landlock ABI 7**, so each exec runs in its own Landlock domain.
`describe()["mode"]` reports `landlock` on AgentCore and `uid+landlock` where the capabilities exist.

| | AgentCore microVM per session (VM boundary, guaranteed by AgentCore) | Shared m5 sub_session (what the farm enforces + what is still shared) |
|---|---|---|
| **Filesystem** | own root fs per session, discarded with the microVM | **enforced (Landlock):** read+exec only on the system dirs (`/usr /bin /lib /etc /opt /app /proc /sys`), read+write **only its own dir** plus `/dev/null` and friends. Sibling episode dirs, the farm root, `/tmp`, `/var/tmp`, `/dev/shm`, `/workspace` and `/root` are all `EACCES`. HOME, TMPDIR and caches are inside the dir, with `umask 077`. The server's `read_file`/`write_file` refuse `..`, absolute and symlink escapes. Close deletes the dir. **Shared:** the image's read-only system files. `/dev/shm` is shared if you set `SANDBOX_FARM_SHM=allow` (needed by `multiprocessing`) |
| **Process / user** | own process tree and PID namespace | **enforced:** signals, `ptrace` and `/proc/<pid>/{environ,cwd,mem}` are scoped to the episode's own Landlock domain, so another episode's processes return `EPERM`. The sub_session id stays off every command line. The subreaper kills the whole tree, setsid'd daemons included, when each exec ends, so **nothing outlives its exec**. **Shared:** one uid (0 in the userns) for all episodes; one PID namespace, so `ps` shows other episodes' command lines |
| **RAM / CPU** | own vCPU / memory allocation | **enforced (watchdog, 5–100 ms polling):** more than 256 processes in the tree, or more than 4 GiB RSS, kills the tree (exit 137). `FSIZE=1 GiB`, `CORE=0`, plus the per-op wall-clock timeout. **Shared:** no per-episode cgroup, so one episode can take the m5's CPU. A fast allocation spike between polls can still OOM a neighbour or the server |
| **Network** | own network interface per microVM | **enforced (Landlock TCP):** every TCP `connect`/`bind` is denied (`SANDBOX_FARM_NET=deny`). An episode can't reach the server's port 8080, the instance-metadata endpoint or the internet. The server's `AWS_*` env is not inherited. **Shared:** one network namespace. **UDP and DNS are not covered** by Landlock. `SANDBOX_FARM_NET=allow` gives up the TCP rule |
| **Kernel** | own guest kernel (Firecracker) | **shared:** one kernel for all sub_sessions. A kernel exploit breaks **all** of the above, and so does a Landlock bypass |
| **Fault blast radius** | one session | **one instance:** a crash, OOM or disk-full of the server or instance hits every episode on that pool member (`SANDBOX_POOL` of them in parallel) |

**What this means:**

- **Good enough for RLVR, including adversarial-ish policies.** The jail stops the reward hacks a
  policy could stumble into:
  - reading a sibling's `solution.py` or answer file;
  - killing a neighbour's checker;
  - calling the server port to act on another episode;
  - leaving a daemon that tampers with the next episode.
- **Not good enough against a kernel exploit**, or anything that needs its own network stack or
  guaranteed CPU/RAM. For those, use a **microVM per session**: `SANDBOX=agentcore` (Code Interpreter)
  or `SANDBOX=agentcore-runtime` on a microVM Runtime. This demo's image is > 2 GB, so for it that
  means a fresh Instance per episode, paying the cold boot every time.
- **Tasks that mutate the whole OS** (Terminal-Bench style), or that need network access, also need a
  fresh session per episode.
- **Per-exec domains.** Each exec is its own Landlock domain, so a process started in one exec can't
  be signalled by a later exec of the **same** episode. It is also killed when its exec ends. A task
  that starts a server in one call and uses it in the next must do both in one exec.

Check it with `isolation_probe.py`. A plants a secret file, tries `/tmp`, and runs a setsid'd daemon
inside a long exec. Meanwhile B tries to reach all of it.

```bash
python3 isolation_probe.py                                     # offline: CpuFarm in-process
AGENTCORE_SANDBOX_RUNTIME_ARN=<arn> python3 isolation_probe.py --live   # live, two sub_sessions on ONE warm m5
```

**22/22 checks pass in all three settings** (2026-09-25):

- offline in `agentcore_p4` (`uid+landlock`);
- the image under `docker run --cap-drop ALL --security-opt no-new-privileges` (`landlock`, the AgentCore
  simulation);
- **live on the m5** (`landlock`; the 22 checks follow).

**Live checks** (Runtime v3, image `:landlock-20260925`):

- **Same host:** A and B had the same boot_id and both ran as uid 0 in `landlock-abi7`.
- **Filesystem:** B can't list the farm root or A's dir, and can't read A's `marker.txt` by path or
  via `../`, or write into A's dir. All give Permission denied.
- **Server file ops:** the server's `read_file`/`write_file` refuse `../<A>` and A's absolute path.
- **`/tmp`:** A's write to `/tmp` is denied.
- **A's running daemon:** B's `kill -0` of it gives "Operation not permitted", and B's
  `cat /proc/<pid>/environ` gives Permission denied.
- **Env:** `AWS_*` is not inherited.
- **Limits:** a 1.2 GB write stops at 1 GiB, and a 400-process fan-out is killed (`process cap: 401 > 256`).
- **Network:** `curl 127.0.0.1:8080/ping` and `curl 169.254.169.254` return `000` (connect refused).
- **Command lines:** A's sub_session id is in no command line.
- **Lifetime:** A's daemon is gone ("No such process") once A's exec ends.

**Live cost of the jail:**

- one cold boot (83 s: the new Runtime version got a fresh Instance), then warm opens of 214 ms and a
  warm exec p50 of **156 ms**, compared with 117 ms before the jail;
- the jail itself costs about 50 ms per exec (a python wrapper start plus Landlock setup), measured
  offline;
- one MockPolicy episode of **each of the 13 task families** on the same warm m5, all concurrent:
  **reward 1.0 on all 13**.

**Before the hardening:**

- first live probe: both sub_sessions ran as root with no jail, and B could `ls ..` and
  `cat ../<A>/marker.txt`;
- per-uid image (`:uidiso`, Runtime v2): broken on AgentCore by the capability drop above.

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
| `SANDBOX_FARM_NET` / `SANDBOX_FARM_SHM` / `SANDBOX_FARM_LANDLOCK` | `deny` / `deny` / `1` | **server-side** (Runtime env): Landlock TCP rule, `/dev/shm` access (`allow` for `multiprocessing`, then shared), `0` turns the jail off |
| `SANDBOX_FARM_NPROC` / `SANDBOX_FARM_MEM_MB` / `SANDBOX_FARM_FSIZE_MB` | 256 / 4096 / 1024 | **server-side:** per-exec process-tree cap, RSS cap (`0` = off), max file size |
| `SANDBOX_FARM_UID_ISOLATION` / `SANDBOX_FARM_UID_BASE` / `SANDBOX_FARM_MAX_SLOTS` | 1 / 20000 / 256 | **server-side:** the extra per-episode-uid layer, used only where the container has `CAP_SETUID`/`CAP_CHOWN` (**not** on AgentCore); uid range; concurrent sub_sessions |
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
- `run.sh` (RL run), `metrics.py`, `pool_report.py`, `isolation_probe.py` (cross-episode check; the jail is `agentcore_slime_demo/sandbox_farm_jail.py`), `probe_bigimage.py`, `netcheck.py` (+ samples),
  `prepare_data.sh`, `build_dataset.py`, `data/`.
- `logs/bigimage/`: the pre-pool results (moved from `demo_cpu_tasks`).

Prereqs: the `agentcore_p4` container (repo at `/work/code`), `AWS_REGION=us-west-2` and AWS creds,
the base checkpoint `code/p4_slime_train/ckpt/Qwen3-4B_torch_dist`, and the Qwen3-4B HF snapshot in the
container cache.
