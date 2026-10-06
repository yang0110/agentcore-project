# Terminal-Bench tasks the microVM pool can't run: x86-only, docker-compose, and big images

**Scope.** The microVM-only `demo_cpu_terminalbench` (removed 2026-10-06) ran 218 of the 241 Terminal-Bench
tasks, each on its own AgentCore **serverless microVM** Runtime (`acs_tb_<task>`, arm64, one fresh session
per episode). `demo_terminalbench_kernelbench` now runs the rest on m5 EC2 Docker hosts
([plan](terminalbench-kernelbench-demo-plan.md)). This doc covers
the 23 tasks that are left out (`terminalbench/tasks/manifest.json` + `images.csv`). For each of the
three blockers it says whether a **capacity-provider Instance** (e.g. `m5.xlarge`) or a change to how
the task is packaged fixes it.

Two corrections to the usual wording:

- The microVM image cap is **2 GB** (compressed ECR size, `MAX_IMAGE_MB = 2048` in
  `deploy_terminalbench.py`), not 3 GB.
- Not every "arm64 build failure" is an arm64 problem. Only 2 of the 7 failures are x86-specific.

## 1. What is left out, and why

| Blocker | Tasks | Root cause |
|---|---|---|
| **Multi-container compose** (12) | `ancient-puzzle`, `debug-long-program`, `find-restaurant`, `home-server-https`, `interactive-maze-game`, `postgres-csv-clean`, `security-celery-redis-rce`, `security-vulhub-minio`, `simple-sheets-put`, `simple-web-scraper`, `spring-messaging-vul`, `sql-injection-attack` | `docker-compose.yaml` declares 2–6 services (`client` + sidecars). An AgentCore session is a single container. |
| **x86-only** (2) | `accelerate-maximal-square` | Base `taichidev/taichi:v0.7.26` exists only for `linux/amd64` (`InvalidBaseImagePlatform`). |
| | `reverse-engineering` | Builds `validate.s`, which is x86-64 assembly (`unknown mnemonic 'pushq'` on arm64). |
| **arm64 mirror 404** (2) | `qemu-startup`, `qemu-alpine-ssh` | `debian:bullseye-slim` arm64 security packages return 404. They also boot an **x86** guest with `qemu-system-x86`. |
| **Upstream bit-rot** (3; fails on x86 too) | `fix-git` | `setup.sh` clones a GitHub repo that now needs auth. |
| | `leelachess0-pytorch-conversion` | `training.lczero.org` download fails (wget exit 8). |
| | `word2vec-from-scratch` | `load_dataset('wikitext', …)`: newer `huggingface_hub` requires `namespace/name` (`Salesforce/wikitext`). |
| **Over the 2 GB cap** (4) | `mteb-eval`, `mteb-leaderboard`, `mteb-retrieve` (~6.7 GB), `png-generation` (~6.5 GB) | `pip install mteb` / `easyocr` pulls the **CUDA** torch wheel plus `cuda-toolkit` (~4 GB of GPU libraries), which is dead weight on a CPU sandbox. |

## 2. Constraints that decide the design

1. **The managed microVM fleet is arm64 with a 2 GB image cap.** It has no x86 option and no larger
   size tier.
2. **Instances are EC2 from a capacity provider that you define**: instance type, `LINUX_X86_64`, and
   root-volume size. We ran a ~4 GB amd64 image on `m5.xlarge` on 2026-09-21 (402 episodes, passed).
   The cost:
   - ~60–80 s cold boot per new instance.
   - **1 session = 1 instance.** N concurrent episodes means N instances, so the account's EC2 and
     capacity-provider quotas cap concurrency.
3. **No Docker inside AgentCore, on either substrate.** The container runs as uid 0 inside a user
   namespace with every capability dropped (`CapEff=0`, `NoNewPrivs=1`, seccomp on; measured on the m5
   2026-09-25). `dockerd`, rootless Docker, `unshare`, cgroups and `mount` all fail with `EPERM`. **An
   Instance does not make compose work by itself.** Compose tasks have to be rewritten whichever
   substrate runs them.
4. **No KVM.** Neither substrate exposes `/dev/kvm` (m5 is not a metal instance), so any `qemu-system-*`
   task runs under TCG software emulation. It is slow but works, as it does in upstream Docker without
   KVM.
5. **Sessions can't accept inbound connections from each other.** A session is reached only through
   `InvokeAgentRuntime`. In VPC mode a session can **dial out** (that is how our in-session agent loop
   reaches the adapter over SG→SG). So "client session talks to a sidecar session" isn't possible
   directly. The sidecar has to live somewhere the client can dial.

## 3. x86-only tasks → an x86 Instance Runtime (m5)

**Recommendation.** Deploy these tasks as **Instance Runtimes on an x86 capacity provider** built for
`linux/amd64`. This is the only route that runs the task unchanged.

```
capacity provider  acs_tb_x86_cp   operatingSystem=LINUX_X86_64, allowedInstanceTypes=[m5.xlarge],
                                   rootVolume.freeSpaceGiB = 3-4x the unpacked image, VPC = the TB subnets/SG
runtime            acs_tb_<task>   same image recipe as the microVM pool (task Dockerfile + /opt/acs layer),
                                   but PLATFORM=linux/amd64 and the x86_64 standalone CPython
                                   in acs_ctx/py; capacityProviderConfiguration={cp_arn}
routing            arn_by_image.json  unchanged: image -> ARN. The harness doesn't care which substrate
                                      the ARN is on.
```

What changes in `deploy_terminalbench.py`:

- **Per-task platform.** Set `PLATFORM=linux/amd64` for tasks on an x86 allow-list, or any task whose
  arm64 build fails with `InvalidBaseImagePlatform` / x86 asm. Drop the "try arm64 for
  `FROM --platform=linux/amd64`" override for those tasks.
- **Per-arch standalone Python.** Use the `x86_64-unknown-linux-gnu` build next to the aarch64 one.
- **Instance registration.** Register on the capacity provider instead of the serverless fleet; the
  code to reuse is the old `cpu_capacity_provider()` / `runtime_on_cp()` from `deploy_bigimage.py`, in
  git history. Record `substrate=instance` in `images.csv`.

What changes in the run:

- **One fresh Instance per episode, the same isolation model as the microVM pool.** TB tasks mutate
  the whole OS (apt, services, files), so the warm-shared `sub_session` farm, which only isolates
  workdir/HOME/caches, is **not** sufficient.
- **The boot has to come off the critical path.** ~60–80 s on m5 (vs ~0.6–1.9 s microVM opens) is
  already handled by the look-ahead pre-open (`TB_PREOPEN_BATCHES`). Give Instance tasks a deeper
  look-ahead, or open them first within a batch. Heartbeat is unchanged: an Instance session's idle
  timeout is per-runtime (`RUNTIME_IDLE_TIMEOUT`).
- **Watch the concurrency cost.** A batch containing one x86 task with `NS=8` opens 8 m5s at once.
  Cap concurrent Instance sessions with `SANDBOX_OPEN_RATE` or a per-substrate gate, and size the
  capacity-provider quota for `TB_PREOPEN_BATCHES × NS`.

**Alternatives considered:**
- **qemu-user on the arm64 microVM: rejected.** `binfmt_misc` registration needs privilege (constraint
  3), so x86 binaries can't run transparently, and wrapping every exec in `qemu-x86_64` changes what
  the agent sees.
- **Porting the task to arm64: rejected for `accelerate-maximal-square`.** Pinning a newer multi-arch
  taichi changes the task. It is fine only as a *new* task, not as Terminal-Bench.

**qemu tasks.**
1. First retry on arm64 with a fresh `apt-get update`, or pin `debian:bullseye-slim` to a current
   digest. The 404 is a stale index, not an arch limit.
2. If that fails, use the x86 Instance path above.
3. Either way the guest runs under TCG (constraint 4). Check that `run-tests.sh` timeouts hold with
   `validate_oracle.py` before adding them to the pool.

**Bit-rot tasks** fail on any architecture. Patch the Dockerfile minimally and re-verify the oracle,
or leave them out:
- `word2vec`: `Salesforce/wikitext`.
- `fix-git`: vendor the repo into the build context.
- `leelachess0`: mirror the network file to S3/ECR.

## 4. Big images (>2 GB) → slim first, then an Instance

**Step 1 — slim it (preferred; keeps the cheap, fast microVM).** All 4 oversized images are large for
the same reason: the CUDA build of torch. Pre-install the CPU wheel before the task's own `pip install`:

```dockerfile
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
RUN pip install mteb==1.36.8          # the task's original line; torch is already satisfied
```

This should put each image well under 2 GB, since the CPU aarch64 torch wheel is ~200 MB vs ~4.5 GB of
CUDA wheels plus toolkit. The agent sees the same Python packages without the GPU runtime, which it
could never use on a CPU sandbox anyway.

Gate: `images.csv size_mb < 2048` **and** `validate_oracle.py` passes (the reference solution still
scores 1.0) before the task joins the pool. Inject the line in `deploy_terminalbench.py`'s appended
layer *before* the task's `RUN`, via a per-task pre-layer, so the upstream Dockerfile stays untouched
in `tasks/`.

**Step 2 — if it can't be slimmed, use an Instance Runtime.** Same recipe as §3 (arm64 or amd64 both
work on Instances; pick the instance family to match):
- **Root volume.** Size `rootVolume.freeSpaceGiB` to hold the *unpacked* image: a 6.7 GB compressed
  image is ~15–20 GB on disk, so use 40 GiB as we did for the 4 GB image.
- **Cold start.** Expect the boot to grow by the image pull, roughly 1–3 min for 6–7 GB from same-region
  ECR, on top of the ~60–80 s boot. Pre-open covers it only if the look-ahead is deep enough. Measure
  with `probe_coldstart.py` before the first run.
- **Not tested.** We have only run ~4 GB on an Instance. We have not found an AWS-documented image
  ceiling for Instances; treat >4 GB as something to verify with one deploy.

**Rejected: lazy-loading weights or data from S3 at session open.** It keeps the image small but moves
the download into every episode's open, and needs egress. Use it only for genuinely large *data*, not
packages.

## 5. docker-compose tasks → flatten into one session, or move sidecars out

Compose needs a container runtime, which no AgentCore substrate has (constraint 3). Two designs, chosen
per task by **whether the agent may see the sidecar's internals**.

### 5a. Flatten: all services as processes in the client session (default)

Build one image holding the client **and** the sidecars' software. Add an entrypoint (or a session-open
hook in the pool) that starts each sidecar as a background process before the agent's first turn, and
waits on the compose healthchecks.

| Compose feature | Flattened equivalent |
|---|---|
| `image: python:3.13-slim` + `command: pip install flask && python x.py` | `pip install` at **build** time into a separate venv (`/opt/sidecar/venv`), `python x.py &` at open |
| `image: postgres:16` / `redis` / `minio` | install the same major version from apt/upstream binaries into the image; run as a background daemon |
| service hostname (`db`, `api`, `redis`, `node1..3`) | `/etc/hosts` lines → `127.0.0.1`. Use distinct loopback IPs (`127.0.0.2`, `127.0.0.3`…) when two services share a port (e.g. 3× minio on `:9000`). Bake them into the image, since writing `/etc/hosts` at runtime may fail without caps. |
| `depends_on: condition: service_healthy` | run the compose `healthcheck.test` in a loop at open; fail the open (`infra_error`) if it never passes |
| named volume shared by client + sidecar (`shared_results:/app`) | the same directory; it's one filesystem now |
| `environment:` (`DATABASE_URL=…@db:5432`) | baked as `ENV` in the image (the hostname resolves via the hosts lines) |

Fits: `debug-long-program`, `interactive-maze-game`, `find-restaurant`, `ancient-puzzle` (flask
sidecars); `postgres-csv-clean`, `simple-sheets-put`, `simple-web-scraper` (db/api);
`security-celery-redis-rce` (redis + celery). Works the same on the microVM (if it stays under 2 GB:
postgres + flask is ~300 MB) and on an Instance.

**Trade-off.** No isolation between agent and sidecar: same filesystem, same uid, same PID namespace.
The agent can read `/app/decryptor_server.py`, `ps` the master service, or `kill` the database. Per-uid
separation is impossible (constraint 3). Landlock can hide paths from the agent's own execs, but the
in-session loop runs the agent in the same domain. So:

- **Fine when the sidecar is plumbing** (a DB to clean, an API to call).
- **Breaks the task when the sidecar is the secret or the target.** These are reward-hackable when
  flattened:
  - `ancient-puzzle` (decryptor source holds the answer)
  - `sql-injection-attack` (the vulnerable and master services)
  - `security-vulhub-minio` (credentials in the env of a process the agent can inspect)
  - `spring-messaging-vul`
  - `home-server-https` (the mock API)

  Use 5b for these.

### 5b. Sidecars out: client in AgentCore, sidecars in a per-episode ECS task

Keep the client as the normal per-episode AgentCore session. Run the **non-client services** as one
**ECS (Fargate) task per episode** in the same VPC, from the task's own compose file (one container per
service). With `awsvpc` each ECS task gets its own ENI and private IP, so the services keep their
**original ports**. Two episodes of the same task never collide.

```
pre-open (look-ahead)  ─┬─ open AgentCore session (client image)        ~1 s microVM / ~70 s m5
                        └─ RunTask sidecars (compose → task definition)  ~30-60 s Fargate
open completes          client gets "db/api/node1 → <task ENI IP>" written to its hosts file (or baked
                        ENV with the IP), sidecar healthchecks polled from the client
episode                 agent works in the client; client dials sidecars over the VPC
grade + close           oracle runs in the client; StopTask the sidecars; stop the session
```

- **Network.** SG→SG from the TB session SG to the sidecar SG, scoped to each task's declared ports,
  the same pattern as the adapter hole (`agentcore-networking.md`). No inbound into AgentCore is needed
  (constraint 5): the client always dials out.
- **Isolation.** Real. The agent can only reach the sidecars over the ports the task exposes, which is
  what the upstream compose setup gives it.
- **Pool integration.**
  - `terminalbench_pool.py`'s open/close grows a "sidecar handle", and the look-ahead pre-open
    already hides the extra latency.
  - `image_routing.py` maps the task to (client ARN, sidecar task definition).
  - The reward path is unchanged.
- **Cost.** Fargate per episode (sidecars are small: 0.25–0.5 vCPU). Plus an ECS cluster and a task
  definition per compose task, generated from `docker-compose.yaml` by a small converter. Most of the
  12 use stock public images (`python:3.13-slim`, `postgres`, `redis`, `vulhub/*`); a few need a build
  (`api/`, `server/`, `Dockerfile.api-mock`).
- **Rejected: sidecars as a second AgentCore session.** It isn't dial-able (constraint 5), and would
  need a relay through the trainer host.
- **Rejected: one long-lived EC2 host running `docker compose` per episode.** It works, but needs
  per-episode port remapping, because every episode wants `:5432` on the same IP, and some tests
  hardcode ports. It also puts a docker daemon on the critical path.

## 6. Summary

| Blocker | microVM (arm64, ≤2 GB) | Instance (e.g. m5, x86_64) | Recommended |
|---|---|---|---|
| x86-only (2) | ✗ (no x86, no binfmt) | ✓ native | **Instance, amd64 build** |
| qemu tasks (2) | retry apt; TCG guest | ✓ amd64 build; TCG guest | retry arm64, else Instance |
| bit-rot (3) | patch | patch | patch Dockerfile + re-verify oracle, or drop |
| >2 GB (4) | ✓ **after CPU-torch slim** | ✓ as-is (root volume ≥ 40 GiB, longer boot) | **slim → microVM**; Instance if slimming breaks the oracle |
| compose, plumbing sidecars (~7) | ✓ flattened | ✓ flattened | **5a flatten** |
| compose, secret/target sidecars (~5) | client ✓ + Fargate sidecars | client ✓ + Fargate sidecars | **5b sidecars in ECS** |

An Instance fixes **architecture** and **size**. It does **not** fix compose: no AgentCore substrate
can run a container runtime, so compose tasks need flattening (5a) or external sidecars (5b) either way.
Every route keeps the demo's invariant of one fresh, fully isolated environment per episode, graded by
the task's own hidden oracle.

Suggested order:

1. CPU-torch slim (4 tasks, deploy-script change only).
2. Bit-rot patches (3).
3. Flatten the ~7 plumbing compose tasks.
4. x86 capacity provider (2–4 tasks).
5. ECS sidecars (~5).

Steps 1–3 recover ~14 tasks without new infrastructure.

## Related

- `code/demos/demo_terminalbench_kernelbench/README.md`: the pool, pre-open, heartbeat, and the m5 tier for these tasks.
- `docs/agentcore-networking.md`: SG→SG dial-back checklist (applies to 5b).
- `docs/sandbox-fresh-env-on-persistent-instance.md`: why warm-shared Instances don't fit TB.
- `terminalbench/tasks/images.csv`, `manifest.json`: per-task status and size.
