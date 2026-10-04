# Plan: one demo that runs all 241 Terminal-Bench tasks (g6e trainer + AgentCore microVMs + an m5 Docker host)

**Goal.** Build a new demo, `demo_hybrid_terminalbench`, that trains on the **whole** Terminal-Bench
corpus (241 tasks). It uses three machines:

- **The g6e.48xlarge we run on today** (`i-REDACTEDc7a41968`, PRIVATE-IP-f99864bc, us-west-2b, 8×L40S) is
  the trainer.
- **The existing ~218 AgentCore microVM Runtimes** run every task that fits there.
- **A new plain-EC2 m5 Docker host** runs the 23 tasks that can't run on a microVM: compose, x86-only,
  big images, and the bit-rot tasks once patched.

The background is in `terminalbench-unsupported-tasks.md` (why the 23 are left out) and
`docker-compose-on-aws.md` (Docker on plain EC2).

`demo_cpu_terminalbench` stays as it is. The new demo is a self-contained copy (own vendored lib,
`deploy/`, step scripts) that adds a second sandbox backend.

## 1. What runs where

| Tasks | Count | Substrate | Why |
|---|---|---|---|
| Single-container, arm64, ≤2 GB | 218 | **AgentCore microVM** (existing `acs_tb_*` Runtimes, unchanged) | already deployed and passing; ~1 s pre-opened open, one kernel per episode |
| Multi-container compose | 12 | **m5 Docker host**, one `docker compose -p` project per episode | needs a container engine; AgentCore has none |
| x86-only (`accelerate-maximal-square`, `reverse-engineering`) | 2 | m5 | amd64 base image / x86 asm |
| qemu (`qemu-startup`, `qemu-alpine-ssh`) | 2 | m5 | amd64 build avoids the arm64 mirror 404; guest under TCG (no KVM) |
| Over 2 GB (`mteb-*` ×3, `png-generation`) | 4 | m5 | run as upstream builds them, no slimming, so the task is unchanged |
| Bit-rot (`fix-git`, `leelachess0-…`, `word2vec-…`) | 3 | m5 after a minimal patch | broken upstream on any architecture |
| **Total** | **241** | | |

The m5 host takes every task the microVM pool can't, and runs it as upstream Terminal-Bench does
(`docker compose`). Each m5 task gets **one substrate only**. Slimming the big images onto the microVM
(the `terminalbench-unsupported-tasks.md` §4 idea) becomes an optional cost optimisation later. It
isn't needed for coverage.

## 2. Architecture

```
g6e.48xlarge (trainer, PRIVATE-IP-f99864bc)                       us-west-2, same VPC
┌───────────────────────────────────────────────┐
│ slime (Megatron + SGLang), 8×L40S              │
│ AnthropicAdapter per engine :18100-18163  ◄────┼────────── agent loops dial back (SG→SG)
│ TerminalBenchPool                              │
│   row.image → route  (strict, no fallback)     │
│   ├─ "arn:aws:bedrock-agentcore:…" ────────────┼──► AgentCore microVM session (218 tasks)
│   └─ "docker://m5/<task>"  ────────────────────┼──► m5 host agent :18300 ─► docker compose -p ep_<id> up
└───────────────────────────────────────────────┘        client container: acs_http_server :8080
                                                          published on <m5-ip>:<random port>
                                                          + sidecars (db, redis, api …) on the project network
```

**Key idea: one protocol, two transports.** The microVM image is the task's Dockerfile plus a layer
holding a standalone Python and `acs_http_server.py`. That server answers the same JSON ops
(`ping`/`exec`/`write_file`/`read_file`/`put_tar`) on `:8080` whether AgentCore invokes it or a
plain HTTP client calls it directly. So on the m5:

- The compose **`client`** service runs the *same wrapped image* (built for amd64). Its
  `command: sleep infinity` is overridden to start the server.
- The trainer talks to it with a new `DockerComposeSandbox`. It has the same interface as
  `AgentCoreRuntimeSandbox`, but POSTs to `http://<m5-ip>:<port>/invocations` instead of calling
  `InvokeAgentRuntime`.
- Everything above the sandbox stays unchanged, because it only sees that interface:
  - the in-session agent loop (`in_session_harness.py` / `session_runner.py`)
  - the hidden-oracle grader (`grade_episode`, `put_tar` into `/tests`)
  - the reward, the metrics, and pre-open

## 3. Components to build

### 3.1 m5 Docker host (new infra)

- **Instance.** `m5.8xlarge` (32 vCPU / 128 GB), Amazon Linux 2023 x86_64, us-west-2b (same AZ as the
  g6e).
  - Root volume: 500 GB gp3, enough for all 23 projects' images (the mteb/png images are ~6.5 GB
    compressed, ~15–20 GB unpacked each) plus build cache.
  - Upgrade to `m5.12xlarge` if the qemu tasks (TCG is CPU-heavy) starve their neighbours.
- **Network.**
  - Subnet with NAT egress, because `run-tests.sh` installs uv + pytest from the internet. Reuse the
    `acs-egress` subnets, or a private subnet behind the existing NAT.
  - No public IP.
- **Security groups.**
  - `m5-sg` inbound from `trainer-sg`: TCP 18300 (host agent) and 32768–60999 (Docker's published
    ports).
  - `trainer-sg` inbound from `m5-sg`: TCP 18100–18163 (adapter dial-back), the same rule pattern as
    the AgentCore runtime SG (`agentcore-networking.md`).
- **Hardening.** Policy-model code runs as root in the client containers:
  - **No IAM instance role.** If one is needed for ECR, scope it to `ecr:GetAuthorizationToken` +
    pull from `acs-tb-*` only.
  - Block IMDS from containers: `iptables -I DOCKER-USER -d 169.254.169.254 -j DROP`.
  - Never mount the Docker socket into a task container.
  - No `--privileged` unless a task's own compose file asks for it (audit the 12).
- **Software.** Docker Engine + compose plugin (`docker-compose-on-aws.md` §1), and the host agent
  (3.2) as a systemd unit.
- **Provisioning.** `deploy/m5_host.py` (boto3): create the SGs and rules, launch from a user-data
  script, wait for the host agent's `/health`, and write the host's private IP to
  `deploy/demo.env` (`DOCKER_HOST_IP`). Cleanup terminates it.

### 3.2 m5 host agent (`m5_agent/compose_agent.py`, stdlib HTTP, port 18300)

| Op | Does | Returns |
|---|---|---|
| `up {task, episode}` | `mkdir /srv/ep/<episode>/{logs,agent-logs}`, then `docker compose -p ep_<episode> -f tasks/<task>/docker-compose.yaml -f acs-override.yaml up -d --wait` with the `T_BENCH_*` vars set | `{host, port, workdir, boot}`; `port` is the client's published `:8080` |
| `down {episode}` | `docker compose -p ep_<episode> down -v --remove-orphans --timeout 5`, then `rm -rf /srv/ep/<episode>` | `{ok}` |
| `list` / `gc {keep}` | lists `ep_*` projects; tears down any not in `keep` (crash recovery after a trainer restart) | |
| `health` | Docker daemon up, free disk, running-project count | |

- **`acs-override.yaml`** is one file for all tasks. It applies only to `client`:
  - `command: ["/opt/acs/py/bin/python3", "/opt/acs/acs_http_server.py"]`
  - `ports: ["8080"]`: a random host port, so concurrent episodes never collide.
  - The image tag for the wrapped image.

  Sidecars are untouched.
- **The `T_BENCH_*` variables** that upstream compose files use (`T_BENCH_TASK_DOCKER_CLIENT_IMAGE_NAME`,
  `…_CONTAINER_NAME`, `T_BENCH_TEST_DIR=/tests`, `T_BENCH_TASK_LOGS_PATH`, …) are set per episode.
  Set `container_name` to `ep_<episode>_client`, which is unique.
- **Concurrency gate.** `MAX_PROJECTS` (default 16 on m5.8xlarge). `up` blocks or rejects beyond it,
  so the m5 can't be over-committed by pre-open.
- **Task files on the m5 exclude `tests/`.** Sync `tasks/<task>/` without the oracle, as on the
  microVM path. The grader injects the tests from the trainer after the agent stops, so the agent
  can't read them.

### 3.3 Images for the m5 tasks

- **Fetch.** `fetch_terminalbench.py` gains `TB_INCLUDE_MULTI=1` (default on in this demo), so the 12
  compose tasks join `tasks/` and the manifest gains `substrate` per task.
- **Build** on the m5 itself: native amd64, no cross-build, layers cached where they run.
  - `deploy_terminalbench.py --substrate docker` builds the task's `client` image plus the same
    `/opt/acs` layer, using the **x86_64** python-build-standalone next to the aarch64 one in
    `acs_ctx/py`.
  - Sidecar images are pulled or built by `docker compose build/pull`.
  - No ECR push is needed (the m5 is the only consumer). Optionally push to `acs-tb-<task>` for
    rebuild-free host replacement.
- **Bit-rot patches** go in per-task pre-layers in `terminalbench/patches/<task>.Dockerfile.pre` (or a
  sed patch), so the upstream `tasks/` files stay pristine:
  - `word2vec`: `Salesforce/wikitext`.
  - `fix-git`: vendor the repo into the build context.
  - `leelachess0`: mirror the network file to S3, fetched at build time.
- **Pool map.** `arn_by_image.json` becomes `route_by_image.json`, mapping
  `{task: "arn:…" | "docker://m5/<task>"}`. `images.csv` gains `substrate` and `host`.

### 3.4 Trainer side (vendored lib in the new demo)

| File | Change |
|---|---|
| `docker_compose_sandbox.py` (new) | `DockerComposeSandbox` with the same interface the pool uses: `__aenter__` (`up` on the host agent, then a `ping` on the client server, filling in `open_reply`/`boot`/`open_*_epoch`), `exec`, `write_file`, `read_file`, `_invoke` (put_tar), `__aexit__` (`down`). HTTP via `aiohttp` |
| `terminalbench_pool.py` | `_open_session(route)` picks the class by URI scheme (`arn:` vs `docker://`). Per-substrate open semaphore: microVM keeps `SANDBOX_OPEN_RATE`, docker uses `DOCKER_MAX_PROJECTS`. Heartbeat registers only microVM sessions, since docker sessions have no idle reclaim. Count docker pre-opens against `MAX_PROJECTS` so the look-ahead can't exhaust the m5 |
| `image_routing.py` | accept both route kinds; still strict (unmapped task = error, no fallback) |
| `metrics.py` / sessions.csv | add a `substrate` column; `open_ms`, solve rate and grade time per substrate; `[demo] substrate=… m5_projects=…` |
| `run.sh` / `deploy/demo.env` | `DOCKER_HOST_IP`, `DOCKER_AGENT_PORT=18300`, `DOCKER_MAX_PROJECTS=16`, `TB_SUBSTRATES=microvm,docker` (either can be switched off for ablations) |
| `1_prepare_data.sh` | rows for all routed tasks. Keep the old train/eval split for the 218, and assign the 23 new tasks to train/eval with the same rule (by task). Report eval as both **"218-pool"** (comparable to `demo_cpu_terminalbench`) and **"all-241"** |

Unchanged: the in-session harness, the adapter, `grade_episode`, the fully-async driver, and the
look-ahead source.

## 4. Isolation, per substrate

| | microVM | m5 compose project |
|---|---|---|
| Unit per episode | one session = one microVM | one `docker compose -p` project (client + sidecars), destroyed with `down -v` |
| Kernel | own kernel | shared host kernel (namespaces + cgroups) |
| Between episodes | nothing shared | images only (read-only layers); networks, volumes and containers are per project |
| Agent vs sidecars | n/a | real container boundary; agent reaches sidecars only over the compose network, as upstream |

The m5 is weaker than a microVM (shared kernel) but **identical to how Terminal-Bench itself is
scored**. Put this in the deck's isolation slide. The hardening in 3.1 (no role, IMDS blocked) limits
the damage of a container escape to a disposable host.

## 5. Latency and capacity

- **Open.** A compose project with prebuilt local images starts in ~2–10 s (sidecar healthchecks
  dominate; qemu tasks boot a guest, ~30–90 s under TCG). Pre-open already hides this:
  `TB_PREOPEN_BATCHES=2` applies to both substrates.
- **Mix.** Mix rate is 23/241 ≈ 10% of rows. With the default draw (RB=4 groups × NS=8), most batches
  have 0–1 m5 groups (8 episodes). Two m5 groups in flight plus their pre-opens ≈ 16–32 projects, so
  `DOCKER_MAX_PROJECTS=16` on m5.8xlarge plus the gate is enough. Episodes beyond the gate wait,
  rather than fail.
- **Memory.** The mteb tasks load models (~2–4 GB RSS each). Set `mem_limit` per client in the override
  (e.g. 6 GB) so one task can't OOM the host.
- **Cost.**
  - m5.8xlarge: ~$1.54/h on-demand, plus 500 GB gp3 at ~$40/month.
  - Stop it between runs: images and cache survive on EBS.

## 6. Phases and gates

| Phase | Work | Gate (must pass to continue) |
|---|---|---|
| **P0 Infra** | `deploy/m5_host.py`: SGs, instance, Docker, IMDS block, host agent | from the g6e: `curl m5:18300/health` ok; a hello compose project's published port reachable; from a test container, `curl g6e:18100` reachable and `169.254.169.254` blocked |
| **P1 Images** | fetch with multi; build the 23 on the m5 (patches for 3) | `images.csv`: all 23 `built` with `substrate=docker`; build errors triaged |
| **P2 Oracle** | `validate_oracle.py --substrate docker`: run each task's reference `solution.sh`, then the hidden tests | each of the 23 scores 1.0 with the solution and 0 with no solution (proves the oracle isn't trivially passing). Drop any task that fails, and record why |
| **P3 Code** | `DockerComposeSandbox`, routing, pool substrate switch, gates, metrics | unit test: open/exec/write/read/put_tar/close against one compose task; `TB_TASKS=<one docker task>` 1-step run gives `infra_error=0` |
| **P4 Smoke** | (a) `TB_SUBSTRATES=docker`, 23 tasks, 3 steps; (b) both substrates, all 241, 10 steps (same NR/EVAL as `5_run.sh`) | 0 infra errors, no leaked `ep_*` projects after the run (`list` empty), no leaked AgentCore sessions; m5 tasks show non-zero turns/tool calls |
| **P5 Report** | README (Use case / Why this demo / Design considerations), deck slide set, per-substrate metrics figure | numbers come from the P4 logs |

Only one `run.sh` may run at a time on the g6e, so don't run P4 alongside a `demo_cpu_terminalbench`
run.

## 7. Risks

| Risk | Mitigation |
|---|---|
| Some compose tasks hardcode container names, host ports, or `network_mode: host` | audit the 12 compose files in P1. Rewrite collisions in the override (unique `container_name`, drop host `ports:` on sidecars). If a task needs fixed host ports, mark it `DOCKER_EXCLUSIVE=1` (the agent runs one episode of it at a time) |
| Security tasks (`security-vulhub-minio`, `security-celery-redis-rce`, `spring-messaging-vul`, `sql-injection-attack`) run deliberately vulnerable services | they live only on the per-project network, with no published ports except the client's; the m5 has no IAM role. Consider a separate `m5-sg` with egress allowed only to NAT, ECR and the g6e |
| qemu under TCG is slow; the test timeout trips | measure in P2; raise that task's `max_test_timeout_sec` only if the oracle needs it, and record it |
| m5 restarts or the trainer crashes, leaving projects running | host agent `gc` on pool start (keeps none); systemd restarts the agent; `6_cleanup.sh` also cleans the m5 |
| Docker images age (base-image tags move) | build once; tag with the build date; rebuild only on `TB_REBUILD=1` |
| The 23 new tasks are harder (compose/security), dragging the mean reward | report per-substrate and 218-pool eval separately (3.4) |

## 8. Decisions taken (change here if needed)

- **Name:** `demo_hybrid_terminalbench`, a new folder; `demo_cpu_terminalbench` is untouched.
- **m5 tasks run unmodified**, except the 3 bit-rot patches. No flattening, no slimming.
- **Build on the m5**; ECR push optional.
- **One m5** (`m5.8xlarge`). Add a second host plus least-loaded routing only if the gate wait shows up
  in the P4 metrics.
- **Not the g6e as Docker host.** It has the disk and CPUs, but running untrusted policy code next to
  the trainer, with a Docker daemon on the training box, is the wrong blast radius.

## Related

- `docs/terminalbench-unsupported-tasks.md`: the 23 tasks and their blockers.
- `docs/docker-compose-on-aws.md`: Docker + compose on plain EC2.
- `docs/agentcore-networking.md`: SG→SG dial-back pattern (adapter ports).
- `code/demos/demo_cpu_terminalbench/README.md`: the microVM pool, pre-open, heartbeat.
