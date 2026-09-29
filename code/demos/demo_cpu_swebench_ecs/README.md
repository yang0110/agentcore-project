# demo_cpu_swebench_ecs — SWE-bench Verified RL on ECS (EC2): one fresh ECS task per episode

**Sandbox substrate: Amazon ECS on EC2, not AgentCore.** This is the ECS twin of
[`../demo_cpu_swebench`](../demo_cpu_swebench/README.md) (AgentCore, one arm64 microVM per episode) and
the sibling of [`../demo_cpu_swebench_eks`](../demo_cpu_swebench_eks/README.md) (one gVisor pod per
episode). Dataset rows, harness, grader and RL recipe are the same in all three. Only the thing that
hosts an episode changes. Here it is a fresh **ECS task** (`RunTask`, `awsvpc` network mode: its own
ENI and security group) on a shared m5 node of an Auto Scaling group behind a capacity provider.

Created 2026-09-28 as a copy of `demo_cpu_swebench_eks`. The vendored library is unchanged apart from:

- one new file: `ecs_task_sandbox.py` (`EcsTaskSandbox`, a drop-in for `AgentCoreRuntimeSandbox`);
- `terminalbench_pool.py`, which now opens an `EcsTaskSandbox` per episode.

The in-image server (`deploy/acs_swe_http_server.py`, `POST /invocations` on :8080), the in-session
harness (`InSessionHarness`, the loop dials the trainer's `AnthropicAdapter` back), the hidden-test
grader (`swebench.grade_swebench`) and the harness tier (`HarnessPool`, W = RB×NS workers,
least-loaded engine) are all identical. **REMOTE-ONLY** (`SANDBOX` must be `ecs-ec2`).

**Status (2026-09-28): code complete, never deployed.** No ECS resource exists; `deploy/ecs_deploy.py`
has only been exercised offline against fake AWS clients. (The trainer box's role also has no `ecs:*`
permission today; `up` needs ECS, EC2, Auto Scaling, IAM and Logs rights.)

---

## Use case covered

RL on real GitHub issues (django, sympy, sphinx, pytest, astropy, requests, pylint, flask). The agent
edits `/testbed` and is rewarded when the instance's hidden FAIL_TO_PASS + PASS_TO_PASS tests pass.
The episodes run on the **most common self-managed AWS container substrate**: ECS on EC2, no
Kubernetes. Default subset: the same 20 instances as the AgentCore demo (16 train + 4 eval), pinned in
`fetch_swebench.DEFAULT_INSTANCES`.

## Why this demo

- **The third point of the substrate comparison.** AgentCore (managed microVM), EKS + gVisor
  (user-space kernel) and ECS (plain container with its own ENI) on identical rows and metrics:
  open latency, cost per episode, isolation, operational burden.
- **The baseline most teams already run.** No cluster control plane, no kubectl, no gVisor install:
  RunTask / StopTask plus an ASG. It shows what the managed sandbox buys over the simplest
  do-it-yourself option.
- **No AgentCore image limits.** Stock amd64 SWE-bench images, no 2 GB cap, so all 500 Verified rows
  are possible (`SWE_QUOTA` / `SWE_MAX_IMAGE_MB`).

## Design considerations

- **Isolation: a container per episode, kernel shared.** This is the weakest of the three tiers and
  the demo says so:
  - one task per episode, never reused, so no state leaks between episodes through the filesystem;
  - all Linux capabilities dropped (`linuxParameters.capabilities.drop = ALL`), non-privileged, init
    process, 8 GiB hard memory limit;
  - **no task IAM role** and `ECS_AWSVPC_BLOCK_IMDS=true` on the node, so the agent cannot get AWS
    credentials (neither the task's nor the node's); IMDSv2 hop limit 1 as a second fence;
  - but the node **kernel is shared** by every task on the node (no microVM, no gVisor): a kernel
    exploit escapes to the node. Run only trusted-model RL here, or add Firecracker/gVisor yourself.
- **Network: own ENI + security group per task (`awsvpc`).**
  - Tasks get VPC IPs in the `${NAME_PREFIX}-egress-*` subnets; the trainer talks HTTP straight to
    `taskIP:8080` (no AWS data plane in the path).
  - Task SG `${NAME_PREFIX}-swe-ecs-task`: ingress tcp/8080 from the trainer SGs only; the default
    allow-all egress is revoked and replaced by egress to the trainer's adapter ports (18100-18163,
    18021) only. That is the AgentCore demo's "egress none", enforced by the SG, not a policy engine.
  - Trainer side: the same SG→SG dial-back rules (`agentcore_deploy.open_dialback`).
  - `awsvpcTrunking` is enabled for the instance role, so an m5.2xlarge gets dozens of task ENIs
    instead of 3.
- **Images: shared with the EKS twin.** `deploy/deploy_swebench.py` is the same script: stock amd64
  `swebench/sweb.eval.x86_64.*` + standalone CPython + the server, `acs-swe-*` repos, same recipe tag.
  One **task definition per image** (`${NAME_PREFIX}-swe-ecs-<id>`, image pinned by digest), written to
  `swebench/taskdef_by_task.json`; re-registered only when its spec hash changes (`acs-spec` tag).
- **Image cache per node + pre-pull.** `ECS_IMAGE_PULL_BEHAVIOR=prefer-cached` (safe: digests), long
  image-cleanup ages, and with `ECS_PREPULL=1` the node user data pulls every pool image at boot, in
  the background. Disk budget as on EKS: 3-6 GB per image per node; default root 300 GiB gp3.
- **Scaling: capacity provider with managed scaling.** Target capacity 100 %, managed termination
  protection, `ECS_NODES` desired (4), `ECS_NODES_MIN`..`ECS_NODES_MAX` (0..16). A RunTask that finds
  no room waits in `PROVISIONING` while the ASG adds a node; that wait is inside `open_ms`.
- **Open path + timings.** `open_sent_epoch` is stamped before `RunTask`, so `open_ms` means the same
  as on AgentCore. The `[demo]` line adds `provision_ms` (until the image pull starts: placement /
  scale-out), `pull_ms` (`pullStartedAt`→`pullStoppedAt`, ~0 when cached), `task_running_ms`,
  `server_ready_ms` (first 200 on `/ping`) and `node` (container instance id).
  - One background thread per process polls **all** waiting tasks with batched `DescribeTasks`
    (100 per call), so 64 pre-opened episodes cost one API call per `ECS_POLL_SEC`, not 64.
  - `RunTask` `failures` (e.g. no capacity yet) are retried every `ECS_RUNTASK_RETRY_SEC` until
    `ECS_OPEN_TIMEOUT` (1200 s); a task that stops before ready raises with its `stoppedReason`.
- **Pre-open + heartbeat, both on by default (as in the AgentCore demo).** `TB_PREOPEN_BATCHES=2`
  keeps the next 2 steps' tasks booted. `HEARTBEAT=1` registers every open task with `SessionKeeper`
  (no-op exec when idle past `WARM_PING_FRACTION × WARM_IDLE_TIMEOUT`, 120 s). Tasks have no idle
  reclaim, so the ping is a liveness check that also heals the pre-open queue:
  - A pre-opened task that fails the ping (node scaled in or drained, OOM-killed) is closed and
    replaced before a worker claims it (`[demo] preopen_replaced=dead`).
  - At claim, a task idle more than `TB_PREOPEN_RECHECK_SEC` (2 s) is pinged once. If it is dead
    (`dead_at_claim`) or its boot failed (`open_failed`), the episode opens a fresh task.
  - A running episode's task is only logged (`[demo] heartbeat_failed=…`). `HEARTBEAT=0` turns the ping off.
  - Pre-open is capped to the fleet: `SANDBOX_SLOTS` (default `ECS_NODES_MAX × ECS_TASKS_PER_NODE` =
    16 × 4) minus `HARNESS_WORKERS`, so extra tasks never wait in PROVISIONING until
    `ECS_OPEN_TIMEOUT`. Set `SANDBOX_SLOTS=0` for no cap.
  - Old pre-opens retire: a task is handed out only if `age + TB_EPISODE_BUDGET_SEC` fits within
    `ECS_TASK_MAX_LIFETIME`. Otherwise it is replaced (`expiring`) rather than SIGKILLed mid-episode.
- **Leaks.** A leaked task (trainer crash) is killed by its entrypoint, `timeout -s KILL ECS_TASK_MAX_LIFETIME`
  (7200 s); `ecs_deploy.py down` also stops every task with `startedBy=acs-swe`.
- **Sizing.** Default task: 1.5 vCPU (1536 units) / 8 GiB hard, 6 GiB reservation. An m5.2xlarge
  (8 vCPU / 32 GiB) holds ~4 tasks, so the default 4 nodes run ~16 concurrent episodes; for
  `TB_PREOPEN_BATCHES=2 × RB × NS` = 64 pre-opened sessions on top of 32 running, the default
  `ECS_NODES_MAX=16` (64 slots) leaves room for 32; the pool clips to that and logs it. Raise
  `ECS_NODES_MAX` for the full look-ahead.
- **No CloudFormation, no CLI.** `deploy/ecs_deploy.py` is pure boto3 and idempotent (every resource is
  looked up by name first); the trainer container needs only boto3.

## How an episode flows

```
slime → terminalbench_pool_generate.generate → TerminalBenchPool.submit
  harness worker pulls episode → route by task.image (= instance id) → taskdef_by_task.json → task definition ARN
  (at submit) EcsTaskSandbox: RunTask (capacity provider, awsvpc, task SG) while the episode waits
    → PROVISIONING (scale-out if full) → PENDING (pull, skipped if cached) → RUNNING → /ping on the ENI IP
      InSessionHarness writes session_runner.py into the task (HTTP to taskIP:8080) and polls it
        in-task loop in /testbed: LLM via the least-loaded engine's adapter (dial-back, the only egress the SG allows)
    grade: swebench.grade_swebench (reset test files → apply hidden test_patch → test_cmd → parse F2P/P2P)
  finish_session → ONE loss-masked Sample (the longest); adapter.drop_session; StopTask
```

## Step by step (numbered scripts; run from this dir on the HOST — container steps `docker exec` themselves)

```bash
bash 1_prepare_data.sh     # 1. container: fetch_swebench.py (pinned 20 -> swebench/manifest.json) -> data/
bash 2_build_image.sh      # 2. deploy_swebench.py: wrap + push one amd64 image per instance (shared acs-swe-* repos)
                           #    -> swebench/image_by_task.json + images.csv
bash 3_push_ecr.sh         # 3. no-op: prints swebench/images.csv (step 2 already pushed)
bash 4_deploy_runtime.sh   # 4. ecs_deploy.py up: RENDER ONLY unless ECS_CREATE=1 (see below);
                           #    then swebench_oracle_check.py (on ECS tasks) and fetch_swebench.py --data-only
bash 5_run.sh              # 5. container: 10 train steps + 1 eval through the pool; logs/<tag>.{log,csv,eval.csv,png}
ECS_DELETE=1 bash 6_cleanup.sh   # 6. ecs_deploy.py down: tasks, cluster, CP, ASG, LT, task defs, SGs, role, logs
```

- **Creating the substrate is gated.** Step 4 without `ECS_CREATE=1` only writes
  `deploy/ecs/rendered/` (`plan.json`, `user_data.sh`, `taskdef-example.json`), prints the plan and
  stops. Look first: `python3 deploy/ecs_deploy.py render`.
- `ECS_CREATE=1 bash 4_deploy_runtime.sh` is billed (EC2 nodes; ECS itself is free). It creates the
  instance role + profile, SGs, log group, launch template, ASG, capacity provider, cluster and one task
  definition per image, writes `swebench/taskdef_by_task.json` + `swebench/ecs_network.json`, and runs
  one smoke task, which must show `CapEff` all zero, IMDS and the internet unreachable, and print the
  testbed Python version.
- **Validate on 2 instances first**: `SWE_LIMIT=2 bash 2_build_image.sh && ECS_CREATE=1 bash 4_deploy_runtime.sh`.
  If the EKS twin already pushed the images, step 2 only resolves digests.
- `python3 deploy/ecs_deploy.py status` shows nodes, running/pending tasks and the ASG;
  `python3 deploy/ecs_deploy.py smoke` re-runs the smoke task.
- Train to convergence (inside the container): `train.sh` (`STEPS`, `EVAL_EVERY`, `SANDBOX_CONC`).

## Knobs (env vars)

| Knob | Default | Meaning |
|---|---|---|
| `SANDBOX` | `ecs-ec2` | the only accepted value |
| `ECS_CLUSTER` / `ECS_CAPACITY_PROVIDER` | `acs-swe-ecs` / `acs-swe-ecs-cp` | from `swebench/ecs_network.json` in `run.sh` |
| `ECS_SUBNETS` / `ECS_SECURITY_GROUPS` | from `ecs_network.json` | task ENI placement + task SG |
| `ECS_TASK_CPU` / `ECS_TASK_MEMORY` / `ECS_TASK_MEMORY_RESERVATION` | `1536` / `8192` / `6144` | per-episode task size (task-def time) |
| `ECS_TASK_MAX_LIFETIME` | `7200` | entrypoint `timeout`: hard cap on a leaked task (s) |
| `ECS_OPEN_TIMEOUT` / `ECS_POLL_SEC` / `ECS_RUNTASK_RETRY_SEC` | `1200` / `1.0` / `5` | open deadline / batched status poll / RunTask retry |
| `ECS_NODE_TYPE` / `ECS_NODE_DISK_GIB` | `m5.2xlarge` / `300` | nodes (launch template) |
| `ECS_NODES` / `ECS_NODES_MIN` / `ECS_NODES_MAX` | 4 / 0 / 16 | ASG desired / min / max |
| `ECS_PREPULL` | `1` | pull every pool image at node boot |
| `ECS_CREATE` / `ECS_DELETE` | unset | must be `1` to create / delete |
| `TB_PREOPEN_BATCHES` | `2` | upcoming batches kept open ahead (`× RB × NS` tasks); size the ASG to it |
| `HEARTBEAT` / `WARM_IDLE_TIMEOUT` / `WARM_PING_FRACTION` | `1` / `240` / `0.5` | keeper liveness ping on tasks idle past fraction × timeout (dead pre-opens replaced); `0` = off |
| `SANDBOX_SLOTS` | `ECS_NODES_MAX × ECS_TASKS_PER_NODE` (64) | fleet capacity; pre-open is clipped to `SANDBOX_SLOTS − HARNESS_WORKERS`; `0` = no cap |
| `TB_EPISODE_BUDGET_SEC` / `TB_PREOPEN_RECHECK_SEC` | `turns × TB_EXEC_TIMEOUT + 660` / `2` | a pre-opened task is used only if an episode still fits `ECS_TASK_MAX_LIFETIME`; claim-time ping after this much idle |
| `HARNESS_WORKERS`, `CONC`, `EPISODE_MAX_TURNS`, `RB`/`NS`/`NR`, … | as in `demo_cpu_swebench` | RL + harness knobs are unchanged |

- Deploy / data: `SWE_TASKS`, `SWE_LIMIT`, `SWE_BUILD_CONC`, `SWE_REBUILD`, `SWE_NO_PUSH`, `SWE_QUOTA`,
  `SWE_INSTANCES`, `SWE_MAX_IMAGE_MB`, `SWE_TEST_TIMEOUT`, as in the AgentCore demo.

## What to watch

- `rollout/raw_reward` and `eval/swebench`: should match the AgentCore and EKS runs within noise.
- `[demo] fresh_session=1 … provision_ms=… pull_ms=… task_running_ms=… server_ready_ms=… node=…`:
  large `provision_ms` = no room (scale-out); non-zero `pull_ms` = image not cached on that node yet.
- `perf/sandbox_ms_*`: exec latency on runc; the fastest of the three tiers (no gVisor syscall tax,
  no remote data plane), which is the price of the weaker isolation.
- `swebench/oracle.csv`: the gold/empty check on real ECS tasks (caps dropped, no egress).

## Limitations / known risks

- **Shared kernel** (see Isolation). Not a sandbox for adversarial code.
- ENI trunking must be on before the nodes register (`up` sets it first); nodes launched earlier
  keep 3 task ENIs each — recycle them.
- RunTask is API-rate-limited per account and region; throttled calls are retried with botocore's
  adaptive backoff, and the pool's pre-open cap (64) bounds a burst.
- Per-task disk is the node's shared Docker storage (no per-task quota on EC2); a runaway episode
  can fill a node. Size `ECS_NODE_DISK_GIB` with headroom.

## Folder layout (self-contained)

- `agentcore_slime_demo/`: vendored library (as in `demo_cpu_swebench`), plus:
  - `ecs_task_sandbox.py`: `EcsTaskSandbox`, one ECS task per episode (batched status watcher);
  - `terminalbench_pool.py`: patched to open `EcsTaskSandbox` (keeper pings = liveness check).
- `deploy/ecs_deploy.py`: render / up / smoke / status / down for the substrate.
- `deploy/ecs/rendered/` (gitignored): `plan.json`, `user_data.sh`, `taskdef-example.json`.
- `deploy/deploy_swebench.py`: wrap + push one image per instance → `image_by_task.json` (shared with EKS).
- `deploy/acs_swe_http_server.py`: the in-image server (the same one AgentCore and EKS use).
- `swebench_oracle_check.py`: gold/empty check per instance on its own task → `swebench/oracle.csv`.
- `fetch_swebench.py` · `prepare_data.sh` (step 1) · `run.sh` (step 5) · `metrics.py` · `train.sh`.
- `1_…6_*.sh` + `deploy/demo.env`: the numbered steps (`SUBSTRATE=ecs`, BUILD/PUSH/DEPLOY/CLEANUP_CMD).
- `swebench/` (gitignored): `manifest.json` (hidden test patches), `image_by_task.json`,
  `taskdef_by_task.json`, `ecs_network.json`, `images.csv`, `oracle.csv`, build logs.
- `data/`: `swebench_train.jsonl` / `swebench_eval.jsonl`.
