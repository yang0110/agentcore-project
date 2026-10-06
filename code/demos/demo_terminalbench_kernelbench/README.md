# demo_terminalbench_kernelbench — RL on ALL of Terminal-Bench + KernelBench: AgentCore microVMs for what fits, m5 Docker hosts for the rest, g6 GPUs for KernelBench

**Three machines, one run.** The g6e.48xlarge (8×L40S) trains. Each task is routed to one of three sandbox
substrates, and gets a fresh sandbox per episode on each:

| Substrate | Tasks | One episode = |
|---|---|---|
| **AgentCore microVM** (`acs_tbh_<task>` Runtimes, arm64, ≤ 2 GB) | single-container tasks whose image builds for arm64, fits, and whose oracle passes there | one fresh microVM session |
| **m5 EC2** (plain EC2, not AgentCore-managed; Amazon Linux 2023, x86_64, Docker + compose) | everything else: the 12 multi-container compose tasks, x86-only and > 2 GB images, and microVM tasks whose reference solution fails on a capability-less microVM | **one `m5.2xlarge` (40 GB) per session** running the task's whole compose project (client + sidecars). `TB_M5_MODE=warm` (default): kept after the episode and reused after a *verified* reset; `per_episode`: terminated after grading; `shared`: one compose project each on one shared m5 |
| **g6 GPU tier** (AgentCore capacity provider `acs_tbh_gpu_cp`, g6.xlarge / NVIDIA L4; one session = one instance) | **KernelBench** (40 level-1 train + 10 level-2 eval rows, mixed into the batch; no Terminal-Bench task needs a GPU, opt in with `TB_GPU_TASKS=a,b`) | one fresh g6 Instance session |

The policy is Qwen3-4B trained by slime (fully-async GRPO). The agent is the mini_claude tool-calling loop
(`bash` / `write_file` / `read_file`) and it runs **inside the episode's sandbox** on both substrates,
dialling the trainer's per-engine `AnthropicAdapter` back over the VPC. The reward is the task's own
hidden pytest oracle, injected after the agent stops.

Plan and background: [`docs/terminalbench-kernelbench-demo-plan.md`](../../../docs/terminalbench-kernelbench-demo-plan.md),
[`docs/terminalbench-unsupported-tasks.md`](../../../docs/terminalbench-unsupported-tasks.md),
[`docs/docker-compose-on-aws.md`](../../../docs/docker-compose-on-aws.md).
The microVM-only predecessor, `demo_cpu_terminalbench`, was removed on 2026-10-06 (in git history).

## Use case covered

Agentic RL on a benchmark whose environments don't all fit one sandbox product: most tasks are a single
container that a managed microVM runs well, but some need several networked containers, an x86 CPU, a
multi-GB image, or root powers (raw networking, mounts) a capability-less microVM doesn't grant.

## Why this demo

- It trains on **the whole corpus** instead of the ~165 tasks the microVM pool alone can validate:
  every task gets the substrate it actually runs on, decided from data (build result, image size, oracle
  result), not a hand-kept list.
- It shows that the sandbox tier can be **heterogeneous without touching the RL side**: the agent
  loop, the grader, the reward, pre-open and the metrics are the same code for both substrates.
- It keeps the microVM numbers comparable: `eval/terminalbench` is the same held-out microVM task set
  the microVM-only demo used, and the m5 tasks get their own `eval/terminalbench_docker`.

## Design considerations

- **One protocol, two transports.** Every task image (both substrates) is the task's own Dockerfile plus
  the same wrapper layer: a standalone CPython in `/opt/acs/py` and the stdlib `acs_http_server.py`.
  On a microVM, AgentCore invokes it. On the m5 it runs as the compose `client` and the trainer POSTs to
  it directly (`DockerComposeSandbox`, a subclass of `AgentCoreRuntimeSandbox` that only swaps the transport).
- **Routing is data.** `terminalbench/tasks/route_by_image.json` maps each task to
  `arn:aws:bedrock-agentcore:…` or `docker://<m5 ip>:18300/<task>`; `image_routing.py` refuses any
  unmapped task. The m5 gets a task when its manifest says multi-container, when its arm64 build failed
  or exceeded 2 GB (`images.csv`), or when its oracle failed on the microVM (`oracle.csv`, retried on the
  m5 in step 4 and kept where it passes).
- **Per-episode compose rewrite** (`m5_agent/compose_agent.py`): upstream compose files assume one
  episode per host. Each episode gets its own copy of the task dir; fixed `container_name`s become
  `<project>-<service>`, with the old name kept as a network alias; sidecar `ports:` are dropped;
  the client publishes only `:8080` on a random port; `sleep infinity` becomes the server (a prelude
  like `echo … >> /etc/hosts &&` is kept). Privileged, host network or pid, devices, `docker.sock`,
  and bind mounts outside the episode dir are refused.
- **Hidden oracle stays hidden.** Task dirs reach the m5 without `tests/`, `run-tests.sh` or the
  reference solution; the grader injects the tests from the trainer after the agent stops, as on
  the microVM path.
- **Pre-open covers both substrates** (`TB_PREOPEN_BATCHES=2`). An m5 open (compose up + server ready) is
  ~6–10 s, vs ~1–2 s for a microVM. `TB_DOCKER_MAX_PROJECTS` (16) caps m5 projects (pre-opened plus
  running), held from open to close. The heartbeat covers microVM sessions only, because the m5 has
  no idle reclaim.
- **Reuse, don't rebuild.** The microVM image recipe is byte-identical to the removed microVM-only
  demo's (same recipe tag), so the images already in the `acs-tb-<task>` ECR repos are reused (its
  `acs_tb_*` Runtimes were deleted on 2026-10-06; the repos are kept). This demo registers its own
  `acs_tbh_<task>` Runtimes.
- **Bit-rot.** `word2vec-from-scratch` gets a one-line patch (`Salesforce/wikitext`). Two tasks are
  broken upstream with no faithful fix, and are reported as `build_failed`: `fix-git` (its GitHub repo
  was deleted) and `leelachess0-pytorch-conversion` (the lczero network file returns 404).

## How the m5 tier runs (who launches what)

AgentCore doesn't manage the m5s; the trainer does. `TerminalBenchPool` (in slime's rollout process) calls
the EC2 API with boto3, using the g6e's role. See [`docs/m5-per-episode-isolation.md`](../../../docs/m5-per-episode-isolation.md).
- **Builder** (`acs-tbh-m5`, one `m5.8xlarge`, `deploy/m5_host.py up`): builds every m5 task's images and
  pushes them, sidecars included, to ECR `acs-tbhdk-<task>-<service>`. It is also where oracle validation
  runs (`TB_M5_MODE=shared`). Training doesn't need it.
- **Per episode** (`Ec2EpisodeSandbox`):
  1. `run_instances` from the baked AMI (`deploy/m5_host.py bake-ami`: Docker + compose agent, no task
     images), with a fresh agent secret.
  2. Upload the task folder (no tests or solution) and its ECR image list.
  3. The agent runs `docker login` + pull, then `compose up`.
  4. The episode and its grading run.
  5. `terminate_instances`.
- **Pre-open:** slow substrates are opened `TB_PREOPEN_BATCHES_SLOW=3` batches ahead (microVMs 2); at most
  `TB_M5_MAX_INSTANCES=32` instances are live at once.
- **Leftovers** (tag `acs-tbh-m5-ep`) are terminated at the next pool start, by `6_cleanup.sh`, or by
  `deploy/m5_host.py gc-episodes`.

## KernelBench on the g6 tier

KernelBench rows (`kernelbench/`, restored from `demo_gpu_single`) name the image `kernelbench-gpu`, which
routes to `gpu:<acs_tbhgpu_kernelbench ARN>`. `TerminalBenchPool` runs them with the same fresh, pre-opened,
gated session as Terminal-Bench rows, but grades them with the coding episode (`mini_claude_episode`):
- the agent writes `solution.py` with `ModelNew`;
- the hidden checker (random name, nonce'd PASSED line) builds it next to the reference `Model` on the
  L4 and checks shape, values, and that it ran on CUDA;
- it prints `speedup=`;
- the reward is 1 if the kernel is correct and ≥ `TB_KB_SPEEDUP_MIN` (1.10×) faster, 0.5 if only correct,
  0 otherwise. Copies of the reference measured 1.00–1.03×, so the margin is above timing noise.

The image (`deploy/Dockerfile.kernelbench_gpu`, 145 MB) installs CUDA torch at container start, because
AgentCore's 2 GB image cap applies to capacity-provider Runtimes too. Its server is
`deploy/acs_http_server_chunked.py`: Instance Runtimes forward request bodies chunked. `g5.xlarge` (A10G) is
accepted by AgentCore as an alternative (`TB_GPU_INSTANCE`), but g6 is cheaper (~$0.80/h vs ~$1.01/h)
and the verified default.

## Where the g6 instances are

AgentCore runs capacity-provider instances in a service-managed account, so they do **not** appear in
your EC2 console. A g6 is launched when a session's first call arrives (the pool's pre-open ping), serves
that one session, and is released after `StopRuntimeSession` at the episode's end (any instance with no
session goes after the capacity provider's 600 s idle timeout).

To see them: CloudWatch log group `/aws/bedrock-agentcore/runtimes/<acs_tbhgpu_kernelbench id>-DEFAULT`
has one log stream per session (= per g6 instance), named by the session id; the run log has each
session's `[sbx-op]` open (first `ping`) and `stop_session` lines.

## Warm m5 pool (how instances are reused and when they are terminated)

`WarmPool` (`agentcore_slime_demo/ec2_episode_sandbox.py`) serves one session per instance at a time:
- **Opening:** use an idle instance (prefer one already holding the task's images); else launch one if
  live + launching < `TB_M5_MAX_INSTANCES` (24); else wait. So the pool grows only to the peak concurrent
  demand.
- **After each episode:** `compose down -v`, then `reset_check`: no containers, volumes or non-default
  networks, an empty episode dir, and disk ≥ 8 GB (cached images are evicted otherwise).
  - Clean: the instance goes back to the pool.
  - Dirty, or a failed health check on reuse: it is terminated, never reused.
- **No periodic recycle:** the agent never touches the host, so a verified reset leaves nothing behind.
- **Termination** (AgentCore doesn't manage these instances; our code does):
  - the trainer's `atexit` hook;
  - `5_run.sh` → `deploy/m5_host.py gc-episodes` (every instance tagged `acs-tbh-m5-ep`);
  - the next run's pool start;
  - `6_cleanup.sh`;
  - a **dead-man switch**: an instance with no trainer call for `TB_M5_IDLE_SHUTDOWN_S` (1800) shuts
    itself down. Its `InstanceInitiatedShutdownBehavior=terminate` makes that a terminate.

## Isolation

| | microVM | m5 compose project |
|---|---|---|
| Per episode | its own microVM (own kernel) | per_episode: its own EC2 VM (own kernel), terminated after grading. shared: its own containers, networks, volumes and task-dir copy, removed with `down -v` |
| Shared | the read-only image | per_episode: nothing (images pulled from ECR). shared: the read-only images and the host kernel |
| Agent ↔ sidecars | n/a | container boundary; reachable only over the project network, as upstream |

The m5 is weaker than a microVM (shared kernel), and it is how Terminal-Bench itself runs. The host is
hardened:
- No IAM role.
- IMDSv2 with hop limit 1, and `DOCKER-USER` drops traffic to 169.254.169.254.
- Containers can't reach the host itself: an `INPUT` drop on `docker0`/`br-*` covers the agent, sshd
  and the daemon.
- The agent API needs a shared secret and is open only to the trainer's SG.
- Docker uses its own `10.200.0.0/16` address pool, so it never shadows the VPC's `172.31/16`.

## Step by step (run from this dir on the HOST; container steps `docker exec` themselves)

```bash
bash 2_build_image.sh      # fetch all 241 tasks; arm64 images -> ECR (reused); launch the m5; build m5 tasks there
bash 3_push_ecr.sh         # report only: images.csv (microVM) + images_docker.csv (m5)
bash 4_deploy_runtime.sh   # register acs_tbh_* Runtimes; validate every oracle; retry microVM failures on the m5
bash 1_prepare_data.sh     # rows for every routed + validated task (train mixes both; two eval files)
bash 5_run.sh              # 10-step smoke -> logs/<tag>.{log,csv,eval.csv,png}
bash 6_cleanup.sh          # delete acs_tbh_* Runtimes + terminate the m5 (KEEP_M5=1 keeps it)
```

`TB_SUBSTRATES=microvm` or `=docker` runs one half of steps 2 and 4. `TB_TASKS=a,b` (or `TB_LIMIT=2`)
validates the pipeline on a few tasks first. Manage the m5 directly with
`python3 deploy/m5_host.py up|status|update|down` (`update` pushes a new `compose_agent.py`).

## Knobs (env vars; defaults in `run.sh` / the deploy scripts)

| Knob | Default | Meaning |
|---|---|---|
| `NR` / `EVAL_INTERVAL` | 11 / 10 (5_run.sh) | rollouts (NR-1 train steps) / eval cadence |
| `TB_PREOPEN_BATCHES` | 2 | batches of sandboxes booted ahead, both substrates (0 = open on demand) |
| `TB_M5_MODE` | warm | `warm` = one m5 per session, reused after a verified reset; `per_episode` = terminated after each episode; `shared` = one compose project each on the builder |
| `TB_M5_MAX_INSTANCES` | 24 | warm / per_episode: live m5 instances at once (~87 % of moments need ≤ 24) |
| `TB_PREOPEN_BATCHES_SLOW` | 5 | batches ahead for m5 VMs and g6 Instances (microVMs: `TB_PREOPEN_BATCHES`); a g6 needs ~2.5 min (boot + torch install) |
| `TB_DOCKER_MAX_PROJECTS` | 16 | shared mode: m5 episode projects at once (trainer-side gate; the host's `M5_MAX_PROJECTS` is the backstop) |
| `TB_KERNELBENCH` / `TB_KB_TRAIN` / `TB_KB_EVAL` | 1 / 40 / 10 | the KernelBench g6 Runtime (step 4) / KernelBench rows in train / eval |
| `TB_GPU_TASKS` / `TB_GPU_MAX_SESSIONS` / `TB_GPU_INSTANCE` | empty / 32 / g6.xlarge | Terminal-Bench tasks on the g6 tier / live g6 sessions at once, per pool (at 16 the gate was full 66 % of smoke 5 and 31/36 pre-opened sessions were still booting; ~4 KernelBench prompts × 8 are in the look-ahead window) / instance type |
| `TB_KB_SPEEDUP_MIN` | 1.10 | KernelBench reward: 1 if correct and at least this much faster than the reference, 0.5 if only correct, 0 otherwise (0 = correctness only) |
| `TB_M5_IDLE_SHUTDOWN_S` | 1800 | m5 dead-man switch: an episode instance with no trainer call for this long terminates itself (0 = off). Must exceed the longest normal silence of an idle warm instance (final checkpoint save, eval: ~10 min); shorter = idle warm instances get replaced mid-run, longer = orphans bill longer after a crash |
| `INVOKE_EXEC_MAX_RETRIES` | 1 | retries of a failed `exec` (not idempotent; 9 retries of a 4-min call stalled an eval) |
| `HEARTBEAT` | 1 | keep idle microVM sessions alive across trainer stalls |
| `EPISODE_MAX_TURNS` / `EPISODE_MAX_TOKENS` | 8 / 1024 | agent turns / tokens per turn |
| `TB_EXEC_TIMEOUT` | 180 | per bash-tool command, seconds |
| `M5_INSTANCE_TYPE` / `M5_DISK_GB` | m5.8xlarge / 500 | the Docker host |
| `M5_CLIENT_MEM` / `M5_CLIENT_CPUS` | 8g / 4 | limits on each episode's client container |
| `TB_DOCKER_ORACLE_FALLBACK` | 1 | retry microVM oracle failures on the m5 (step 4) |
| `TB_RUNTIME_TAG` | tbh | microVM Runtime names `acs_<tag>_<task>` |
| `TB_VALIDATE` | 1 | step 4 validates every oracle (0 = skip; then step 1 trains on unvalidated tasks) |

## Metrics

Every `[demo]` line carries `substrate=microvm|docker`. `metrics.py` prints a **by substrate** block
(solve rate, open/agent/grade/episode p50, infra errors), and m5 opens appear as `[sbx-op] op=open
arn=docker:<task>` (compose up + server ready), so `perf/sandbox_open_*` covers both substrates. Eval
reports `eval/terminalbench` (microVM held-out set, the one the microVM-only demo used) and
`eval/terminalbench_docker` (m5 held-out set).

## Verified locally (2026-10-04, compose agent on the g6e, no AWS)

- **Build:** the 4 compose tasks that exercise the rewrite built natively for amd64 in 15–60 s each:
  `postgres-csv-clean`, `sql-injection-attack`, `home-server-https` (its client isn't named `client`)
  and `simple-web-scraper` (it builds a sidecar).
- **Oracle:** `validate_oracle.py` through `DockerComposeSandbox` passed 4/4. The reference solution
  passes and the untouched environment fails (10/10, 14/14, 5/5, 2/2 tests).
- **Concurrency:** 3 concurrent `postgres-csv-clean` episodes (fixed `postgres_db` name, `5432:5432`,
  `./:/app/data` bind mount) each got their own DB and port. A file written by one was invisible to the
  others, and nothing leaked after `down`.
- **Agent loop:** the real `InSessionHarness` ran inside a compose client, dialled a stand-in adapter
  on the trainer IP, and ran a tool call against the Postgres sidecar.

## Run on AWS (2026-10-04)

- **Deploy:**
  - 218 `acs_tbh_*` microVM Runtimes registered, all images reused from ECR.
  - On the builder, 71 of 78 m5 tasks built and were published to ECR.
  - Per-episode AMI `ami-REDACTEDbf653f3d` baked.
- **Oracle:** **191 usable tasks** (163 microVM + 28 m5), against 165 for the microVM-only pool;
  2 more passed after the data was built (`word2vec-from-scratch` patched, `vul-flink`). The m5 rescued:
  - all 12 compose tasks except those failing upstream;
  - `accelerate-maximal-square` (x86-only);
  - the four > 2 GB images;
  - 21 microVM oracle failures.
- **Per-episode probe:** two fresh `m5.xlarge` opened in 51 s and 34 s (launch → ECR pull → compose up);
  the oracle passed 2/2 and both were terminated.
- **RL smoke 2** (10 steps, both substrates): 401 microVM + 88 per-episode-m5 episodes, 1 infra error;
  m5 open p50 35 s, episode-start wait p50 6 s (hence `TB_PREOPEN_BATCHES_SLOW`); eval 2/62 (microVM)
  and 0/8 (m5). The final checkpoint save hung on `pidfd_getfd` (now fixed by making
  `PYTORCH_CUDA_ALLOC_CONF` opt-in).
- **RL smoke 3 — PASSED** (`Job … succeeded`, 53 GB checkpoint `iter_0000010`):
  - ~380 microVM + 88 per-episode-m5 episodes, **0 infra errors** on both.
  - Train solve rate 4.2 % (microVM) and 0 % (m5).
  - **eval/terminalbench 4/62** (microVM-only demo smoke: 1/62); eval/terminalbench_docker 0/8.
  - m5 open p50 35 s; episode-start wait p50 11 s. The deeper look-ahead (`TB_PREOPEN_BATCHES_SLOW=3`,
    32 instances) was added after this run started, so the next run is its first.
  - Logs: `logs/smoke3-hybrid_20261004_113716.*`.
- **Mixed smoke 4 — PASSED** (Terminal-Bench microVM + warm m5 + KernelBench g6, 53 GB checkpoint):
  - microVM: 322 episodes, 13 solved, 1 infra error.
  - Warm m5: 58 episodes, 1 infra error. Capped at 24 instances; 31 of 81 opens reused a warm instance;
    0 dirty resets.
  - g6 KernelBench: 75 episodes, 48 solved, 0 infra errors; speedup up to 1.29×.
  - **eval/kernelbench 16/20**, eval/terminalbench 2/62, eval/terminalbench_docker 0/10.
  - Findings:
    - g6 episodes all waited at start (p50 240 s), because `TB_GPU_MAX_SESSIONS=8` equals one group, so
      nothing can pre-open while a group runs. Use 16.
    - The KernelBench reward is correctness-only and easy (copying `Model` onto CUDA passes); a
      speedup-gated reward is the next step.
  - Logs: `logs/smoke4mix-hybrid_20261004_135153.*`.
- **Smoke 5 — PASSED** (3 steps + eval; `TB_GPU_MAX_SESSIONS=16`, speedup-gated KernelBench reward):
  - KernelBench rewards 26 × 0.5, 10 × 0, 0 × 1.0: correct-but-not-faster kernels no longer score 1, so
    groups carry a real GRPO signal.
  - eval/kernelbench 0.475 (mean reward), eval/terminalbench 4/62.
  - All 36 g6 episodes were pre-opened; start wait p50 **82 s** (240 s with cap 8). A g6 needs ~2.5 min
    and 3 short batches of lead weren't quite enough, so `TB_PREOPEN_BATCHES_SLOW` is now 5.
  - All 24 warm m5s were terminated at exit.
  - Logs: `logs/smoke5-hybrid_*`.

## Folder layout (self-contained)

- `1_…6_*.sh`: the step scripts. Also `run.sh` (slime launch), `train.sh` (convergence wrapper),
  `prepare_data.sh`, `image_routing.py` (task → route, strict) and `metrics.py` (log → csv/png).
- `agentcore_slime_demo/`: vendored harness lib, including `terminalbench_pool.py` (both substrates)
  and `docker_compose_sandbox.py` (the m5 client).
- `m5_agent/`: `compose_agent.py` (the m5 host agent, stdlib, Python ≥ 3.9) and `bootstrap.sh`
  (cloud-init: Docker, compose/buildx, firewall, systemd unit).
- `terminalbench/`:
  - fetch and the dataset builder;
  - `deploy_terminalbench.py` (microVM half) and `deploy_docker_tasks.py` (m5 half);
  - `validate_oracle.py` and `probe_coldstart.py`, which handle both substrates;
  - `tasks/` (the corpus and route map, not in git).
- `deploy/`: `demo.env`, `agentcore_deploy.py`, `steps_common.sh`, `m5_host.py` (provisioning),
  and `m5_host.json` (state + secret, not in git).
