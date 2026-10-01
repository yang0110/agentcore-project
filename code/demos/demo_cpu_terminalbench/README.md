# demo_cpu_terminalbench — RL on Terminal-Bench: an image pool of ~218 AgentCore microVM Runtimes, one fresh session per episode, pre-opened and heartbeated

**Sandbox tier 3: own Runtime, custom image — on the managed arm64 microVM fleet (2 GB image cap).**
Every Terminal-Bench task ships its own Docker environment, so every dataset row gets its own image and
its own AgentCore Runtime. See the tier table in [`../README.md`](../README.md#which-sandbox-does-a-task-need-three-tiers).

**History.** Its own demo until 2026-09-28, then merged into `demo_cpu_tasks` as `DATASET=terminalbench`,
and split out again on 2026-09-30, keeping the fully-async driver, the harness tier, and the heartbeat and
pre-open defaults. `demo_cpu_tasks` itself was removed on 2026-09-30, and this is now the **primary demo**.

The policy is Qwen3-4B trained by slime (async/decoupled GRPO). The agent is the mini_claude
tool-calling loop (`bash` / `write_file` / `read_file`), and it runs **inside the episode's session**:
`in_session_harness.py` writes `session_runner.py` into the session and the loop dials the trainer's
per-engine `AnthropicAdapter` back over the VPC (SG→SG rule on 18100-18163). The reward is the task's
own hidden pytest oracle (`tests/` + `run-tests.sh`, injected after the agent stops — RLVR, no LLM judge).
**REMOTE-ONLY** (`SANDBOX=agentcore-runtime` is the only backend).

## Use case covered

Agentic RL where **every task needs a different OS image** (packages, services, files, compilers) and the
agent changes the whole machine through the shell. This is the "large image pool" shape: hundreds of
bespoke images, each served by its own AgentCore Runtime, with strict row → image → Runtime routing.

## Why this demo

- It is the hardest sandbox shape in the literature datasets: no generic image can serve it, and a
  shared session would leak one episode's changes into the next.
- It shows AgentCore scaling to a **pool of ~218 Runtimes** built from the tasks' own Dockerfiles, with
  no per-task code: the row names its image, `image_routing.py` maps image → ARN.
- It shows the two latency fixes a per-episode microVM needs: **pre-open** (boot the next batches'
  sessions ahead of time) and **heartbeat** (keep them alive across the weight-sync stall).

## Design considerations

- **One fresh microVM session per episode** (`terminalbench_pool.py`): full isolation between episodes,
  since the task mutates the OS. The session is stopped when the episode is graded.
- **Look-ahead pre-open** (`tb_lookahead_source.py`, `TB_PREOPEN_BATCHES=2`): the data source peeks the
  next 2 rollouts' rows and the pool boots their sessions on the right task Runtime, so the cold start
  leaves the critical path. The cold-start probe of all 218 Runtimes is in
  `terminalbench/tasks/coldstart.csv` (`terminalbench/probe_coldstart.py`).
- **Heartbeat** (`HEARTBEAT=1`, default): a SessionKeeper pings every idle pre-opened session below the
  microVM idle timeout, so a trainer stall can't reclaim it (a reclaimed id comes back empty).
- **Wrapping, not rewriting, the task image**: `deploy_terminalbench.py` appends a layer with a
  standalone Python (`/opt/acs/py`) and the stdlib AgentCore server to each task's own Dockerfile;
  the task's interpreter is never touched (images range from python3.8 to no Python at all).
- **What doesn't fit**: of 229 single-container tasks, 7 fail to build for arm64 and 4 exceed the
  2 GB microVM cap; they are reported in `terminalbench/tasks/images.csv` and left out of the data.
- **Network**: VPC with NAT egress (`TB_NETWORK=VPC`): the loop dials the adapter back, and
  `run-tests.sh` installs uv + pytest from the internet.
- **Shared names with nothing else**: Runtimes are `acs_tb_<task>`, ECR repos `acs-tb-<task>`.

## Isolation

One episode = one microVM = one session: a separate kernel per episode, nothing shared between
episodes except the task's read-only image. The adapter port hole (SG→SG) is the only inbound path.

## Step by step (run from this dir on the HOST — container steps `docker exec` themselves)

```bash
bash 2_build_image.sh      # fetch the corpus (once) + build/push one arm64 image per task (resumable, ~2 h cold)
bash 3_push_ecr.sh         # report only (step 2 pushed)
bash 4_deploy_runtime.sh   # one VPC Runtime per image -> terminalbench/tasks/arn_by_image.json
bash 1_prepare_data.sh     # rows for the deployed tasks only (train/eval split by task)
bash 5_run.sh              # 10-step smoke -> logs/<tag>.{log,csv,eval.csv,png}
bash 6_cleanup.sh          # delete every acs_tb_* Runtime (ECR images kept)
```

`TB_LIMIT=2` (or `TB_TASKS=a,b`) on steps 2 and 4 validates the pipeline on a couple of tasks first.
`python3 terminalbench/validate_oracle.py` checks each task's oracle on its Runtime.

## Knobs (env vars; defaults in `run.sh`)

| Knob | Default | Meaning |
|---|---|---|
| `NR` / `EVAL_INTERVAL` | 11 / 10 (5_run.sh) | rollouts (NR-1 train steps) / eval cadence |
| `TB_PREOPEN_BATCHES` | 2 | batches of sessions booted ahead (0 = open on demand) |
| `HEARTBEAT` | 1 | keep idle sessions alive across trainer stalls |
| `EPISODE_MAX_TURNS` / `EPISODE_MAX_TOKENS` | 8 / 1024 | agent turns / tokens per turn |
| `TB_EXEC_TIMEOUT` | 180 | per bash-tool command, seconds |
| `TB_BUILD_CONC` / `TB_REBUILD` | 16 / 0 | parallel image builds / force rebuild |

## Folder layout (self-contained)

- `1_…6_*.sh` — the step scripts (above); `run.sh` (slime launch), `train.sh` (convergence wrapper),
  `prepare_data.sh`, `image_routing.py` (row image → Runtime ARN, strict), `metrics.py` (log → csv/png)
- `agentcore_slime_demo/` — vendored harness lib (`terminalbench_pool.py`, `tb_lookahead_source.py`,
  `in_session_harness.py`, `session_runner.py`, `sandbox_server_agent.py`, …)
- `terminalbench/` — `fetch_terminalbench.py`, `deploy_terminalbench.py`, `build_terminalbench_dataset.py`,
  `validate_oracle.py`, `probe_coldstart.py`, `warmstart/` (pre-open + heartbeat probe, no trainer),
  `tasks/` (fetched corpus + pool map `arn_by_image.json`, not in git)
- `deploy/` — `demo.env`, `agentcore_deploy.py`, `steps_common.sh`
