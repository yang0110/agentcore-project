# Demos: Terminal-Bench on AgentCore

Two demos, both **agentic RL on Terminal-Bench with slime (Qwen3-4B, fully-async GRPO)**, each a
self-contained recipe (vendored lib, `deploy/`, numbered step scripts):
- [`demo_cpu_terminalbench`](demo_cpu_terminalbench/README.md) uses a pool of AgentCore microVM
  Runtimes: one arm64 image and one VPC Runtime per task, one fresh microVM session per episode.
- [`demo_hybrid_terminalbench`](demo_hybrid_terminalbench/README.md) covers the **whole corpus**. It
  keeps the microVM pool for the tasks that fit, and adds a plain-EC2 m5 Docker host for the rest
  (compose, x86-only, > 2 GB, microVM oracle failures), with one fresh `docker compose` project per
  episode.

| Demo | Dataset | Pool | Reward |
|---|---|---|---|
| [`demo_cpu_terminalbench`](demo_cpu_terminalbench/README.md) | Terminal-Bench (single-container tasks) | ~218 Runtimes, one per task, built from each task's own Dockerfile | the task's hidden pytest oracle |
| [`demo_hybrid_terminalbench`](demo_hybrid_terminalbench/README.md) | Terminal-Bench (all 241 tasks) | `acs_tbh_*` microVM Runtimes + one m5 Docker host, routed per task | the task's hidden pytest oracle |

The machinery, vendored in `demo_cpu_terminalbench/agentcore_slime_demo/`:
- **Fresh session per episode** (`TerminalBenchPool`): full isolation, because the agent changes the OS.
- **Look-ahead pre-open** (`TB_PREOPEN_BATCHES=2`): sessions for the next batches boot ahead of time, so
  the cold start is off the critical path.
- **Heartbeat** (`HEARTBEAT=1`, default): keeps idle pre-opened sessions alive across trainer stalls.
- **In-session harness**: the mini_claude loop runs inside the episode's session and dials the trainer's
  per-engine `AnthropicAdapter` back over the VPC (SG→SG rule, ports 18100-18163).
- **Harness tier** (`HarnessPool`): W = RB×NS workers pull episodes off one queue. Each episode takes
  its session first, then the least-loaded SGLang engine, which stays sticky for all its turns.

Until 2026-09-30 this folder also held `demo_cpu_tasks`, `demo_gpu_single`, `demo_cpu_instance`,
`demo_cpu_branching` and `demo_cpu_swebench`, along with the multi-demo smoke drivers. They were removed
to keep only Terminal-Bench, and they are still in git history.

## Which sandbox does a task need? Three tiers

| Tier | Sandbox | Use it when |
|---|---|---|
| 1 | managed Code Interpreter (`SANDBOX=agentcore`, no image, no Runtime) | **probes only**: there is no route back to the adapter, so no in-session agent loop (used by `demo_cpu_terminalbench/terminalbench/warmstart/`) |
| 2 | own Runtime, generic image (`SANDBOX=agentcore-runtime`) | the task needs no task-specific environment |
| 3 | **own Runtime per row, custom image** | the environment differs per row: **Terminal-Bench** |

Terminal-Bench is tier 3 on the managed arm64 microVM fleet. Its 2048 MB image cap cannot be raised, so
`demo_cpu_terminalbench` reports and leaves out the 4 tasks whose image doesn't fit, the 7 that don't
build for arm64, and the 12 multi-container tasks. `demo_hybrid_terminalbench` runs those on an m5
Docker host instead.

## Run the demo step by step

Run the numbered scripts from the demo's folder **on the host** (shown for `demo_cpu_terminalbench`;
`demo_hybrid_terminalbench` has the same six steps, see its README). The container steps
`docker exec` into `agentcore_p4` themselves.

```bash
bash 2_build_image.sh      # build + push one arm64 image per task (resumable)
bash 3_push_ecr.sh         # report only
bash 4_deploy_runtime.sh   # one VPC Runtime per image -> arn_by_image.json
bash 1_prepare_data.sh     # rows for the deployed (+ oracle-verified) pool only
bash 5_run.sh              # 10-step smoke -> logs/<tag>.{log,csv,eval.csv,png}
bash 6_cleanup.sh          # delete the pool's Runtimes (ECR images kept)
```

Constraints:
- **Never run two `run.sh` at the same time** (e.g. `5_run.sh` and the convergence wrapper): the second
  one's Ray start kills the first one's job.
- The demo trains from the base Megatron checkpoint `code/p4_slime_train/ckpt/Qwen3-4B_torch_dist`,
  which you can override with `CKPT_BASE`.

Long run: [`convergence/`](convergence/README.md) (`20_cpu_terminalbench.sh`).
