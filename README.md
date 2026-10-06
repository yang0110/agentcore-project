# Agentic RL on Terminal-Bench + KernelBench with slime and AgentCore

This repo trains an LLM agent with reinforcement learning on **Terminal-Bench** and **KernelBench**, using
**slime** (THUDM/slime) as the trainer and **AWS Bedrock AgentCore** (plus plain EC2 where AgentCore
can't run a task) as the sandbox. The policy (Qwen3-4B, fully-async GRPO) solves each task by working
in a real shell: it runs commands, reads the output, edits files, and tries again. The reward is the
task's own hidden checker (Terminal-Bench's pytest oracle, KernelBench's correctness + speedup check),
so there is no LLM judge. The design follows the Agent Lightning v1.0 principle of keeping the agent
harness out of the trainer (arXiv 2608.17528).

## The demo: `code/demos/demo_terminalbench_kernelbench`

Every task ships its own environment, and they don't all fit one sandbox product, so each task is
routed (by data: build result, image size, oracle result) to one of three substrates, with a fresh
sandbox per episode on all of them:

| Substrate | Tasks | One episode = |
|---|---|---|
| **AgentCore microVM** (`acs_tbh_<task>` Runtimes, arm64, ≤ 2 GB image) | single-container Terminal-Bench tasks that build for arm64, fit, and pass their oracle there | one fresh microVM session |
| **m5 EC2** (plain EC2, Docker + compose) | multi-container compose tasks, x86-only or > 2 GB images, microVM oracle failures | one m5 instance per session (warm, reused after a verified reset) |
| **g6 GPU** (AgentCore capacity provider, NVIDIA L4) | KernelBench (40 train + 10 eval rows, mixed into the batch) | one fresh g6 Instance session |

- **Same RL side for all three.** The mini_claude loop (`bash` / `write_file` / `read_file`) runs
  inside the episode's sandbox and dials back over the VPC to the trainer's per-engine
  `AnthropicAdapter`; the grader, reward, pre-open and metrics are one code path.
- **Look-ahead pre-open**: sandboxes for the next batches boot ahead of time (`TB_PREOPEN_BATCHES=2`,
  `TB_PREOPEN_BATCHES_SLOW=5` for the m5 and g6), which takes the cold start off the critical path.
- **Heartbeat** (on by default): idle pre-opened microVM sessions are kept alive across the
  trainer's weight-sync stall.
- **Harness tier** (`HarnessPool`): W = RB×NS workers pull episodes off one queue. Each episode
  takes its sandbox first, then the least-loaded SGLang engine, which stays sticky for all its turns.

Full facts, knobs and layout: [`code/demos/demo_terminalbench_kernelbench/README.md`](code/demos/demo_terminalbench_kernelbench/README.md).

## Run it

Run the numbered steps **on the host**, from the demo folder. The steps that need the
`agentcore_p4` slime container `docker exec` into it themselves (start the container once with
`code/p4_slime_train/00_start_container.sh`).

```bash
cd code/demos/demo_terminalbench_kernelbench
bash 2_build_image.sh      # fetch all tasks; arm64 images -> ECR; launch the m5 builder; build m5 tasks there
bash 3_push_ecr.sh         # report only (step 2 already pushed)
bash 4_deploy_runtime.sh   # register acs_tbh_* + the g6 KernelBench Runtime; validate every oracle
bash 1_prepare_data.sh     # train/eval rows for the routed + validated tasks
bash 5_run.sh              # 10-step smoke -> logs/<tag>.{log,csv,eval.csv,png}
bash 6_cleanup.sh          # delete the Runtimes + terminate the m5s (ECR images kept)
```

Use `TB_LIMIT=2` (or `TB_TASKS=a,b`) on steps 2 and 4 to check the pipeline on a couple of tasks
first. The trainer starts from the base Megatron checkpoint
`code/p4_slime_train/ckpt/Qwen3-4B_torch_dist` (override with `CKPT_BASE`).

For a longer run, raise the step count, e.g. 48 train steps:

```bash
RUN_TAG=train48 NR=49 bash 5_run.sh
```

Budget about 56 GB of disk per saved checkpoint. Run only one `run.sh` at a time: a second Ray
start kills the first job.

## Repo map

```
code/demos/demo_terminalbench_kernelbench/   the demo (self-contained: vendored lib, deploy/, steps 1-6)
code/p4_slime_train/                         agentcore_p4 container start script + base checkpoint
docs/report/                                 the project report (md + tex + pdf)
docs/slides/aws/agentic_rl_on_agentcore/     the deck: agentic_rl_on_agentcore.tex (build: ./build.sh)
docs/*.md                                    design notes; most describe removed demos and are marked historical
```

Status and next steps: [`PROGRESS.md`](PROGRESS.md). The microVM-only `demo_cpu_terminalbench` and
its `convergence/` wrapper were removed on 2026-10-06; the other demos (tasks, SWE-bench, GPU, CPU
Instance, branching, GPU farm) on 2026-09-30. All are still in git history.
