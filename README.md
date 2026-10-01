# Agentic RL on Terminal-Bench with slime and AgentCore Runtime

This repo trains an LLM agent with reinforcement learning on **Terminal-Bench**, using **slime**
(THUDM/slime) as the trainer and **AWS Bedrock AgentCore Runtime** as the sandbox. The policy
(Qwen3-4B, fully-async GRPO) solves each task by working in a real shell: it runs commands, reads
the output, edits files, and tries again. The reward is the task's own hidden pytest oracle, so
there is no LLM judge. The design follows the Agent Lightning v1.0 principle of keeping the agent
harness out of the trainer (arXiv 2608.17528).

## The demo: `code/demos/demo_cpu_terminalbench`

Every Terminal-Bench task ships its own Docker environment, so every task gets its own image and
its own AgentCore Runtime:

- **Image pool of 218 Runtimes.** Each task's own Dockerfile is built for arm64, wrapped with a
  small AgentCore server layer, and deployed as a VPC microVM Runtime (`acs_tb_<task>`). Of 229
  single-container tasks, 7 fail to build for arm64 and 4 exceed the 2 GB microVM image cap.
- **One fresh session per episode.** The agent changes the whole machine, so no microVM is
  shared between episodes.
- **Look-ahead pre-open** (`TB_PREOPEN_BATCHES=2`): the sessions for the next 2 batches boot
  ahead of time, which takes the cold start off the critical path.
- **Heartbeat** (on by default): idle pre-opened sessions are kept alive across the trainer's
  weight-sync stall.
- **Harness in the session.** The mini_claude loop (`bash` / `write_file` / `read_file`) runs
  inside the episode's session and dials back over the VPC to the trainer's per-engine
  `AnthropicAdapter`.
- **Harness tier** (`HarnessPool`): W = RB×NS workers pull episodes off one queue. Each episode
  takes its session first, then the least-loaded SGLang engine, which stays sticky for all its
  turns.

Full facts, knobs and layout: [`code/demos/demo_cpu_terminalbench/README.md`](code/demos/demo_cpu_terminalbench/README.md).

## Run it

Run the numbered steps **on the host**, from the demo folder. The steps that need the
`agentcore_p4` slime container `docker exec` into it themselves (start the container once with
`code/p4_slime_train/00_start_container.sh`).

```bash
cd code/demos/demo_cpu_terminalbench
bash 2_build_image.sh      # fetch the corpus + build/push one arm64 image per task (resumable, ~2 h cold)
bash 3_push_ecr.sh         # report only (step 2 already pushed)
bash 4_deploy_runtime.sh   # one VPC Runtime per image -> terminalbench/tasks/arn_by_image.json
bash 1_prepare_data.sh     # train/eval rows for the deployed tasks only
bash 5_run.sh              # 10-step smoke -> logs/<tag>.{log,csv,eval.csv,png}
bash 6_cleanup.sh          # delete every acs_tb_* Runtime (ECR images kept)
```

Use `TB_LIMIT=2` (or `TB_TASKS=a,b`) on steps 2 and 4 to check the pipeline on a couple of tasks
first. The trainer starts from the base Megatron checkpoint
`code/p4_slime_train/ckpt/Qwen3-4B_torch_dist` (override with `CKPT_BASE`).

For the long run (about 200 steps), run the convergence wrapper **inside** `agentcore_p4`:

```bash
bash /work/code/demos/convergence/20_cpu_terminalbench.sh
```

See [`code/demos/convergence/README.md`](code/demos/convergence/README.md) for its knobs and
disk budget. Run only one `run.sh` at a time: a second Ray start kills the first job.

## Repo map

```
code/demos/demo_cpu_terminalbench/   the demo (self-contained: vendored lib, deploy/, steps 1-6)
code/demos/convergence/              20_cpu_terminalbench.sh (the long run)
code/p4_slime_train/                 agentcore_p4 container start script + base checkpoint
docs/report/                         the project report (md + tex + pdf)
docs/slides/aws/                     the deck: terminalbench.tex (build: ./build.sh terminalbench)
docs/*.md                            design notes; most describe removed demos and are marked historical
```

Status and next steps: [`PROGRESS.md`](PROGRESS.md). Other demos (tasks, SWE-bench, GPU, CPU
Instance, branching, GPU farm) were removed on 2026-09-30 and are still in git history.
