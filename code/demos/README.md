# Demos: Terminal-Bench + KernelBench on AgentCore

Two demos, **agentic RL on Terminal-Bench + KernelBench with slime (fully-async GRPO)**, each a
self-contained recipe (vendored lib, `deploy/`, numbered step scripts). The second is the first plus HyperPod:

| Demo | Dataset | Sandboxes | Reward |
|---|---|---|---|
| [`demo_terminalbench_kernelbench`](demo_terminalbench_kernelbench/README.md) | Terminal-Bench (all 241 tasks) + KernelBench | `acs_tbh_*` AgentCore microVM Runtimes + m5 EC2 Docker hosts + a g6 GPU Runtime, routed per task | Terminal-Bench: the task's hidden pytest oracle; KernelBench: correct and ≥ 1.10× faster |
| [`demo_hyperpod_terminalbench_kernelbench`](demo_hyperpod_terminalbench_kernelbench/README.md) | the same, Qwen3.5-9B | the same sandbox tiers; slime on a SageMaker HyperPod Slurm cluster (2 x ml.g6e.48xlarge: one trains, one serves inference; FSx checkpoints). Code only, not run yet | the same |

The machinery, vendored in `demo_terminalbench_kernelbench/agentcore_slime_demo/`:
- **Fresh sandbox per episode** (`TerminalBenchPool`): full isolation, because the agent changes the OS.
- **Look-ahead pre-open** (`TB_PREOPEN_BATCHES=2`, `TB_PREOPEN_BATCHES_SLOW` for m5/g6): sandboxes for the
  next batches boot ahead of time, so the cold start is off the critical path.
- **Heartbeat** (`HEARTBEAT=1`, default): keeps idle pre-opened microVM sessions alive across trainer stalls.
- **In-session harness**: the mini_claude loop runs inside the episode's sandbox and dials the trainer's
  per-engine `AnthropicAdapter` back over the VPC (SG→SG rule, ports 18100-18163).
- **Harness tier** (`HarnessPool`): W = RB×NS workers pull episodes off one queue. Each episode takes
  its sandbox first, then the least-loaded SGLang engine, which stays sticky for all its turns.

`demo_cpu_terminalbench` (microVM-only Terminal-Bench, 218 `acs_tb_*` Runtimes) and the `convergence/`
wrapper were removed on 2026-10-06. Until 2026-09-30 this folder also held `demo_cpu_tasks`,
`demo_gpu_single`, `demo_cpu_instance`, `demo_cpu_branching` and `demo_cpu_swebench`, along with the
multi-demo smoke drivers. All are still in git history.

## Which sandbox does a task need? Three tiers

| Tier | Sandbox | Use it when |
|---|---|---|
| 1 | managed Code Interpreter (`SANDBOX=agentcore`, no image, no Runtime) | **probes only**: there is no route back to the adapter, so no in-session agent loop |
| 2 | own Runtime, generic image (`SANDBOX=agentcore-runtime`) | the task needs no task-specific environment |
| 3 | **own Runtime per row, custom image** | the environment differs per row: **Terminal-Bench**, KernelBench (on a g6 capacity provider) |

Terminal-Bench is tier 3 on the managed arm64 microVM fleet. Its 2048 MB image cap cannot be raised, and
some tasks need several containers, an x86 CPU, or root powers a capability-less microVM doesn't grant;
the demo runs those on m5 EC2 Docker hosts instead.

## Run the demo step by step

Run the numbered scripts from the demo's folder **on the host** (details in its README). The container
steps `docker exec` into `slime_trainer` themselves.

```bash
bash 2_build_image.sh      # arm64 images -> ECR; launch the m5 builder; build the m5 tasks there
bash 3_push_ecr.sh         # report only
bash 4_deploy_runtime.sh   # register the Runtimes; validate every oracle
bash 1_prepare_data.sh     # rows for the routed (+ oracle-verified) tasks only
bash 5_run.sh              # 10-step smoke -> logs/<tag>.{log,csv,eval.csv,png}
bash 6_cleanup.sh          # delete the Runtimes + terminate the m5s (ECR images kept)
```

Constraints:
- **Never run two `run.sh` at the same time**: the second one's Ray start kills the first one's job.
- The demo trains from the base Megatron checkpoint `code/p4_slime_train/ckpt/Qwen3-4B_torch_dist`,
  which you can override with `CKPT_BASE`.

Long run: raise `NR`, e.g. `RUN_TAG=train48 NR=49 bash 5_run.sh` (48 train steps).
