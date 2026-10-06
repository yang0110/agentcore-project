# Progress / Handoff: Terminal-Bench RL on AgentCore

_Last updated: 2026-10-06. Branch: `master` (the only branch)._

## State

The repo is one demo: **`code/demos/demo_terminalbench_kernelbench`** (named `demo_hybrid_terminalbench`
until 2026-10-06), agentic RL on all of Terminal-Bench plus KernelBench with slime and Qwen3-4B
(fully-async GRPO, 4 train + 4 inference GPUs on the 8×L40S box). Each task is routed to one of three
substrates:

- **AgentCore microVMs:** `acs_tbh_<task>` arm64 VPC Runtimes, one fresh session per episode, built
  from the shared `acs-tb-<task>` ECR images.
- **m5 EC2:** compose, x86-only, > 2 GB and microVM-oracle-failing tasks, one m5 per session
  (warm pool, reused after a verified reset). The builder `acs-tbh-m5` builds the images and pushes
  them to ECR `acs-tbhdk-*`.
- **g6 GPU Instances:** KernelBench, Runtime `acs_tbhgpu_kernelbench` on an AgentCore capacity
  provider.

Usable: 191 Terminal-Bench tasks (163 microVM + 28 m5) + KernelBench. The microVM-only
`demo_cpu_terminalbench` was **removed on 2026-10-06** (folder, `convergence/20_cpu_terminalbench.sh`,
its 218 `acs_tb_*` Runtimes and `acs_warmstart_cpu`); the `acs-tb-*` ECR repos stay because
`acs_tbh_*` uses them.

Latest runs: train48 run 2 (2026-10-05) finished 48/48 steps with healthy infra but almost no GRPO
gradient. Partial-credit training reward and a history-based task filter (197 → 121 train rows)
followed; smoke 8 passed (eval: KernelBench 0.5, microVM 2/62, m5 0/8). Open: m5 tasks have never
been solved; g6 start wait (~18 min p50 in smoke 8) is the bottleneck. Details in the demo README and
`docs/terminalbench-kernelbench-demo-plan.md`.

### Before 2026-10-06 (microVM-only demo, removed)

- **Pool:** 218 arm64 VPC microVM Runtimes (`acs_tb_<task>`), one per task, each built from the
  task's own Dockerfile plus the AgentCore server layer. Of 229 single-container tasks, 7 fail to
  build for arm64 and 4 exceed the 2 GB image cap (`terminalbench/tasks/images.csv`).
- **Data:** `data/terminalbench_{train,eval}.jsonl`, 134 train / 31 eval rows. Step 1 keeps only the
  deployed and oracle-verified tasks, split by task.
- **Episode:** one fresh session per episode; look-ahead pre-open (`TB_PREOPEN_BATCHES=2`); heartbeat
  on by default; the mini_claude loop runs in the session and dials the trainer's `AnthropicAdapter`
  back (SG→SG, ports 18100-18163); harness tier `HarnessPool` (W = RB×NS workers, least-loaded
  sticky engine). The reward is the task's hidden pytest oracle.
- **Cold start** (`terminalbench/tasks/coldstart.csv`, probe of all 218 Runtimes, 2026-09-29): first
  open p50 0.60 s / p95 1.24 s (217/218 ok, all warm hits); burst of 872 opens p50 4.7 s / p95 16.3 s.

## The 2026-09-30 trim

The codebase was cut from 6 demos to 1. **Removed:** `demo_cpu_tasks`, `demo_gpu_single`,
`demo_cpu_instance`, `demo_cpu_branching`, `demo_gpu_farm`, `demo_cpu_swebench` (20 arm64 SWE-bench Verified
instances, reward 0 with Qwen3-4B; the 404-instance expansion was dropped) with its EKS/ECS twins,
convergence wrappers 01-19 and 21 (only `20_cpu_terminalbench.sh` remains), the multi-demo smoke
drivers, and the old `code/agentcore_slime_demo`, `code/deploy`, `code/scripts` and `code/tests`.
Their AWS resources are being deleted. Everything is still in git history.

The docs were rewritten around Terminal-Bench under the title "Agentic RL on Terminal-Bench with
slime and AgentCore Runtime": `README.md`, this file, the report (`docs/report/`) and the deck
(`docs/slides/aws/terminalbench.tex`, then the only deck; on 2026-10-06 it was folded into
`docs/slides/aws/terminalbench_kernelbench/deck.tex`, now the only deck). The design notes in `docs/*.md` that
describe removed demos carry a dated "historical" banner. They were not rewritten.

## What's next

1. **Training run on the filtered task set** (`demo_terminalbench_kernelbench`, partial-credit reward,
   121 train rows): check that GRPO groups now carry a gradient (`train/pg_loss` != 0).
2. **m5 tasks score 0** in every run so far: tell apart "too hard for Qwen3-4B" from a setup bug.
3. **g6 start wait:** a higher g6 cap or an S3-hosted torch wheel (cuts the ~90 s install at boot).
4. The builder `acs-tbh-m5` is only needed for image builds; stop it between builds.

Earlier (microVM-only demo, done): the 2026-09-30 smoke re-run passed (10 steps + eval, 22 min,
0 `infra_error`, `eval/terminalbench` 1/62) after making `get_tb_pool` one pool per event loop; the
200-step convergence run `conv200_t16` ran 2026-10-03.

## Brief history

- **2026-10-06:** `demo_cpu_terminalbench` removed (AWS Runtimes too); `demo_hybrid_terminalbench`
  renamed `demo_terminalbench_kernelbench`, now the only demo.
- **2026-10-04/05:** hybrid demo built and run (microVM + m5 + g6 KernelBench), train48 48/48.
- **2026-09-30:** trimmed to `demo_cpu_terminalbench` only (SWE-bench removed too).
  Terminal-Bench split back out of `demo_cpu_tasks`. Docs retitled. The Terminal-Bench smoke re-run passed
  (10 steps + eval, 22 min, 0 infra errors, eval 1/62) after the per-loop pool fix.
- **2026-09-29:** cold-start probe of all 218 Terminal-Bench Runtimes.
- **2026-09-28:** harness in the sandbox session in every demo (in-trainer harness removed); harness
  tier (`HarnessPool`) the default; fully-async the only rollout driver (`MODE` removed);
  Terminal-Bench merged into `demo_cpu_tasks` for two days; SWE-bench moved to arm64 microVMs.
- **2026-09-26:** all Runtimes in VPC mode; heartbeat on by default.
- **2026-09-25:** demos made self-contained (vendored lib, `deploy/`, numbered step scripts).
- **2026-09-23:** first Terminal-Bench smoke passed on AWS (382 episodes, 0 infra errors, boot fully
  hidden by pre-open, but about 95 % of pre-opened microVM time idle; reward 4/382).
- **2026-09-14 to 09-22:** earlier demo sets (MBPP/KernelBench, real-CLI, two-Runtime, GPU farm and
  others), all since removed.
