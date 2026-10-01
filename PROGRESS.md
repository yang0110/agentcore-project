# Progress / Handoff: Terminal-Bench RL on AgentCore

_Last updated: 2026-09-30. Branch: `master` (the only branch)._

## State

The repo is one demo: **`code/demos/demo_cpu_terminalbench`**, agentic RL on Terminal-Bench with
slime and Qwen3-4B (fully-async GRPO, RB=4 × NS=8 = 32 episodes per step, 4 train + 4 inference
GPUs on the 8×L40S box).

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
(`docs/slides/aws/terminalbench.tex`, the only deck). The design notes in `docs/*.md` that
describe removed demos carry a dated "historical" banner. They were not rewritten.

## What's next

1. **Terminal-Bench smoke: done (2026-09-30 re-run).** `bash 5_run.sh` ran 10 steps plus the final
   eval in 22 min (`logs/smoke_tbonly-agentcore-runtime_20260930_153115.{log,csv,eval.csv,png}`):
   449 episodes, 0 `infra_error`, 22/387 train solved, `eval/terminalbench` 0.016 (1/62). No train
   episode waited for a boot; 4 of 49 GRPO groups had a gradient. It needed one fix: eval used to
   hang after its last episode, because `get_tb_pool` was a process-wide singleton bound to the
   fully-async worker's loop. It is now one pool per event loop (`terminalbench_pool.py`).
2. **Convergence run 20.** Inside `agentcore_p4`:
   `bash /work/code/demos/convergence/20_cpu_terminalbench.sh` (NR=201, eval every 20 steps, a
   checkpoint every 100 steps at about 56 GB each). Never run two `run.sh` at once.
3. Open items from the smokes: reward is sparse, so consider an easy-first curriculum or
   partial-credit reward (`tests_passed / tests_total` is already logged). `TB_PREOPEN_BATCHES=1`
   would reduce idle pre-opened microVM time.

## Brief history

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
