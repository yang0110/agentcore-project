# `convergence/` — train each demo TO CONVERGENCE, one at a time

The demo `run.sh` scripts elsewhere in `code/demos/` are tuned for a **10-step smoke**
(`NR=11`): just enough to prove the pipeline runs end-to-end and rewards are non-zero.
They are **not** meant to actually train the policy to a plateau.

This directory is the opposite: **one well-documented wrapper per trainable demo that
sets convergence-scale knobs and launches the real run.** You run them **manually, one
by one** — these are long, GPU-hungry jobs, not a sweep to fire in parallel.

Each wrapper is a thin, override-friendly shell around the demo's own `run.sh`: it
exports the convergence knobs (all `${VAR:=default}`, so any of them can be overridden
from the environment) and then execs the demo's `run.sh`, teeing everything to a
timestamped log next to that demo's `ckpt/`.

```
convergence/
  README.md            <- you are here (the runbook)
  commands.md          <- copy-paste cheat-sheet, one line per demo
  01_cpu_mbpp.sh       <- CPU demos (managed remote AgentCore sandbox, SANDBOX=agentcore-runtime);
                          01–10 all train demo_cpu_tasks, one DATASET each (mbpp … mcp)
  02_cpu_math.sh
  03_cpu_gsm8k.sh
  04_cpu_math500.sh
  05_cpu_deepscaler.sh
  06_cpu_humaneval.sh
  07_cpu_gretelsql.sh
  08_cpu_hotpotqa.sh
  09_cpu_xlam.sh
  10_cpu_mcp.sh
  11_cpu_mixed.sh      <- demo_cpu_tasks DATASET=mixed (the multi-family batch)
  12_gpu_single.sh     <- GPU demos (KernelBench; CUDA sandbox; remote g6 or LOCAL_GPU=1)
  13_gpu_farm_fractional.sh  <- demo_gpu_farm FARM=fractional (local, GPU 6 shared 0.5+0.5)
  14_gpu_farm_whole.sh       <- demo_gpu_farm FARM=whole (local, GPUs 6,7, one GPU each)
  15_cpu_heartbeat.sh    <- property demos, now real trainers (headline knob default-ON;
  16_cpu_scale_lanes.sh    set it OFF => byte-identical to the plain single-substrate trainer)
  17_cpu_customruntime.sh
  18_gpu_single_snapshot.sh  <- demo_gpu_single SNAPSHOT_EVERY=1 (was demo_gpu_snapshot)
```

## Where you run these

**All wrappers run INSIDE the persistent slime container `agentcore_p4`** (the repo is
mounted at `/work/code`), exactly like the demo `run.sh` they wrap. Launch from the host
with something like:

```bash
docker exec -it agentcore_p4 bash -lc \
  'cd /work && bash code/demos/convergence/03_cpu_gsm8k.sh'
```

(See [how-to-launch-demo-runs] in project memory: the host launch dies on a
`/root/slime` model-source permission error — you must be inside the container.)

## Prerequisites per tier

**CPU demos (01–11)** talk to a **remote managed AgentCore sandbox Runtime**
(`SANDBOX=agentcore-runtime`; `SANDBOX=local` is rejected for CPU demos). Each demo's
`run.sh` bakes in the default K=2 CPU sandbox ARNs
(`acs_sandbox_cpu-…,acs_sandbox_cpu_1-…`). You need valid AWS creds in the container
(`AWS_BEARER_TOKEN_BEDROCK` / region). Override the endpoints with
`AGENTCORE_SANDBOX_RUNTIME_ARNS` if yours differ.

**GPU-single (12)** is **remote-only**: it *requires* `AGENTCORE_SANDBOX_RUNTIME_ARN`
pointing at a live **g6** CUDA sandbox Runtime (one dedicated GPU per session). Without a
real g6 ARN, use the local farm runs (13/14) instead.

**GPU-farm (13, 14)** train `demo_gpu_farm` against a Ray farm on **this box's reserved
GPUs** — 13 is `FARM=fractional` (GPU 6, two 0.5-GPU slots), 14 is `FARM=whole` (GPUs 6,7,
one GPU each). You must first start the local runtime container **on the HOST** (not
inside `agentcore_p4`) in the matching mode:

```bash
bash code/deploy/run_local_gpu_runtime.sh fractional   # for 13: reserves GPU 6
bash code/deploy/run_local_gpu_runtime.sh whole        # for 14: reserves GPUs 6,7
# then, inside agentcore_p4:
bash code/demos/convergence/13_gpu_farm_fractional.sh  # or 14_gpu_farm_whole.sh
# tear down when finished:
bash code/deploy/run_local_gpu_runtime.sh stop
```

Both preflight the local `/ping` and refuse to start if the farm is down.

## The convergence knobs (what changed vs the smoke)

| knob | smoke | convergence (CPU / GPU default) | why |
|------|------:|-------:|-----|
| `NR` (num-rollout) | 11 | **201 / 101** | real train steps = `NR-1` ⇒ ~200 (CPU) / ~100 (GPU) optimizer steps |
| `RB` (rollout-batch-size) | 4 | **8** | more prompts/step ⇒ steadier gradient |
| `NS` (n-samples-per-prompt) | 8 | **8** | GRPO group size — kept at 8 (group-relative advantage needs a group) |
| `TEMP` | 1.0 | **1.0** | keep intra-group reward variance (GRPO signal is 0 when a group is uniform) |
| `EVAL_INTERVAL` | 10 | **20** | held-out eval every 20 steps; large enough that Mode-B re-warm cost is amortized |
| `EVAL_NS` | 2 | **4** | more eval samples ⇒ lower-variance solve-rate estimate |
| `SAVE_INTERVAL` | 1e6 (end only) | **100** | keep periodic checkpoints so you can resume / pick the best — **see disk caveat** |
| `MODE` | B | **B** (CPU only) | fully-async deep buffer. GPU demos are stock 1-step-async (no MODE knob) |
| `CONC` | 4 (2 for local) | demo default | sandbox/episode concurrency; GPU-local pinned to the farm's whole-GPU slot count (2) |

Every value is overridable, e.g. `NR=401 RB=16 bash 03_cpu_gsm8k.sh`.

### ⚠ Disk caveat (read before setting SAVE_INTERVAL low)

A single Qwen3-4B checkpoint is **~56 GB**. With `SAVE_INTERVAL=100` a 200-step CPU run
writes checkpoints at step 100 and step 200 (~112 GB); the final rollout always
force-saves. `/` on this box is ~1.9 TB, so **run one demo at a time and clean each
demo's `ckpt/` between runs**. If you only care about the convergence *curve* (the eval
solve-rate, logged every `EVAL_INTERVAL` — you do not need the weights), set
`SAVE_INTERVAL=1000000` to save **only at the end**:

```bash
SAVE_INTERVAL=1000000 bash 03_cpu_gsm8k.sh
```

## How to tell it converged

Convergence is judged from the **held-out eval curve**, not the checkpoints. Every
`EVAL_INTERVAL` steps the run logs `eval/<name>` (mean held-out reward = solve-rate) to
the console and to wandb (if configured). You are converged when:

1. **`eval/<name>` rises then plateaus** across several eval points (the primary signal).
2. **`train reward mean` rises** and its **per-group variance shrinks** (GRPO is learning
   the easy modes; a group whose samples all get the same reward contributes 0 gradient).
3. The eval curve stops improving for ~3+ consecutive eval points ⇒ stop (or you've hit
   `NR`).

Watch it live with:

```bash
tail -f code/demos/demo_cpu_tasks/train.convergence_gsm8k_*.log | grep -E 'eval/|reward'
```

## Property demos (15–19) — now trainable too

The five demos that were formerly offline sims / contract probes are now **real slime
trainers** (each layers its distinctive operational property on top of the standard
trainer as env-gated glue that stays live during the run). They therefore get
convergence wrappers like everything else. Each demo's **headline knob defaults ON**;
setting it to its off value makes the run byte-identical to the plain single-substrate
trainer (`demo_cpu_tasks` / `demo_gpu_single`), so the wrapper doubles as an A/B for the
property's effect on convergence:

| # | wrapper | substrate | property (default-on knob) |
|---|---------|-----------|-----------------------------|
| 15 | `15_cpu_heartbeat.sh` | CPU sandbox (baked ARNs) | keep the session warm across the weight-sync barrier (`HEARTBEAT_KEEP_WARM=1`) |
| 16 | `16_cpu_scale_lanes.sh` | CPU sandbox (baked ARNs) | 4×4×4 lanes: 4 warm harness workers : 4 SGLang engines : 4 persistent microVMs (`LANES`; MODE pinned to A) |
| 17 | `17_cpu_customruntime.sh` | your **custom CPU** Runtime | training runs unchanged on a custom Runtime vs the managed Code Interpreter |
| 18 | `18_gpu_single_snapshot.sh` | remote **g6** or `LOCAL_GPU=1` | `demo_gpu_single` with app-level workspace snapshot/resume every episode (`SNAPSHOT_EVERY=1`) |

(The former `19_gpu_customruntime.sh` is gone: `demo_gpu_customruntime` was merged into
`demo_gpu_single`, so `LOCAL_GPU=1 bash 12_gpu_single.sh` is the same run.)

The retained seconds-long **offline probes** (`heartbeat_demo.py`, `snapshot_demo.py`,
`harness_scaling_demo.py`, `gpu_torch_smoke.py`) are NOT trainers — invoke them directly
for a fast legibility check, not via these wrappers.

> **Hardware status.** 19's `LOCAL_GPU=1` path is validated on this 8×L40S box; the two
> remote-g6 paths (18, and 19 without `LOCAL_GPU`) are **UNVERIFIED on hardware** — treat
> a first convergence run as a bring-up.

## Scope / what is NOT here

Only **trainable** demos get a convergence script. Still excluded:

- `convergence/` is not for **`demo_cpu_imagepool`** (needs a deployed image→ARN pool) or
  **`demo_cpu_branching`** (needs a live CLI-capable sandbox + bring-up) — add wrappers
  once those substrates are provisioned and smoke-verified.

## Teardown

- CPU demos: nothing host-side; the remote sandbox sessions close per-episode.
- GPU-local: `bash code/deploy/run_local_gpu_runtime.sh stop` to free GPUs 6,7.
- All: `ray stop --force` inside the container if a run left ray/sglang behind (the demo
  `run.sh` also does this at the top of the next run).
- Reclaim disk: remove the finished demo's `ckpt/Qwen3-4B_*` directory once you've copied
  off anything you want to keep.
