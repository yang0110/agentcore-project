# Demos — the sandbox demo set

Every demo here is about **the sandbox**. The agent harness stays **in-process**: a warm
`run_episode` loop next to the trainer. Each demo varies the **task** and the **AgentCore sandbox
substrate** it talks to. The policy is always Qwen3-4B trained by slime (async/decoupled GRPO).

Each demo directory is **self-contained**. It owns its `data/`, `ckpt/`, `run.sh`, a vendored
copy of `agentcore_slime_demo/` (`PYTHONPATH` puts the demo dir first), and its own `deploy/`. The
`deploy/` dir holds the Dockerfile(s), a vendored AgentCore helper (`agentcore_deploy.py`), and the
settings for the step scripts (`demo.env`). A reader can follow one folder end to end: dataset,
Dockerfile, image build, ECR push, Runtime deploy, run, cleanup. Two things are referenced rather
than copied: the base Megatron checkpoint (`code/p4_slime_train/ckpt/Qwen3-4B_torch_dist`) and the
slime install (`/root/slime`) in the `agentcore_p4` container.

## Run any demo step by step

Every demo ships the same six numbered scripts. Run them from the demo dir **on the host**. The
steps that belong in the container (data prep, training) `docker exec` into `agentcore_p4`
themselves, and they also work if you are already inside the container.

```bash
cd code/demos/<demo>
bash 1_prepare_data.sh     # download + prepare data/ (container)
bash 2_build_image.sh      # docker build the sandbox image(s) (host; arm64 via buildx for microVMs, amd64 for Instances)
bash 3_push_ecr.sh         # create the ECR repo(s) + push -> deploy/images.env
bash 4_deploy_runtime.sh   # create the AgentCore Runtime(s) -> deploy/runtime.env (ARNs for run.sh)
bash 5_run.sh              # 10 train steps + 1 eval + final ckpt (container), then metrics.py -> logs/<tag>.{csv,eval.csv,png}
bash 6_cleanup.sh          # delete the Runtime(s) (and the capacity provider with DELETE_CP=1)
```

Knobs pass straight through, for example `DATASET=gsm8k bash 5_run.sh` or `NR=51 bash 5_run.sh`.
Without a `deploy/runtime.env`, `5_run.sh` falls back to the defaults in `run.sh`. For most CPU demos
those are the baked shared `acs_sandbox_cpu` ARNs.

## The demos

There are **6 demos**: 5 main (3 CPU + 2 GPU) plus one experimental.

- `demo_cpu_tasks` trains **any of ~19 datasets, or any mix**, via `DATASET=`. `VARIANT=bigimage`
  moves the same trainer onto a large image on a CPU capacity-provider Instance. `VARIANT=imagepool`
  routes each family to its own custom image. `compare_backends.py` shows the custom-vs-managed
  Runtime contrast.
- `demo_gpu_single` covers the warm g6 GPU sandbox, the custom-Runtime angle and snapshot/resume
  (`SNAPSHOT_EVERY=1`).
- `demo_gpu_farm` covers the local GPU-sharing splits (`FARM=fractional|whole`).
- **`HEARTBEAT=1` works in every demo**: an opt-in `SessionKeeper` that keeps idle sandbox sessions
  warm across the weight-sync barrier. It defaults to off, which leaves the run unchanged.
  `demo_cpu_terminalbench` always runs its own keeper for its pre-opened sessions.

| Substrate | Demo | Use case |
|---|---|---|
| **CPU serverless microVM** | [`demo_cpu_tasks/`](demo_cpu_tasks/) | **The reference.** RLVR on any dataset: real HF `mbpp` (default) `gsm8k` `math500` `deepscaler` `humaneval` `gretelsql` `hotpotqa` `xlam` `mcp`, plus the synthetic family zoo (`math` `swe` `sql` `ctf` `rag` `tool_use` …). **Mixed-task** batches via `DATASET=a,b,c` or `DATASET=mixed` |
| | `demo_cpu_tasks` `VARIANT=imagepool` | Mixed-task RL over a **pool of custom images** (pandas / scipy / sklearn), one Runtime per image, routed per family via `AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE` |
| | `demo_cpu_tasks` `compare_backends.py` | The same episode on **your own** custom CPU Runtime and the managed sandbox: same reward, and a table of the operational differences |
| **CPU capacity-provider Instance** | `demo_cpu_tasks` `VARIANT=bigimage` | A ~4 GB image (over the 2 GB microVM cap) on a warm m5. Warm-shared session, fresh env per episode |
| **Large image pool** | [`demo_cpu_terminalbench/`](demo_cpu_terminalbench/) | One image + Runtime per Terminal-Bench task (~229), a fresh microVM per episode, pre-opened to hide the cold start |
| **Rollout scaling** | [`demo_cpu_scale_lanes/`](demo_cpu_scale_lanes/) | **4 × 4 × 4 lanes**: 4 harness workers : 4 SGLang engines : 4 persistent microVMs, pinned 1:1:1. Shows `throughput = min(inference, sandbox, harness)` |
| **GPU capacity-provider Instance** (g6) | [`demo_gpu_single/`](demo_gpu_single/) | KernelBench CUDA RL on **one warm custom GPU Runtime**. Only a custom Runtime has CUDA (`gpu_torch_smoke.py`). `SNAPSHOT_EVERY=1` adds app-level snapshot/resume. `LOCAL_GPU=1` runs the on-box farm |
| **Local GPU Ray farm** (GPUs 6,7) | [`demo_gpu_farm/`](demo_gpu_farm/) | GPU sharing: `FARM=fractional` (two episodes on ONE GPU, 0.5 + 0.5) or `FARM=whole` (a whole GPU each across 6,7) |
| *Experimental* | [`demo_cpu_branching/`](demo_cpu_branching/) | One episode → many samples ("fan-out") with the real Claude Code CLI in the sandbox. **Blocked:** the in-sandbox CLI exits 1 before any model call, so there are no trajectories yet |

**Merge history.**
- 2026-09-23: the ten per-dataset CPU folders (`demo_cpu_mbpp` … `demo_cpu_mcp`) became `demo_cpu_tasks`.
- 2026-09-25: `demo_cpu_mixtask` → `demo_cpu_tasks DATASET=mixed`.
- 2026-09-25: `demo_gpu_shared_single`, `demo_gpu_shared_multi` and `demo_gpu_local` → `demo_gpu_farm`.
- 2026-09-25: `demo_cpu_bigimage` → `demo_cpu_tasks VARIANT=bigimage`.
- 2026-09-25: `demo_gpu_snapshot` and `demo_gpu_customruntime` → `demo_gpu_single`.
- 2026-09-25: `demo_cpu_heartbeat` → the `HEARTBEAT=1` knob in every demo (probe: `demo_cpu_tasks/heartbeat_demo.py`).
- 2026-09-25: `demo_cpu_customruntime` → `demo_cpu_tasks/compare_backends.py`; `demo_cpu_imagepool` → `demo_cpu_tasks VARIANT=imagepool`.

Older logs and docs keep the old names.

**Sandbox mode by substrate.** The CPU demos are **remote-only**. Their `run.sh` rejects
`SANDBOX=local`, so a CPU run can never silently fall off AgentCore. The GPU demos keep an on-box
path (`LOCAL_GPU=1`, or `demo_gpu_farm`), because a remote g6 may be unavailable.

## Which to run first

- **`demo_cpu_tasks`**: the reference and the most battle-tested. Try `DATASET=gsm8k bash 5_run.sh`.
- **`demo_gpu_single`**: the GPU-sandbox path. The KernelBench checker's GPU is the remote g6, so this
  box stays 4 train + 4 infer.
- **`demo_gpu_farm`**: GPU sharing on this box with no AWS GPU needed.
- **`HEARTBEAT=1`** and **`demo_gpu_single SNAPSHOT_EVERY=1`**: the resilience angles. Their
  offline probes (`demo_cpu_tasks/heartbeat_demo.py`, `demo_gpu_single/snapshot_demo.py`) run in
  seconds with no AWS or GPU.

All trainers share one training path: slime `train_async.py`, Qwen3-4B, GRPO, the
checkpoint-deadlock fix and save-only-at-end. They differ only in the **task** and the **sandbox**.

**`run.sh` vs `train.sh`.** `run.sh` is the full runner (all env knobs; defaults to a short smoke).
Three demos also ship **`train.sh`**, a train-to-convergence wrapper: `demo_cpu_tasks` (all
datasets and variants), `demo_cpu_terminalbench` and `demo_gpu_single` (`demo_cpu_branching` has one
too, but it is blocked). `train.sh` exposes `STEPS` → `NR`, `EVAL_EVERY` → `EVAL_INTERVAL`,
`SANDBOX_CONC` → `CONC` and, on CPU, `SANDBOX_RUNTIMES` → `K`. For the 100–200-step runs, see
[`convergence/`](convergence/).
