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
Without a `deploy/runtime.env`, `5_run.sh` falls back to the defaults in `run.sh`. For
`demo_cpu_tasks` that is the managed Code Interpreter, where no Runtime is involved.

## Which sandbox does a task need? Three tiers

Every demo sits in one sandbox tier. **Pick the lowest tier that works.** Go to tier 2 only for a
run-level need, and to tier 3 only when the task's environment itself must change. A task that runs
on tier 1 gives the same reward on tier 2 (`demo_cpu_tasks/compare_backends.py` asserts it).

| Tier | Sandbox | Use it when | Tasks and runs here | Demos |
|---|---|---|---|---|
| **1 · Managed Code Interpreter** | `SANDBOX=agentcore`: AWS-managed microVM, **no image, no Runtime, no IAM role** | Python 3.12 with numpy, pandas, scipy, scikit-learn, torch (CPU), sqlite, gcc and node is enough; no git, internet or GPU; a session fits the hard `sessionTimeoutSeconds` cap (≤ 8 h) | **every literature task**: `mbpp`, `humaneval`, `gsm8k`, `math500`, `deepscaler`, `gretelsql`, `hotpotqa`, `xlam`, `mcp`, `libs` (pandas/scipy/sklearn), the synthetic zoo (13 families), mixes | `demo_cpu_tasks` (default) |
| **2 · Own Runtime, generic image** | `SANDBOX=agentcore-runtime` with the stdlib `deploy/Dockerfile.sandbox.cpu` (microVM) | the **task** fits tier 1 but the **run** needs Runtime behaviour: more than ~75 episodes in flight (`InvokeCodeInterpreter` is 30/s; a Runtime's data plane is far higher), sessions kept alive past the idle timeout by heartbeat, or network/VPC control | DeepScaleR over 256 pooled sessions; the heartbeat and pre-open probes | `demo_cpu_scale_lanes`; `demo_cpu_tasks` with `SANDBOX=agentcore-runtime` / `HEARTBEAT=1`; `demo_cpu_terminalbench/warmstart/` |
| **3 · Own Runtime, custom image** | `SANDBOX=agentcore-runtime` or `agentcore-ray-farm` with a task-specific image | the **environment** differs | Terminal-Bench: own OS image per task (microVM) · Claude Code CLI: node 22 + `claude` (microVM) · torch + sci stack ≈ 4 GB > 2 GB cap (**CPU Instance**, m5) · KernelBench CUDA (**GPU Instance**, g6; the farm demos run the same image on this box's GPUs 6,7) | `demo_cpu_terminalbench`, `demo_cpu_branching`, `demo_cpu_instance_pool`, `demo_gpu_single`, `demo_gpu_farm_fractional`, `demo_gpu_farm_whole` |

## The demos

There are **7 demos plus one experimental**, each standalone. They are listed in storyline order:
the tasks and where they come from, then training on the default runtime, custom images at scale,
hiding the container launch, a warm CPU instance pool, rollout scaling, and finally GPUs.

| # | Demo | Tier · substrate | What it shows |
|---|---|---|---|
| 1 | [`demo_cpu_tasks/`](demo_cpu_tasks/) | **Tier 1.** **Default AgentCore sandbox**: the managed Code Interpreter (`SANDBOX=agentcore`), no image | **Most literature tasks train on the default runtime as-is.** Its README's task table gives each dataset's source URL, env deps and image need. Datasets: HF `mbpp` (default), `gsm8k`, `math500`, `deepscaler`, `humaneval`, `gretelsql`, `hotpotqa`, `xlam`, `mcp`, pandas/scipy/sklearn tasks (`DATASET=libs`), the synthetic zoo, and mixes (`DATASET=a,b,c`, `DATASET=mixed`). Steps 2/3/4/6 are no-ops. `compare_backends.py` covers the own-Runtime contrast |
| 2 | [`demo_cpu_terminalbench/`](demo_cpu_terminalbench/) | **Tier 3.** Custom images at scale: one image + Runtime per Terminal-Bench task (~229), a fresh microVM per episode | **Custom images in large numbers**, with the container launch hidden. The look-ahead pre-open (`TB_PREOPEN_BATCHES=2`) boots the next 1–2 batches' sessions, and an always-on heartbeat keeps them alive. The `warmstart/` probe (`bash warmstart/run_warmstart.sh`) measures cold vs pre-open vs heartbeat on its own, in about 7 min with no GPU |
| 3 | [`demo_cpu_instance_pool/`](demo_cpu_instance_pool/) | **Tier 3.** CPU capacity-provider Instance (m5): a ~4 GB image, over the 2 GB microVM cap | **A warm instance pool shared by sessions** (`SANDBOX_POOL=N`, default 2): each boot is paid once per instance, and every episode is an isolated sub_session. See the isolation note below |
| 4 | [`demo_cpu_scale_lanes/`](demo_cpu_scale_lanes/) | **Tier 2.** CPU microVMs, generic stdlib image (256 sessions exceed the Code Interpreter's 30/s invoke quota) | **Rollout scaling on decoupled tiers**: 4 SGLang engines : 256 in-process harness workers : 256 pooled microVM sessions, sized independently and joined per episode. `throughput = min(inference, sandbox, harness)` |
| 5 | [`demo_gpu_single/`](demo_gpu_single/) | **Tier 3.** GPU capacity-provider Instance (g6), one warm custom Runtime | KernelBench CUDA RL. Only a custom Runtime has CUDA (`gpu_torch_smoke.py`). `SNAPSHOT_EVERY=1` adds app-level snapshot/resume |
| 6 | [`demo_gpu_farm_fractional/`](demo_gpu_farm_fractional/) | **Tier 3.** Ray GPU farm, **one GPU split in fractions** (GPU 6, 0.5 + 0.5) | Two episodes share one GPU: separate CUDA contexts, with a per-process memory fraction |
| 7 | [`demo_gpu_farm_whole/`](demo_gpu_farm_whole/) | **Tier 3.** Ray GPU farm, **one whole GPU per session** (GPUs 6,7) | One shared warm instance, one GPU per sub_session |
| exp. | [`demo_cpu_branching/`](demo_cpu_branching/) | **Tier 3.** CPU Runtime whose image bakes node 22 + the real Claude Code CLI | One episode → many samples ("fan-out"). RL smoke 2026-09-25 ran end to end (fan-out p50 4, eval 0.47) but every advantage was 0; grouping bug fixed, re-run pending |

**Heartbeat.** `HEARTBEAT` is a `SessionKeeper` in every demo. It keeps idle sessions on **your own
Runtime** warm across the weight-sync barrier and other trainer stalls. It is **on by default** whenever
the sandbox is a remote Runtime (`SANDBOX=agentcore-*` with no local endpoint) and off otherwise; set
`HEARTBEAT=0` for the A/B baseline. It is a no-op on the Code Interpreter: those sessions end at their session timeout whatever
the traffic, so the lever there is `AGENTCORE_CI_SESSION_TIMEOUT`. `demo_cpu_terminalbench` always runs
its own keeper. Without pings, a reclaimed Runtime session silently returns a fresh, empty sandbox
under the same session id (measured by the warmstart probe).

**Idle timeout.** Runtimes created by `deploy/agentcore_deploy.py` get `idleRuntimeSessionTimeout =
RUNTIME_IDLE_TIMEOUT` (default **3600 s**, range 60–28800), sized above the worst trainer stall; it is
the backstop if the heartbeat does not fire. It applies only when a Runtime is created: an existing one
keeps its old timeout (600 s for Runtimes made before 2026-09-26; step 4 prints a note), so recreate
it with `FORCE_RECREATE=1`. `HEARTBEAT_IDLE` stays 600 s, which is safe for both. The cost is that an
abandoned session is billed for up to the timeout; normal episode ends call `StopRuntimeSession`.

**Isolation on a shared instance (demos 3, 6, 7).** Several episodes share one warm instance, so
there is no VM boundary between them. The farm puts back as much isolation as one kernel allows:

- **CPU pool (demo 3): a Landlock jail per exec.** The AgentCore container drops every capability
  (no `setuid`/`chown`), so per-episode uids are impossible there. Each exec may write only its own
  dir, can't read sibling dirs, `/tmp` or `/dev/shm`, can't open TCP connections, and can't signal or
  inspect other episodes' processes. 22/22 checks pass live on the m5 (`isolation_probe.py --live`).
- **GPU farms (demos 6, 7): its own uid per sub_session** (`20000+slot`) with a private `0700` dir,
  where the container has the capabilities (the local farm); on the AgentCore g6 they fall back to one
  shared uid. The CPU pool also adds a uid on top of Landlock where it can.
- **Limits:** process-count, RAM and file-size caps, and a timeout kills the whole process tree.
- **Server file ops** refuse `..`, absolute and symlink escapes.
- **No leaks:** `AWS_*` is not inherited, and the session id never appears on a command line.
- **Close** kills the episode's processes and deletes its dir.

The episodes still share the kernel, the network namespace, and CPU/RAM (there is no per-episode
cgroup). On the GPU farms they also share the GPU: CUDA contexts are separate, but the memory fraction
is cooperative, and with no MIG on L4/L40S a GPU fault hits every co-tenant. For a hard boundary use a
microVM per session (demos 1, 2, 4). Each shared-instance demo's README has the full table and its
probe results.

**Merge history.**
- 2026-09-23: the ten per-dataset CPU folders (`demo_cpu_mbpp` … `demo_cpu_mcp`) became `demo_cpu_tasks`.
- 2026-09-25: `demo_cpu_mixtask` → `demo_cpu_tasks DATASET=mixed`; `demo_cpu_heartbeat` → the
  `HEARTBEAT=1` knob; `demo_cpu_customruntime` → `demo_cpu_tasks/compare_backends.py`;
  `demo_gpu_snapshot` + `demo_gpu_customruntime` → `demo_gpu_single`.
- 2026-09-25, storyline restructure: `demo_cpu_tasks` defaults to the Code Interpreter;
  `demo_cpu_imagepool` (briefly `VARIANT=imagepool`) → `DATASET=libs` on the Code Interpreter, which
  already has pandas/scipy/sklearn; `demo_cpu_bigimage` (briefly `VARIANT=bigimage`) →
  `demo_cpu_instance_pool`; the GPU farms (`demo_gpu_shared_single/_multi`, `demo_gpu_local`, briefly
  `demo_gpu_farm FARM=`) → `demo_gpu_farm_fractional` + `demo_gpu_farm_whole`; the short-lived
  `demo_cpu_warmstart` → `demo_cpu_terminalbench/warmstart/`.

Older logs and docs keep the old names.

**Sandbox mode by substrate.** The CPU demos are **remote-only**. Their `run.sh` rejects
`SANDBOX=local`, so a CPU run can never silently fall off AgentCore. The GPU farms run on this box's
GPUs 6/7 (`code/deploy/run_local_gpu_runtime.sh fractional|whole`, or step 4), because a remote g6 may
be unavailable. `demo_gpu_single` keeps `LOCAL_GPU=1` for the same reason.

## Which to run first

- **`demo_cpu_tasks`**: nothing to build or deploy. Try `DATASET=gsm8k bash 5_run.sh`
  (eval/gsm8k 0.90 on the Code Interpreter, 2026-09-25).
- **`demo_cpu_terminalbench/warmstart/run_warmstart.sh`**: the launch-cost story in minutes, no GPU.
- **`demo_cpu_instance_pool`**: `python3 isolation_probe.py` checks the shared-instance isolation
  offline.
- **`demo_gpu_single`** / **`demo_gpu_farm_fractional`**: the GPU paths.
- The offline probes `demo_cpu_tasks/heartbeat_demo.py` and `demo_gpu_single/snapshot_demo.py` run in
  seconds with no AWS or GPU.

All trainers share one training path: slime `train_async.py`, Qwen3-4B, GRPO, the
checkpoint-deadlock fix and save-only-at-end. They differ only in the **task** and the **sandbox**.

**`run.sh` vs `train.sh`.** `run.sh` is the full runner (all env knobs; defaults to a short smoke).
Three demos also ship **`train.sh`**, a train-to-convergence wrapper: `demo_cpu_tasks` (all
datasets), `demo_cpu_terminalbench` and `demo_gpu_single` (`demo_cpu_branching` has one
too, but it is blocked). `train.sh` exposes `STEPS` → `NR`, `EVAL_EVERY` → `EVAL_INTERVAL`,
`SANDBOX_CONC` → `CONC` and, on CPU, `SANDBOX_RUNTIMES` → `K`. For the 100–200-step runs, see
[`convergence/`](convergence/).
