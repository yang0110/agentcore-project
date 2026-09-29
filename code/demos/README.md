# Demos — the sandbox demo set

Every demo here is about **the sandbox**. The agent loop (mini_claude) always runs **inside the
episode's sandbox session**: `in_session_harness.py` writes `session_runner.py` into the session, and
the loop dials the trainer's per-engine `AnthropicAdapter` back over the VPC (SG→SG rule on
`ADAPTER_PORT_BASE` 18100-18163). A hidden grader scores the result after the loop stops. There is no
in-trainer harness and no fallback to the trainer: a session that can't host the loop (no `python3`,
or the adapter unreachable) fails the episode with `HarnessPlacementError`. (`demo_cpu_branching`
runs the real Claude Code CLI in the session instead; it dials its adapter back on port 18021.) Each demo varies the
**task** and the **AgentCore sandbox substrate** it talks to. The policy is always Qwen3-4B trained by slime (async/decoupled GRPO).

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
Without a `deploy/runtime.env`, `5_run.sh` uses the defaults in `run.sh`. For `demo_cpu_tasks`
that is the baked shared K=2 `acs_sandbox_cpu` Runtimes (stdlib datasets only).

## Which sandbox does a task need? Three tiers

Every demo sits in one sandbox tier. **Pick the lowest tier that works.** Tier 1 is no longer a
training path (2026-09-28): the Code Interpreter has no route back to the adapter, so the in-session
agent loop can't run there. Training starts at tier 2; go to tier 3 only when the task's environment
itself must change. A task that runs on tier 1 gives the same reward on tier 2
(`demo_cpu_tasks/compare_backends.py` asserts it).

| Tier | Sandbox | Use it when | Tasks and runs here | Demos |
|---|---|---|---|---|
| **1 · Managed Code Interpreter** | `SANDBOX=agentcore`: AWS-managed microVM, no image, no Runtime, no IAM role | **probes only**: no route back to the adapter, so no in-session agent loop | backend comparison and warm-start probes | `demo_cpu_tasks/compare_backends.py`, `demo_cpu_tasks/terminalbench/warmstart/` |
| **2 · Own Runtime, generic image** | `SANDBOX=agentcore-runtime` with a generic image: stdlib `cpu`, or `libs` (+ pandas/scipy/sklearn) (microVM) | the task needs no task-specific environment; the Runtime also gives VPC dial-back to the adapter, a far higher invoke rate than the Code Interpreter's 30/s, and sessions kept alive by heartbeat | **every literature task**: `mbpp`, `humaneval`, `gsm8k`, `math500`, `deepscaler`, `gretelsql`, `hotpotqa`, `xlam`, `mcp`, `libs`, the synthetic zoo (13 families), mixes; the harness-scaling run (`SCALE=1`: DeepScaleR, 256 sessions in flight) | `demo_cpu_tasks`; `demo_cpu_tasks/terminalbench/warmstart/` |
| **3 · Own Runtime, custom image** | `SANDBOX=agentcore-runtime` or `agentcore-ray-farm` with a task-specific image | the **environment** differs | Terminal-Bench: own OS image per task (microVM) · SWE-bench: own repo+env image per instance, arm64 ≤ 2 GB compressed (microVM) · Claude Code CLI: node 22 + `claude` (microVM) · KernelBench CUDA (**GPU Instance**, g6 with `LOCAL_GPU=0`; the default runs the same image on this box's GPUs 6,7) | `demo_cpu_tasks DATASET=terminalbench`, `demo_cpu_swebench`, `demo_cpu_branching`, `demo_gpu_farm` |

## The demos

There are **3 main demos** (plus the EKS and ECS twins `demo_cpu_swebench_eks` / `demo_cpu_swebench_ecs`) and **one experimental**
demo (`demo_cpu_branching`). Each is standalone. They are listed in storyline order: the tasks and
where they come from, then training on the default runtime (and, with `SCALE=1`, rollout scaling),
custom images at scale (microVM, then capacity-provider Instances), hiding the container launch, and
finally GPUs (one GPU instance serving many sessions, locally or on a remote g6). The harness tier
below is the default rollout topology in all of them.

| # | Demo | Tier · substrate | What it shows |
|---|---|---|---|
| 1 | [`demo_cpu_tasks/`](demo_cpu_tasks/) | **Tier 2.** Own Runtime (`SANDBOX=agentcore-runtime`, the only backend), two generic arm64 microVM images: `cpu` (stdlib) and `libs` (pandas/scipy/sklearn) | **Most literature tasks need only a generic sandbox.** Its README's task table gives each dataset's source URL, env deps and image. **Default = a mixed-task batch** (`mbpp,humaneval,gsm8k,math500,deepscaler,gretelsql,hotpotqa,xlam,mcp,terminalbench` = every literature dataset, SWE-bench excluded: every step holds one prompt per dataset, 2 coding, 3 math, SQL, RAG, 2 tool-calling, Terminal-Bench; `[batch]`/`mix/*` logs prove it). Datasets: HF `mbpp`, `gsm8k`, `math500`, `deepscaler`, `humaneval`, `gretelsql`, `hotpotqa`, `xlam`, `mcp`, pandas/scipy/sklearn tasks (`DATASET=libs`), the synthetic zoo, Terminal-Bench (`DATASET=terminalbench`, formerly `demo_cpu_terminalbench`; see row 2), and mixes (`DATASET=a,b,c`, `DATASET=mixed`). **Every row names a task and an image; each image is one Runtime; `RB × NS` sets the sessions per step.** `image_routing.py` stops the run if a needed image has no Runtime. The Code Interpreter survives only as the `compare_backends.py` probe. **`SCALE=1`** is the harness-scaling run (formerly `demo_cpu_scale_lanes`): 4 SGLang engines : 256 harness workers : 256 sessions in flight, `throughput = min(inference, sandbox, harness)` |
| 2 | [`demo_cpu_tasks/`](demo_cpu_tasks/) `DATASET=terminalbench` | **Tier 3.** Custom images at scale: one image + Runtime per Terminal-Bench task (~229; deployed by `VARIANT=terminalbench` steps 2-4), a fresh microVM per episode | **Custom images in large numbers**, with the container launch hidden (a dataset of `demo_cpu_tasks` since 2026-09-28, so it mixes with the other datasets). The look-ahead pre-open (`TB_PREOPEN_BATCHES=2`) boots the next 1–2 batches' sessions, and an always-on heartbeat keeps them alive. The `terminalbench/warmstart/` probe (`bash terminalbench/warmstart/run_warmstart.sh`) measures cold vs pre-open vs heartbeat on its own, in about 7 min with no GPU |
| 3 | [`demo_cpu_swebench/`](demo_cpu_swebench/) | **Tier 3.** One image + Runtime per SWE-bench Verified instance (20), on managed **arm64 microVMs** (Epoch AI arm64 rebuilds, ≤ 2 GB compressed cap), a fresh microVM per episode | **One image per dataset row, serverless.** The Terminal-Bench pool (pre-open, heartbeat, strict routing) reused for real GitHub issues; reward = hidden FAIL_TO_PASS + PASS_TO_PASS tests. Moved from m5 Instances to arm64 microVMs 2026-09-28 (m5 state in `swebench/m5_legacy/`); local arm64 oracle check passes (django-15127), AWS run pending |
| 3b | [`demo_cpu_swebench_eks/`](demo_cpu_swebench_eks/) | **Same workload, EKS substrate.** One fresh **gVisor** pod (RuntimeClass `gvisor`) per episode on shared m5 nodes; same rows/harness/grader as `demo_cpu_swebench`, amd64 images shared with the ECS twin | **AgentCore vs EKS+gVisor comparison.** No image cap / arm64 / capacity-provider limits, denser packing; weaker isolation (user-space kernel, shared node). Added 2026-09-28; **cluster not created yet** (render-only; `EKS_CREATE=1` to build) |
| 3c | [`demo_cpu_swebench_ecs/`](demo_cpu_swebench_ecs/) | **Same workload, ECS substrate.** One fresh **ECS task on EC2** (`awsvpc`: own ENI + SG, caps dropped, no task role, IMDS blocked) per episode on an m5 ASG behind a capacity provider; amd64 images shared with the EKS twin | **The do-it-yourself baseline.** No image cap / arm64, no control plane or kubectl; weakest isolation (runc, node kernel shared). Added 2026-09-28; **never deployed** (render-only; `ECS_CREATE=1` to build) |
| 4 | [`demo_gpu_farm/`](demo_gpu_farm/) | **Tier 3.** Ray GPU farm on one shared warm instance. `LOCAL_GPU=1` (default): a local sandbox Runtime container on this box's GPUs 6,7. `LOCAL_GPU=0`: a remote g6 capacity-provider Instance, one warm custom Runtime (formerly `demo_gpu_single`) | KernelBench CUDA RL. **One GPU instance serving many sessions.** `FARM=fractional` (default; local GPU 6 or the g6's L4 at 0.5 → 2 slots) or `FARM=whole` (local GPUs 6,7 → 2 slots; the g6.xlarge L4 at 1.0 → 1 slot); `CONC` defaults to the slot count. Only a custom Runtime has CUDA (`gpu_torch_smoke.py`). `SNAPSHOT_EVERY=1` adds app-level snapshot/resume. **Trusted-only:** the episodes share one kernel and GPU (no MIG, a cooperative memory cap) |
| exp. | [`demo_cpu_branching/`](demo_cpu_branching/) | **Tier 3.** CPU Runtime whose image bakes node 22 + the real Claude Code CLI | One episode → many samples ("fan-out"). RL smoke 2026-09-25 ran end to end (fan-out p50 4, eval 0.47) but every advantage was 0; grouping bug fixed, re-run pending |

## Harness tier (default in every demo)

`throughput = min(inference, sandbox, harness)`. Async RL scales inference (SGLang engines) and the
sandbox (sessions); the **harness**, the agent loop that drives each episode, must scale **along with
them**, or a few coroutines silently become the throughput ceiling. And it should scale **with the
sandbox, not as part of the trainer**, as in DeepSeek's DSec (Elastic Compute): the mini_claude loop
runs inside the episode's own session, so N sessions carry N loops. What stays in the trainer is a
dispatch tier plus one `AnthropicAdapter` per engine. `agentcore_slime_demo/harness_pool.py` (in every
demo's lib, taken from the retired `demo_cpu_scale_lanes`) sizes the three tiers independently and
joins them **per episode**; no worker owns an engine or a session:

| Tier | Sized by | Default | Joined how |
|---|---|---|---|
| harness | `HARNESS_WORKERS` = W dispatch workers | RB×NS, one rollout step (32 at RB=4 NS=8) | a free worker **pulls** the next episode off one shared queue |
| sandbox | the demo's own sandbox | a fresh session per episode (`PREOPEN`/`HEARTBEAT` as before) | taken **before** the engine, so waiting for a session is not engine load. `SANDBOX_SESSIONS=M>0` (opt-in, `demo_cpu_tasks` own-Runtime only) keeps M persistent sessions per image, one episode at a time in a fresh `/workspace/ep-<hex>`; `SANDBOX_OPEN_RATE` (default 20/s) paces opens |
| inference | the SGLang engines | least-loaded engine (fewest in-flight episodes, ties rotate) | sticky for all the episode's turns |

This replaces the old per-engine hash pinning (`crc32` → engine) and the Terminal-Bench/SWE-bench
`HARNESS_WORKERS=4` workers pinned `w → w % E`. The pool returns **one bare Sample** per episode, the
shape slime's fully-async driver needs: it is the only rollout driver in every demo except the
experimental `demo_cpu_branching` (its episodes fan out into several Samples, so it keeps the stock driver). Which tier
binds shows in each episode's `[demo]` record: `qdepth` > 0 = harness-bound, `sandbox_wait_ms` > 0 =
sandbox-bound, `engine_inflight` climbing = inference-bound (plus `worker`, `engine`, and
`sandbox_slot`/`session_episode` when pooled). `metrics.py`'s stickiness check reports
sessions/engines per worker and workers per session; a worker using several engines or sessions is
expected, not a violation. `HARNESS_WORKERS` is exported to the sandbox env in every demo (empty → the
pool computes RB×NS). Background: [`docs/harness-scaling.md`](../../docs/harness-scaling.md).

**`SCALE=1` preset** (`demo_cpu_tasks` only; the retired `demo_cpu_scale_lanes` folder, whose logs
are now `demo_cpu_tasks/logs/scale/`): `SCALE=1 bash code/demos/demo_cpu_tasks/5_run.sh` sets
`DATASET=deepscaler RB=32` (NS=8 → 256 per step) `GPUS_PER_ENGINE=1` (E=4 TP1 engines)
`HARNESS_WORKERS=256`, and `CONC=ceil(256/E)=64` → 256 episodes in flight; ckpt suffix
`_scale`, log tag `-scale`. Any knob still overrides. The offline sweep
`demo_cpu_tasks/harness_scaling_demo.py` shows the `min(...)` knee in seconds, no AWS or GPU.

**Heartbeat.** `HEARTBEAT` is a `SessionKeeper` in every demo. It keeps idle sessions on **your own
Runtime** warm across the weight-sync barrier and other trainer stalls. It is **on by default** whenever
the sandbox is a remote Runtime (`SANDBOX=agentcore-*` with no local endpoint) and off otherwise; set
`HEARTBEAT=0` for the A/B baseline. It is a no-op on the Code Interpreter (probes only): those sessions end at their session
timeout whatever the traffic. Terminal-Bench rows (and `demo_cpu_swebench`) always run
their own keeper. Without pings, a reclaimed Runtime session silently returns a fresh, empty sandbox
under the same session id (measured by the warmstart probe).

**Idle timeout.** Runtimes created by `deploy/agentcore_deploy.py` get `idleRuntimeSessionTimeout =
RUNTIME_IDLE_TIMEOUT` (default **3600 s**, range 60–28800), sized above the worst trainer stall; it is
the backstop if the heartbeat does not fire. It applies only when a Runtime is created: an existing one
keeps its old timeout (600 s for Runtimes made before 2026-09-26; step 4 prints a note), so recreate
it with `FORCE_RECREATE=1`. `HEARTBEAT_IDLE` stays 600 s, which is safe for both. The cost is that an
abandoned session is billed for up to the timeout; normal episode ends call `StopRuntimeSession`.

**Isolation on a shared instance (demo 4).** Several episodes share one warm GPU instance, so
there is no VM boundary between them. The farm puts back as much isolation as one kernel allows:

- **Its own uid per sub_session** (`20000+slot`) with a private `0700` dir, where the container has
  the capabilities (the local farm). The AgentCore container drops every capability (no
  `setuid`/`chown`), so on the g6 they fall back to one shared uid. (The removed CPU pool,
  `demo_cpu_instance_pool`, used a Landlock jail per exec instead; not ported to the GPU farm.)
- **Limits:** process-count, RAM and file-size caps, and a timeout kills the whole process tree.
- **Server file ops** refuse `..`, absolute and symlink escapes.
- **No leaks:** `AWS_*` is not inherited, and the session id never appears on a command line.
- **Close** kills the episode's processes and deletes its dir.

The episodes still share the kernel, the network namespace, and CPU/RAM (there is no per-episode
cgroup). On the GPU farm they also share the GPU: CUDA contexts are separate, but the memory fraction
is cooperative, and with no MIG on L4/L40S a GPU fault hits every co-tenant. For a hard boundary use a
microVM per session (demos 1, 2; the `SANDBOX_SESSIONS=0` default) or a fresh Instance per episode (demo 3). `demo_gpu_farm/README.md` has the full table and its
probe results.

**Merge history.**
- 2026-09-23: the ten per-dataset CPU folders (`demo_cpu_mbpp` … `demo_cpu_mcp`) became `demo_cpu_tasks`.
- 2026-09-25: `demo_cpu_mixtask` → `demo_cpu_tasks DATASET=mixed`; `demo_cpu_heartbeat` → the
  `HEARTBEAT=1` knob; `demo_cpu_customruntime` → `demo_cpu_tasks/compare_backends.py`;
  `demo_gpu_snapshot` + `demo_gpu_customruntime` → `demo_gpu_single` (itself merged into `demo_gpu_farm` 2026-09-28).
- 2026-09-25, storyline restructure: `demo_cpu_tasks` defaults to the Code Interpreter;
  `demo_cpu_imagepool` (briefly `VARIANT=imagepool`) → `DATASET=libs` on the Code Interpreter, which
  already has pandas/scipy/sklearn; `demo_cpu_bigimage` (briefly `VARIANT=bigimage`) →
  `demo_cpu_instance_pool`; the GPU farms (`demo_gpu_shared_single/_multi`, `demo_gpu_local`, briefly
  `demo_gpu_farm FARM=`) → `demo_gpu_farm_fractional` + `demo_gpu_farm_whole`; the short-lived
  `demo_cpu_warmstart` → `demo_cpu_terminalbench/warmstart/` (now `demo_cpu_tasks/terminalbench/warmstart/`).
- 2026-09-26: `demo_gpu_farm_fractional` + `demo_gpu_farm_whole` → **`demo_gpu_farm FARM=fractional|whole`**
  (same image and code, only the GPU split differed); `demo_cpu_instance_pool` demoted to a side note.
- 2026-09-28: `demo_cpu_terminalbench BENCH=swebench` split out → **`demo_cpu_swebench`** (new main demo).
- 2026-09-28: `demo_cpu_terminalbench` → **`demo_cpu_tasks DATASET=terminalbench`** (folder retired). Every row
  now carries `metadata.task.image`, so Terminal-Bench rows (one image per task) and generic-image rows mix in
  one batch; the per-task pool is `VARIANT=terminalbench` steps 2-4 and its tooling is `demo_cpu_tasks/terminalbench/`.
- 2026-09-28: **`demo_cpu_swebench_eks`** added — the SWE-bench demo on EKS with one gVisor pod per episode (untested on a cluster).
- 2026-09-28: **`demo_cpu_swebench`** moved to arm64 microVMs (2 GB cap); **`demo_cpu_swebench_ecs`** added — one ECS-on-EC2 task per episode (untested on AWS). EKS/ECS default subset pinned to the arm64 demo's 20 rows.
- 2026-09-28: the in-trainer harness is gone (`HARNESS=direct`, `HARNESS_WHERE=trainer`); every demo
  runs the mini_claude loop in the sandbox session. `demo_cpu_tasks` dropped the Code Interpreter
  (`SANDBOX=agentcore`) and runs on its own `cpu` + `libs` Runtimes. The legacy two-runtime
  `agentcore_runtime_generate.py` / `harness_runtime_agent.py` were deleted from every lib.
- 2026-09-28: the decoupled harness tier (`harness_pool.py`) became every demo's default, and
  `demo_cpu_scale_lanes` → **`demo_cpu_tasks SCALE=1`** (folder removed; its offline sweep and logs
  moved to `demo_cpu_tasks/harness_scaling_demo.py` and `demo_cpu_tasks/logs/scale/`).
- 2026-09-28: `demo_gpu_single` → **`demo_gpu_farm LOCAL_GPU=0`** (remote g6; folder removed). Its
  snapshot/resume (`SNAPSHOT_EVERY=1`, `SNAPSHOT_DIR=demo_gpu_farm/snapshots`), probes, `train.sh` and
  `deploy/Dockerfile.sandbox.gpu` moved with it; its logs are in `demo_gpu_farm/logs/remote/` and
  `logs/snapshot/`.
- 2026-09-28: `demo_cpu_instance_pool` (side note: warm m5 pool, Landlock-jailed sub_sessions) **removed**,
  with its Runtime `acs_sandbox_bigcpu`, ECR repo and capacity provider `acs_cpu_cp_bigimage`. The code is in
  git history (`in_session_harness.py` keeps `MINI_SESSION_MODE=foreground` for a jailed farm).

Older logs and docs keep the old names.

**Sandbox mode by substrate.** The CPU demos are **remote-only**. Their `run.sh` rejects
`SANDBOX=local`, so a CPU run can never silently fall off AgentCore. By default the GPU farm runs on
this box's GPUs 6/7 (`LOCAL_GPU=1`: `code/deploy/run_local_gpu_runtime.sh fractional|whole`, or step 4),
because a remote g6 may be unavailable. `LOCAL_GPU=0`, passed on every step 2-6, moves it to a remote
g6 capacity-provider Instance (ARN required; slime then gets all 8 host GPUs).

## Which to run first

- **`demo_cpu_tasks`**: two generic images, built and deployed once (steps 2-4). Try
  `DATASET=gsm8k bash 5_run.sh`.
- **`demo_cpu_tasks/terminalbench/warmstart/run_warmstart.sh`**: the launch-cost story in minutes, no GPU.
- **`demo_gpu_farm`**: the GPU path (local farm by default, `LOCAL_GPU=0` for the remote g6).
- The offline probes `demo_cpu_tasks/heartbeat_demo.py` and `demo_gpu_farm/snapshot_demo.py` run in
  seconds with no AWS or GPU.

All trainers share one training path: slime `train_async.py`, Qwen3-4B, GRPO, the
checkpoint-deadlock fix and save-only-at-end. They differ only in the **task** and the **sandbox**.

**`run.sh` vs `train.sh`.** `run.sh` is the full runner (all env knobs; defaults to a short smoke).
Two demos also ship **`train.sh`**, a train-to-convergence wrapper: `demo_cpu_tasks` (all
datasets, incl. Terminal-Bench) and `demo_gpu_farm` (`demo_cpu_branching` has one
too, but it is blocked). `train.sh` exposes `STEPS` → `NR`, `EVAL_EVERY` → `EVAL_INTERVAL`,
`SANDBOX_CONC` → `CONC` and, on CPU, `SANDBOX_RUNTIMES` → `K`. For the 100–200-step runs, see
[`convergence/`](convergence/).
