# AgentCore as the Sandbox in Agentic RL

**Kaige Yang** — Senior Applied Scientist, 亚马逊云科技 (AWS)

September 25, 2026

---


## Purpose and thesis

A **hands-on guide for AWS customers doing agentic RL**, grounded in a working **slime + AgentCore** project, the **official AgentCore docs** [D1] and **6 runnable demos**.

- **What AgentCore can and cannot do** as an RL sandbox.
- **How to wire it into a trainer** — one `generate` hook into slime, no fork.
- **How to run each demo** — six numbered step scripts, a few knobs, measured smoke numbers.

**Thesis.** The loop has two seams: the **harness** (the agent loop) and the **sandbox** (where untrusted output runs and the reward is computed). We keep the harness **in-process** and put **AgentCore in the sandbox seam**. Policy is always **Qwen3-4B** trained by **slime**; across demos only the **sandbox substrate** and the **task** change.

# 1 · Agentic RL — pipeline, workflow, architecture

## The agentic-RL loop

![AgentCore fills the RL execution roles around a slime trainer.](figures/fig01_project-theme-two-roles.png){ width=78% }

A rollout is a multi-turn **act → observe → revise** loop: the policy proposes an action, the **sandbox** runs it, and a **checker** in the sandbox produces the reward. The **harness** owns the loop; **slime** trains on the resulting samples.

## The harness — owns the episode

![The harness as hub — generation, action, and the record that becomes training data.](figures/fig04_harness-as-hub-dataflow.png){ width=42% }

- Opens the episode, calls the policy turn by turn, routes actions to the sandbox, runs the checker, packs a training `Sample`.
- The **only component that sees a whole episode**, so it stamps reward, grouping and loss mask.
- A **proxy** records the **exact `token_ids` + logprobs** sampled (**TITO**); without it training is silently off-policy.

## slime — trainer + asynchronous rollout

![slime: rollout generation → data buffer → Megatron training, decoupled.](figures/fig15_slime-architecture.png){ width=66% }

**slime** [9] = **Megatron** (training) + **SGLang** (rollout) behind a router, bridged by a Data Buffer. Agentic workloads plug in as a **custom rollout-generation hook**; this is where the harness and the AgentCore sandbox live. Rollout and training run **async** (`MODE=B`, GRPO).

## Tasks — seed, run, check, reward

One **task-agnostic** harness; a task family changes only **prompt + seed files + checker**. All **RLVR**: binary 1.0/0.0 from a checker, no LLM judge.

```python
for name, content in task.setup_files():
    await sandbox.write_file(name, content)
for _ in range(task.max_turns):
    text = await policy.complete(messages)                 # model turn (via proxy)
    await sandbox.write_file(task.artifact_filename, task.extract_submission(text))
    exit_code, out, err = await sandbox.exec("python3 run_checks.py", timeout=60)
    if exit_code == 0: solved = True; break
    messages.append(task.observation(exit_code, out, err)) # feed failure back
reward = task.reward(0 if solved else 1)
```

**The checker runs inside the sandbox — the sandbox is the reward source.** Frontier labs (DeepSeek-R1 [2], Qwen3 [3], Kimi K2 [5]) train the same way against executable verifiers.

## Scaling — throughput = min(inference, sandbox, harness)

![Scaled rollout — C harnesses and sandboxes feeding E inference engines.](figures/fig08_scaled-rollout-C-E-C.png){ width=46% }

Scale two tiers and the bottleneck moves, so **diagnose first** with `perf/sglang_gen_ms_*` vs `perf/sandbox_ms_*`:

- **Generation dominates** → inference-bound: add SGLang replicas (e.g. long-reasoning math).
- **Sandbox dominates** → sandbox-bound: add sandbox concurrency or instances (e.g. KernelBench `nvcc`).

# 2 · AgentCore — foundation concepts and workflow

## AgentCore basics — can / cannot

![AgentCore's two RL roles — host the harness and/or be the sandbox.](figures/fig16_agentcore-two-roles.png){ width=30% }

- **Runtime** — serverless, **per-session isolated compute**, invoked with `InvokeAgentRuntime` (SigV4) [D2, D8].
- **Code Interpreter** — managed CPU sandbox, Python/JS/TS [D6]. Also Browser, Gateway, Memory, Identity, Observability [D13].
- **Cannot:** GPU on a microVM (GPU is on **Instances** only) [D5]; cross-session GPU time-slicing; microVM sessions past 8 h [D3].

## Two compute substrates

| | **microVM** (serverless, CPU) | **Instances** (EC2 via a capacity provider) |
|---|---|---|
| Compute | own guest kernel, **CPU only** | `m5` (CPU) · `g6` / `g5` … (**GPU**) |
| Arch / image cap | **arm64**, ~**2 GB** | **amd64**, instance root disk |
| Cold start | ~1–2 s | ~60–80 s (EC2 launch) |
| Session length | up to 8 h (15-min idle) | up to 14 days |
| Our reuse policy | fresh microVM per episode | **warm-shared** instance, fresh env per episode |
| Loop code | identical | identical — only the Runtime ARN differs |

## Runtime / session / compute — 1 : N : 1

![Everything keyed by `runtimeSessionId` — one isolated compute unit per session.](figures/fig34_session-id-routing.png){ width=60% }

- **Runtime** = a stateless template: one image, one ARN.
- **Session** = one `runtimeSessionId` (≥33 chars). The first invoke under a new id **cold-starts** its own microVM or instance; the same id routes back to it.
- One Runtime serves many sessions; each session maps to exactly one compute unit. **Runtimes track images, not tasks.**

## Lifecycle and latency

- **Launch:** no "create session" API. The first `InvokeAgentRuntime` under a new id provisions compute [D2].
- **Terminate:** idle timeout (15 min, resets per invoke), max lifetime (microVM 8 h, Instances 14 d), or `StopRuntimeSession` [D3, D4, D12]; memory is sanitized.
- **Latency:** no published cold-start number, so **measure it**. Every sandbox op logs `[sbx-op]` and every generation `[sglang-gen]` → `perf/sandbox_ms_*`, `perf/sandbox_open_*` (cold vs warm), `perf/sglang_gen_ms_*`.

## Build and deploy a Runtime

1. Build the sandbox image (`docker buildx`: **arm64** for microVM, **amd64** for Instances) and push to ECR.
2. `CreateAgentRuntime` with `containerConfiguration` [D9]. The image serves a `BedrockAgentCoreApp` (`sandbox_server_agent.py`) with ops `write_file` / `exec` / `read_file`.
3. For Instances: `CreateCapacityProvider` (`m5` or `g6.xlarge`, VPC subnets/SGs, idle timeout) and attach it [D5].
4. IAM: an execution role (ECR pull, logs) plus an operator role that can `InvokeAgentRuntime`.

Every demo wraps these as `2_build_image.sh` → `3_push_ecr.sh` → `4_deploy_runtime.sh` (§4).

## The GPU limit — one whole GPU per session

- 1 `runtimeSessionId` = 1 EC2 instance = **whole GPU(s)**; no documented cross-session sharing; **no MIG on L4** [D5].
- **Isolated scaling** = more instances (N× cost). **Shared GPU** = many episodes inside one session, which must be mutually trusted.
- Our pattern: a **Ray farm inside one session** (`num_gpus=0.5` actors, a per-process memory cap, an `nvcc` lock).

# 3 · Integrating AgentCore into slime

## Two hooks, no fork

![The three decoupling seams — harness, inference, sandbox.](figures/fig19_three-decoupling-seams.png){ width=64% }

- `--custom-generate-function-path` → `generate.generate` (runs the harness, records turns).
- `--custom-rm-path` → the reward.
- Each sandbox op is one `invoke_agent_runtime(ARN, session_id)`, offloaded with `asyncio.to_thread`, so **N episodes run concurrently**.

## The call path

![The generate hook — harness → proxy → policy, producing a loss-masked trajectory.](figures/fig21_generate-hook.png){ width=50% }

```python
async def generate(args, sample, sampling_params):        # slime hook
    task = task_from_dict(sample.metadata["task"])
    async with make_sandbox(os.environ["SANDBOX"]) as sb:  # session opened / closed here
        traj = await run_episode(task, policy, sb)
    return build_token_stream(traj, sample)               # loss-masked Sample
```

`__aexit__` ends the session the moment the episode returns — one teardown path for every demo.

## Sandbox backends and ARN wiring

**ARN wiring** (step 4 → `deploy/runtime.env`): `AGENTCORE_SANDBOX_RUNTIME_ARN` · `…_ARNS` (round-robin) · `…_ARN_BY_IMAGE` (JSON).

```{=latex}
\small
```

| `SANDBOX=` | Substrate | Used by |
|---|---|---|
| `agentcore` | managed Code Interpreter microVM | offline comparisons |
| `agentcore-runtime` | custom Runtime, one session per episode | CPU demos |
| `agentcore-ray-farm` | one shared `runtimeSessionId` + a per-episode `sub_session` | `demo_gpu_single`, `VARIANT=bigimage` |
| local farm (`:8080`) | same contract on local GPUs 6,7 | `demo_gpu_farm`, `LOCAL_GPU=1` |

## Warm reuse on Instances

The Instances substrate cold-boots in ~60–80 s, so a fresh instance per episode lets boot time dominate. Instead:

- **All episodes share one `runtimeSessionId`** → one warm instance, cold boot paid **once per run**.
- **Each episode gets a `sub_session`** with a fresh isolated working dir (CPU isolates caches too; GPU keeps the CUDA kernel cache warm).
- microVMs boot in ~1–2 s, so they are **not** reused: one fresh microVM per episode.

**Measured (m5, `VARIANT=bigimage`):** cold boots **201 → 1**; 1626 warm ops at ~120–350 ms vs ~65–73 s cold.

## Networking — pull, don't expose

![Network posture per component.](figures/fig32_network-posture-per-component.png){ width=62% }

Every Runtime is reached by a SigV4 `invoke_agent_runtime`, with **no inbound port**. Only dial-back (sandbox → harness, `demo_cpu_branching`) needs ingress: one rule, tcp/`ADAPTER_PORT`, source = the runtime SG, under `networkMode=VPC` [D10]. That requires interface endpoints (`ecr.api`, `ecr.dkr`, `s3`, `logs`, `sts`, `bedrock-agentcore`) [D11]; a missing one fails create with a **null reason**.

# 4 · Demos

## Demo catalogue — 6 demos

**3 CPU + 2 GPU + 1 experimental.** Run any demo with `cd code/demos/<demo>` then `1_prepare_data.sh` … `6_cleanup.sh` on the host. Numbers are 10-step smokes, not convergence.

```{=latex}
\scriptsize
```

| Demo | Substrate | Use case |
|----------|-------|-----------------|
| `demo_cpu_tasks` | microVM (`bigimage`: m5) | **Reference:** RLVR on ~19 datasets or a mix |
| `demo_cpu_terminalbench` | microVM, ~229 images | one Runtime per task, fresh VM, pre-opened |
| `demo_cpu_scale_lanes` | 4 microVMs | 4 harness : 4 engines : 4 VMs, 1:1:1 |
| `demo_gpu_single` | `g6` Instance | KernelBench on one warm g6 (+ snapshot) |
| `demo_gpu_farm` | local Ray farm | GPU sharing: `FARM=fractional` or `whole` |
| *`demo_cpu_branching`* | microVM + CLI | **experimental, blocked:** fan-out |

`demo_cpu_tasks` also covers `VARIANT=imagepool` (3 images), `compare_backends.py` (custom vs managed Runtime); `HEARTBEAT=1` works in every demo (all merged 2026-09-25).

## `demo_cpu_tasks` — the reference

**Use case.** RLVR on any dataset through one selector: real HF `mbpp` (default) `gsm8k` `math500` `deepscaler` `humaneval` `gretelsql` `hotpotqa` `xlam` `mcp`, the synthetic family zoo (`math` `sql` `ctf` `rag` …), a comma list, or `mixed`.

**Knobs.** `DATASET=` · `VARIANT=bigimage|imagepool` · `HEARTBEAT=1` · `NR` / `RB` / `NS` · `K` (Runtimes round-robin) · `CONC`.

```bash
bash 1_prepare_data.sh && bash 2_build_image.sh && bash 3_push_ecr.sh && bash 4_deploy_runtime.sh
DATASET=gsm8k bash 5_run.sh        # or DATASET=gsm8k,gretelsql,hotpotqa / DATASET=mixed
bash 6_cleanup.sh
```

**Metrics (held-out eval).** `mbpp` **trained** (convergence curve + ablation), smoke 0.69 · `math` 1.00 · `gsm8k` 0.90 · `math500` 0.89 · `deepscaler` 0.78 · `humaneval` 0.85 · `gretelsql` 0.60 · `hotpotqa` 0.65 · `xlam` 1.00 · `mcp` 1.00 · `mixed` 0.67.

## `demo_cpu_tasks` `VARIANT=bigimage` — side note

**Use case.** A ~4 GB sandbox image, over the 2 GB microVM cap, on a warm **m5** CPU capacity-provider Instance (amd64). Same trainer, warm-shared session, fresh env per episode.

```bash
VARIANT=bigimage bash 2_build_image.sh && VARIANT=bigimage bash 3_push_ecr.sh
VARIANT=bigimage bash 4_deploy_runtime.sh   # -> deploy/runtime.bigimage.env
VARIANT=bigimage bash 5_run.sh
VARIANT=bigimage DELETE_CP=1 bash 6_cleanup.sh
```

**Metrics.** Trained on m5, `eval` **0.76**. Cold boots **201 → 1** (8 first-wave opens share one boot within 7.9 s; 215 `open`/`close` pairs, each wiping its own dir).

## `demo_cpu_tasks` `VARIANT=imagepool` — a pool of images

**Use case.** Mixed-task RL where families need different software: a pool of 3 custom images (pandas / scipy / sklearn), one Runtime each, and each episode routed to its family's image. (Was `demo_cpu_imagepool`.)

**Knobs.** `AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE` (JSON image → ARN, written by step 4) · `IMAGES=` (which images to build) · `DATASET=imagepool` (set by the variant).

```bash
cd code/demos/demo_cpu_tasks
export VARIANT=imagepool
bash 1_prepare_data.sh && bash 2_build_image.sh    # 3 arm64 images
bash 3_push_ecr.sh && bash 4_deploy_runtime.sh     # -> ARN_BY_IMAGE map
bash 5_run.sh && bash 6_cleanup.sh
```

**Metrics.** Smoke (as `demo_cpu_imagepool`) held-out eval **1.0** (now `eval/imagepool`); all 3 images routed from one batch.

## `demo_cpu_terminalbench`

**Use case.** Terminal/SWE agents on **Terminal-Bench**: one image + Runtime per task (~229), a **brand-new microVM per episode** (tasks change the whole OS), and a look-ahead data source that **pre-opens** the next 2 batches' sessions to hide the cold start.

**Knobs.** `TB_PREOPEN_BATCHES` (default 2 → 64 sessions booted ahead) · `HARNESS_WORKERS` · `TB_EXEC_TIMEOUT`.

```bash
bash 1_prepare_data.sh            # fetch Terminal-Bench tasks
bash 2_build_image.sh             # one image per task (also pushes)
bash 4_deploy_runtime.sh          # one Runtime per task -> tasks/arn_by_image.json
bash 5_run.sh && bash 6_cleanup.sh
```

**Metrics.** 218/229 Runtimes deployed, 166 usable after the oracle check. First smoke (2026-09-23) passed to ckpt 10: boot fully hidden (`open_wait_ms` p95 0), but ~95 % of pre-opened microVM time idle and reward sparse (4/382 solves, eval 0.0). Key signals: `[demo] open_wait_ms`, `preopen_idle_ms`.

## `demo_cpu_tasks` — custom Runtime and `HEARTBEAT=1`

**Custom vs managed Runtime.** `python3 compare_backends.py` runs the same episode against the managed Code Interpreter and **your own** CPU Runtime: same reward, only operations differ. Real RL on your Runtime is just `AGENTCORE_SANDBOX_RUNTIME_ARNS=<yours> DATASET=deepscaler bash 5_run.sh`. Smoke (as `demo_cpu_customruntime`) `eval/deepscaler` **0.69**.

**Heartbeat, `HEARTBEAT=1` (any demo).** During the weight-sync barrier no ops reach the sandbox, and the 15-min idle timeout can kill it. A `SessionKeeper` heartbeat, outside the harness, keeps the session warm. Knobs: `HEARTBEAT_IDLE` (600 s) · `HEARTBEAT_PING_FRACTION` (0.5).

```bash
cd code/demos/demo_cpu_tasks
python3 compare_backends.py    # managed vs custom, same reward
python3 heartbeat_demo.py      # offline, seconds
DATASET=deepscaler HEARTBEAT=1 bash 5_run.sh
```

Smoke (as `demo_cpu_heartbeat`) `eval/deepscaler` **0.71** with the keeper live; the offline probe shows the recovery arc.

## `demo_cpu_scale_lanes`

**Use case.** Makes `throughput = min(inference, sandbox, harness)` concrete: **4 × 4 × 4 lanes** — 4 harness workers : 4 TP1 SGLang engines : 4 persistent microVMs, pinned 1:1:1. DeepScaleR, fixed.

**Knobs.** `LANES` (4) · `HARNESS_WORKERS` · `GPUS_PER_ENGINE` (1) · `LANE_SANDBOX` (1).

```bash
python3 harness_scaling_demo.py   # offline throughput sweep
bash 1_prepare_data.sh && bash 2_build_image.sh && bash 3_push_ecr.sh
bash 4_deploy_runtime.sh && LANES=4 bash 5_run.sh && bash 6_cleanup.sh
```

**Metrics.** Offline sweep + fake-microVM lane test pass; **4 × 4 × 4 not yet run on AWS**. The earlier smoke (0.33) used the old single-engine topology.

## `demo_gpu_single` — KernelBench on one warm g6

**Use case.** The policy writes a CUDA `ModelNew`; the checker runs `allclose` + speedup on `cuda` in your custom GPU Runtime on a **g6** Instance. All episodes share **one warm g6** (`agentcore-ray-farm`), each on a fresh `sub_session`. Train `level_1` (100), eval 20 `level_2` fusions.

**Knobs.** `SANDBOX` · `B4_FARM_MAX_CONCURRENT` · `LOCAL_GPU=1` (on-box farm on GPUs 6,7, no g6) · `SNAPSHOT_EVERY=1`.

```bash
bash 1_prepare_data.sh && bash 2_build_image.sh && bash 3_push_ecr.sh
bash 4_deploy_runtime.sh          # Runtime on the g6 capacity provider   (LOCAL_GPU=1: local farm)
bash 5_run.sh                     # SNAPSHOT_EVERY=1 for snapshot/resume
DELETE_CP=1 bash 6_cleanup.sh     # the g6 is the real cost
```

**Metrics.** **Trained** (committed convergence curve); smoke `eval/kernelbench` **0.525**.

## `demo_gpu_single` — custom-Runtime and snapshot angles

**Custom Runtime = the only GPU.** `python3 gpu_torch_smoke.py`: the managed Code Interpreter reports `cuda_available=False`; only your Runtime on a g6 has CUDA, through the same `write_file`/`exec`/`read_file` contract. Smoke (former `demo_gpu_customruntime`) **0.65**.

**Snapshot/resume, `SNAPSHOT_EVERY=1`.** AgentCore has no VM/GPU snapshot, so the harness does it at app level:

- `maybe_restore` at episode ENTER writes the saved workspace back.
- `maybe_snapshot` at EXIT captures it; EXIT is the quiescent point just before weight sync.
- Store: `SNAPSHOT_DIR` (local) or `SNAPSHOT_S3_BUCKET`.

Smoke (former `demo_gpu_snapshot`, local farm) **0.625**, 3 files saved + restored per episode. The live g6 + store round trip is **unverified**. Offline animation: `python3 snapshot_demo.py`.

## `demo_gpu_farm` — GPU sharing on this box

**Use case.** GPU sharing with no AWS GPU: a local Ray farm on GPUs 6,7 serves the same `/invocations` contract; slime uses the other GPUs.

- `FARM=fractional` (default): two episodes on **one** GPU, 0.5 + 0.5.
- `FARM=whole`: one whole GPU per episode across GPUs 6,7.

**Knobs.** `FARM` · `B4_GPU_FRACTION` · `B4_FARM_GPUS` · `B4_FARM_MAX_CONCURRENT`.

```bash
bash 1_prepare_data.sh && bash 2_build_image.sh
FARM=fractional bash 4_deploy_runtime.sh     # starts the local farm (no ECR push needed)
FARM=fractional bash 5_run.sh
bash 6_cleanup.sh
```

**Metrics.** `fractional` smoke **0.775**; `whole` smokes **0.65 / 0.70**. Balanced `open`==`close` (no leaked leases); `demo/boot_attempts` p50 = 1.

## `demo_cpu_branching` — EXPERIMENTAL, blocked

**Use case.** One episode → **many training samples** (fan-out), with the real **Claude Code CLI** as the in-sandbox agent. Each model call dials back to the harness's proxy (`ADAPTER_PORT` 18021, VPC mode), and every fragment carries a shared `rollout_id` for grouping. In Agent Lightning only 36% of rollouts stay one sample [1].

**Knobs.** `BRANCHING_FORK_THRESHOLD` (512) · `AGENT_TIME_BUDGET_SEC` (900) · `ADAPTER_PORT`.

```bash
bash 1_prepare_data.sh && bash 2_build_image.sh && bash 3_push_ecr.sh
bash 4_deploy_runtime.sh          # VPC Runtime + SG->SG dial-back rule
bash 5_run.sh && bash 6_cleanup.sh
```

**Status.** **Blocked:** the in-sandbox CLI exits 1 before any model call, so there are no trajectories and no reward signal yet. The empty-group crash is fixed; the CLI failure is still open.

## References

```{=latex}
\tiny
```

**Papers & code.** [1] Agent Lightning <https://arxiv.org/abs/2508.03680> · [2] DeepSeek-R1 <https://arxiv.org/abs/2501.12948> · [3] Qwen3 <https://arxiv.org/abs/2505.09388> · [5] Kimi K2 <https://arxiv.org/abs/2507.20534> · [7] KernelBench <https://huggingface.co/datasets/ScalingIntelligence/KernelBench> · [9] slime <https://github.com/THUDM/slime>

**AgentCore docs.** [D1] What is AgentCore <https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/what-is-bedrock-agentcore.html> · [D2] Runtime <https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agents-tools-runtime.html> · [D3] How it works <https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-how-it-works.html> · [D4] Lifecycle <https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-lifecycle-settings.html> · [D5] Instances (GPU) <https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-instances-how-it-works.html> · [D6] Code Interpreter <https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/code-interpreter-tool.html> · [D8] InvokeAgentRuntime <https://docs.aws.amazon.com/bedrock-agentcore/latest/APIReference/API_InvokeAgentRuntime.html> · [D9] CreateAgentRuntime <https://docs.aws.amazon.com/bedrock-agentcore-control/latest/APIReference/API_CreateAgentRuntime.html> · [D10] NetworkConfiguration <https://docs.aws.amazon.com/bedrock-agentcore-control/latest/APIReference/API_NetworkConfiguration.html> · [D11] VPC endpoints <https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/vpc-interface-endpoints.html> · [D12] StopRuntimeSession <https://docs.aws.amazon.com/bedrock-agentcore/latest/APIReference/API_StopRuntimeSession.html> · [D13] Observability <https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/observability.html>

**Repo.** `code/demos/README.md` (catalogue + step scripts) · `docs/report/agentcore_slime_report.md` (full results) · `docs/agentcore-networking.md` · `docs/gpu-snapshot.md`

---

*Thank you — Questions?*
