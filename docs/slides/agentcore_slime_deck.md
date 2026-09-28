# AgentCore as the Sandbox in Agentic RL

**Kaige Yang** — Senior Applied Scientist, 亚马逊云科技 (AWS)

September 25, 2026

---


## Purpose and thesis

A **hands-on guide for AWS customers doing agentic RL**, grounded in a working **slime + AgentCore** project, the **official AgentCore docs** [D1] and **5 runnable demos + 1 experimental + 1 side note**.

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

# 2 · Sandbox tasks from the literature

## Literature tasks that need no custom image

Every row is self-describing: `metadata.task = {family, task_id, spec, …}` rebuilds its seed files, hidden checker and reward, so one harness trains all of them.

```{=latex}
\scriptsize
```

| Task (`DATASET=`) | Dataset (Hugging Face) | Checker deps |
|-----------|--------------------|--------|
| code: `mbpp`, `humaneval` | google-research-datasets/mbpp · openai/openai_humaneval | stdlib |
| math: `gsm8k`, `math500` | openai/gsm8k · HuggingFaceH4/MATH-500 | stdlib |
| math: `deepscaler` | agentica-org/DeepScaleR-Preview-Dataset | stdlib |
| text-to-SQL: `gretelsql` | gretelai/synthetic_text_to_sql | `sqlite3` |
| RAG / multi-hop: `hotpotqa` | hotpotqa/hotpot_qa (distractor) | files in `./docs` |
| tool calling: `xlam`, `mcp` | minpeter/xlam-function-calling-60k-parsed · Team-ACE/ToolACE | stdlib |
| synthetic zoo (swe, ctf, regex, planning, config, data, toolcall) | built in (`task_families.py`) | stdlib |
| data science: `libs` (pandas / scipy / sklearn) | built in | preinstalled in the Code Interpreter |

## Literature tasks that do need a custom image

The default sandbox has **no git and no internet**, so nothing can be `pip install`-ed at run time. A custom image is needed only when the **environment** differs. A missing package is one more reason: bake it into your own Runtime image (`SANDBOX=agentcore-runtime`). Everything else runs on the default with **no Dockerfile, no ECR push, no Runtime to deploy**.

```{=latex}
\small
```

| Task | Source | Why | Where |
|-----------|------------|------------|--------------|
| Terminal-Bench (terminal / SWE) | laude-institute/terminal-bench (~229) | each task pins its own OS + tools | `demo_cpu_terminalbench` |
| big stack (torch + scientific) | any task | ~4 GB image > ~2 GB microVM cap | slim it to fit a microVM; else m5 (`…instance_pool`, side note) |
| CUDA kernels: KernelBench [7] | ScalingIntelligence / KernelBench | CUDA only on GPU Instances | `demo_gpu_*` (g6 / local) |
| agent fan-out (experimental) | real Claude Code CLI | node + CLI in the sandbox | `demo_cpu_branching` |

# 3 · AgentCore — foundation concepts and workflow

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

## AgentCore limits (1/4) — lifetime and rates

```{=latex}
\footnotesize
```

| Limit | Value [D14] | Adj. | Impact · what we do |
|--------------|------------|---|----------------------|
| **Idle timeout** | 15 min default; Runtime 60 s–8 h (we set 3600 s) | cfg | a reclaimed id silently returns an **empty** sandbox → heartbeat on by default |
| Max lifetime | microVM **8 h** · Instance **14 d** · CI ≤ 8 h | no | long episodes need snapshot / re-seed; CI pings don't extend it |
| **New Runtime sessions** | **25 /s** | yes | caps the cold-start burst per batch → pre-open ahead |
| Concurrent Runtime sessions | **5000** | yes | one per in-flight microVM episode |
| Runtime invoke (data plane) | 1000 /s | yes | why high fan-out moves to tier 2 |
| Code Interpreter | 1000 concurrent · **30 /s** start · **30 /s** invoke | yes | ≈ 75 episodes in flight |

## AgentCore limits (2/4) — resources and compute

```{=latex}
\footnotesize
```

| Limit | Value [D14] | Adj. | Impact · what we do |
|--------------|------------|---|----------------------|
| **Runtimes per account** | **1000** (10 endpoints, 1000 versions each) | yes | one per image; Terminal-Bench ≈ 229, account has **234** |
| Control-plane writes | 50 /s | no | creating hundreds of Runtimes: throttle + retry |
| Per request | **15 min** · 100 MB payload | no | a longer `exec` runs async and is polled |
| Image | microVM **2 GB**, arm64 only · CI disk 10 GB | no | bigger image → CPU Instance (`m5`) |
| GPU | Instances only · 1 session = 1 instance · no MIG | no | Ray farm to share a GPU |
| Privileges | container has **no Linux capabilities** | no | no per-episode uid / cgroup → Landlock jail |

## AgentCore limits (3/4) — launch time

```{=latex}
\small
```

**Launch time (measured by us, not published by AWS):**

| Substrate | Cold launch | Warm / reused open |
|---|---|---|
| Code Interpreter (managed microVM) | `start_session` 0.87–2.0 s per op; RL smoke (`libs`) p50 **1.83 s**, p95 3.50 s, **paid every episode** | – (fresh per episode) |
| Own Runtime, microVM | small stdlib image **≈ 0.77 s** median (warm-start probe); **≈ 15 s** for a 447 MB image, grows with size | pre-open wait p50 **5 ms** |
| CPU Instance (`m5.xlarge`, ~4 GB image) | **≈ 60–80 s** (83 s measured) | 0.12–0.33 s |
| GPU Instance (`g6`) | **≈ 123–207 s** | 0.15–0.19 s |

## AgentCore limits (4/4) — how many images

- ECR itself is not the limit. Defaults [D15] are **100,000 repositories per region** and **20,000 images per repository**, both adjustable. This account has **247** repositories.
- A Runtime points at **one image URI**, so the binding limit is **1000 Runtimes per account**. That caps distinct custom images at about 1000, unless tasks share an image.
- Push rate: `PutImage` 10 /s, which matters when building hundreds of task images. Pulls (`GetDownloadUrlForLayer` 3000 /s) are not a concern.
- Image size drives cold launch: the microVM cold boot grows with the image, and the Terminal-Bench images run p50 128 MB to max 2 GB.

## Related work — DeepSeek DSec vs AgentCore [31]

```{=latex}
\scriptsize
```

DSec: DeepSeek's in-house RL sandbox platform (V3.2 → V4.1), **3M sandboxes/day**.

| | DSec | AgentCore (this account) |
|-----------|------------------------------|--------------------------------------|
| Creation rate | > 5000 /s | **25 /s** new Runtime sessions (adj.) |
| Running at once | 380K | 5000 Runtime · 1000 CI sessions |
| Environments | 11K bases × 100K workspaces, layered | one image per Runtime, max **1000** |
| Images | lazy: fetch only the 4–13 % a sandbox touches | full image per boot (≈ 0.8 s stdlib, ≈ 15 s / 447 MB), 2 GB cap; **ours:** no image on CI, deps at boot, > 2 GB on an Instance (boot paid once) |
| Hide boot | lazy loading → low pull time; no pre-warming (volume, not latency, is the bottleneck) | **pre-open (ours)**: next 1–2 batches boot during this one (774 → **5 ms**) |
| Lifetime | p50 15–17 min, p99 > 3 h | 15-min idle; max 8 h microVM / 14 d Instance |
| Trainer stall | pause / resume, state kept | **no pause/resume, no state kept** |
| Agent loop | outside the GPU job | **outside the trainer too (ours)**: harness + tools in the Runtime session (branching today; others being ported) |
| Reward hacking | AppArmor + per-task network allowlists | microVM boundary; Landlock on shared pools; VPC mode |

## Related work — what we take from DSec

- **Their median sandbox lives about as long as our idle timeout.** Heartbeat is now **on by default** for our own remote Runtimes, and new Runtimes get a 3600 s idle timeout.
- **New in `metrics.py`, in every demo:**
  - `perf/sandbox_ready_span_s`: first open start → last open ready, per step (DSec's batch time-to-ready);
  - `perf/sandbox_open_peak_per_s`: peak opens in any 1 s window, compared with the 25/s quota;
  - `<log>.sessions.csv` + summary: lifetime and longest idle gap per session vs the idle limit, peak live sessions vs 5000.
- **Terminal-Bench smoke, re-parsed:** 414 sessions, lifetime p50 **471 s** / p95 **1841 s**; **55** outlived 900 s (pre-open idle, kept alive by heartbeat); peak **36 opens/s**, above 25/s, with **0 throttles** (observed, not an AWS statement).
- **Not yet:** CPU/RAM used vs requested (needs a sandbox-server change + redeploy); co-located-load latency on shared Instances; reward-hack incident counts.
- **What AgentCore doesn't expose:** image layers, session pause, and a creation rate in the thousands. **What we can copy cheaply:** move the agent loop out of the trainer; allowlist the network per task.

## Build and deploy a Runtime

1. Build the sandbox image (`docker buildx`: **arm64** for microVM, **amd64** for Instances) and push to ECR.
2. `CreateAgentRuntime` with `containerConfiguration` [D9]. The image serves a `BedrockAgentCoreApp` (`sandbox_server_agent.py`) with ops `write_file` / `exec` / `read_file`.
3. For Instances: `CreateCapacityProvider` (`m5` or `g6.xlarge`, VPC subnets/SGs, idle timeout) and attach it [D5].
4. IAM: an execution role (ECR pull, logs) plus an operator role that can `InvokeAgentRuntime`.

Every demo wraps these as `2_build_image.sh` → `3_push_ecr.sh` → `4_deploy_runtime.sh` (§5); `demo_cpu_tasks` needs none of them.

## The GPU limit — one whole GPU per session

- 1 `runtimeSessionId` = 1 EC2 instance = **whole GPU(s)**; no documented cross-session sharing; **no MIG on L4** [D5].
- **Isolated scaling** = more instances (N× cost). **Shared GPU** = many episodes inside one session, which must be mutually trusted.
- Our pattern: a **Ray farm inside one session** (`num_gpus=0.5` actors, a per-process memory cap, an `nvcc` lock).

# 4 · Integrating AgentCore into slime

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
| `agentcore` | managed Code Interpreter microVM, no image | `demo_cpu_tasks` (default today → opt-in) |
| `agentcore-runtime` | your own Runtime, one fresh session per episode | `demo_cpu_terminalbench`, `demo_cpu_scale_lanes`, tasks (default, *in progress*) |
| `agentcore-ray-farm` | shared `runtimeSessionId`(s) + a per-episode `sub_session` | `demo_gpu_single`, `demo_cpu_instance_pool` (side note) |
| local farm (`:8080`) | same contract on local GPUs 6,7 | `demo_gpu_farm` (`FARM=fractional` \| `whole`) |

## Networking — pull, don't expose

![Network posture per component.](figures/fig32_network-posture-per-component.png){ width=62% }

Every Runtime is reached by a SigV4 `invoke_agent_runtime`, with **no inbound port**. Only dial-back (sandbox → harness, `demo_cpu_branching`) needs ingress: one rule, tcp/`ADAPTER_PORT`, source = the runtime SG, under `networkMode=VPC` [D10]. That requires interface endpoints (`ecr.api`, `ecr.dkr`, `s3`, `logs`, `sts`, `bedrock-agentcore`) [D11]; a missing one fails create with a **null reason**.

# 5 · The demos — one storyline

## Storyline — 5 demos + 1 experimental + 1 side note

Each demo runs as `cd code/demos/<demo>` then `1_prepare_data.sh` … `6_cleanup.sh`. Numbers are 10-step smokes (`NR=11`), not convergence.

```{=latex}
\scriptsize
```

| Question | Demo | Sandbox tier |
|----------------------|----------------|----------|
| Which tasks does the literature use? | task table (§2) | — |
| Do most train on the **default** runtime? | `demo_cpu_tasks` | 1 · CI today → 2 (*in progress*) |
| **Custom images** in large numbers? | `demo_cpu_terminalbench` | 3 · image per task |
| Cut **launch cost** (pre-open, heartbeat)? | `…terminalbench/warmstart/` | 1 + 2 |
| **Scale** the tiers independently? | `demo_cpu_scale_lanes` | 2 · generic image |
| GPU: one warm GPU Instance | `demo_gpu_single` | 3 · CUDA, g6 |
| GPU: **share one card** or one whole GPU each | `demo_gpu_farm` (`FARM=`) | 3 · CUDA, 0.5 or 1 GPU |
| *Experimental:* episode → many samples | *`demo_cpu_branching`* | 3 · node + CLI |
| *Side note:* > 2 GB image on a warm shared pool | *`demo_cpu_instance_pool`* | 3 · 4 GB image, m5 |

## Which sandbox does a task need? Three tiers

**Pick the lowest tier that works.** Tier 2 only for run-level needs; tier 3 only when the environment itself must change. A tier-1 task scores the same on tier 2 (`compare_backends.py`).

```{=latex}
\scriptsize
```

| Tier | Use it when | Here | Demos |
|:----------|:-------------------|:-------------------|:-----------|
| **1 · Code Interpreter**, no image | Python 3.12 + sci stack, sqlite, gcc, node is enough; no git / internet / GPU | **every literature task** (code, math, SQL, RAG, tool calls, pandas/scipy/sklearn, zoo, mixes) | `demo_cpu_tasks` today |
| **2 · own Runtime, generic image** | task fits tier 1, the **run** doesn't: > ~75 episodes in flight (CI invokes 30/s), heartbeat past idle, VPC | DeepScaleR at 256 sessions; heartbeat | `…scale_lanes`, `warmstart/`, `…tasks` (default, *in progress*) |
| **3 · own Runtime, custom image** | the **environment** differs | own OS per task; node + CLI; 4 GB stack (m5 Instance); CUDA (g6 Instance) | `…terminalbench`, `…branching`, `…instance_pool`, `demo_gpu_*` |

## Code Interpreter vs own Runtime — pros and cons

```{=latex}
\scriptsize
```

| | **Tier 1 · Code Interpreter** | **Tiers 2–3 · own Runtime** |
|:-----|:-------------------------------|:-------------------------------|
| Setup | **+** zero: no image, ECR, IAM role or VPC; nothing to maintain; fresh isolated microVM per session | **−** yours: image, Runtime, exec role, VPC endpoints + SG rule; silent `CREATE_FAILED` trap |
| Env | **+** AWS-maintained Py 3.12 + sci stack, torch-CPU, sqlite, gcc, node · **−** fixed: no git, internet, `pip`, custom OS, GPU | **+** any env: per-task images, > 2 GB (m5), CUDA (g6) · **−** you maintain images; new image = new Runtime |
| Tasks | 12 of 16 task rows | **+** superset: same reward on the 12 (`compare_backends.py`) + 4 more |
| Harness | **−** no outbound path to the trainer → harness **cannot** run inside | **+** VPC mode dials back → harness can live **in the session** |
| Scale | **−** `InvokeCodeInterpreter` 30 /s ≈ 75 episodes in flight | **+** much higher: 256 sessions exercised |
| Launch, idle | **−** new session every episode (p50 1.8 s); heartbeat no effect, session timeout is the only lever | **+** heartbeat bridges weight sync · **−** big images slower; Instances boot 60–80 s (m5) / 2–3 min (g6) → reuse; you pay idle held sessions (~95 % idle, TB smoke) |

## Task coverage and launch cost — CI vs own Runtime

**Launch, measured** (full table: limits 3/4):

- **Code Interpreter:** `start_session` 0.87–2.0 s per op; `libs` RL smoke p50 **1.83 s** / p95 3.50 s, **every episode**.
- **Own Runtime, microVM:** stdlib image **≈ 0.77 s** median cold; ≈ 15 s at 447 MB; pre-open → **≈ 5 ms** wait.
- **Instances:** m5 ≈ 60–80 s cold (83 s measured), g6 ≈ 123–207 s; warm reopen 120–330 / 150–190 ms → reuse.

```{=latex}
\footnotesize
```

| Task rows (16) | Code Interpreter | Own Runtime |
|:------------------------------|:-----|:-----------|
| **12:** 9 literature sets (`mbpp`, `humaneval`, `gsm8k`, `math500`, `deepscaler`, `gretelsql`, `hotpotqa`, `xlam`, `mcp`) + `libs`, `mixed`, zoo | ✓ | ✓ same reward |
| **4:** Terminal-Bench · > 2 GB image · KernelBench / CUDA · Claude-CLI SWE | – | ✓ **only here** |
| SWE-bench (being added) | – | ✓ (planned) |

## Decision (09-26, IN PROGRESS) — own Runtime as default

**Status: decided, not yet run.** No results below; every number in this deck is from the current setup.

- **New default, `demo_cpu_tasks` included:** tier 2, our own Runtime with the generic stdlib image (`acs-tasks-cpu`), `networkMode=VPC`. The Code Interpreter stays as the **zero-setup opt-in** (`SANDBOX=agentcore`).
- **Why: harness in the sandbox session.** One agent loop per sandbox, the harness running inside the **same AgentCore Runtime session as its tools** (DeepSeek DSec pattern [31]): loop and sandbox share one lifetime. The Code Interpreter cannot host it: it has **no outbound path to the trainer**.
- **Today:** only `demo_cpu_branching` runs this way (Claude Code CLI in the microVM, dial-back to the trainer). The port of the other demos is **under way**; §10's "loop stays in the trainer" describes the current code, not the target.
- **Open items:** the generic image is stdlib-only, so `DATASET=libs` needs pandas / scipy / sklearn added; every microVM needs the VPC + SG dial-back path; the tamper surface of §10 (policy commands next to the transcript) applies.

# 6 · Most tasks train on the default AgentCore runtime

## The default sandbox — what is inside (probed live)

`SANDBOX=agentcore` = the managed **AgentCore Code Interpreter** [D6]: one isolated microVM session per episode (`StartCodeInterpreterSession` → `InvokeCodeInterpreter` per op → stop). No ARN, no IAM execution role.

**Live probe, us-west-2, 2026-09-25:**

- Python 3.12.13 on aarch64.
- numpy 1.26.4, pandas 2.3.1, scipy 1.16.0, scikit-learn 1.5.0, sympy, torch 2.3.0 (CPU), statsmodels, networkx, matplotlib, requests, sqlite3.
- gcc 11.5, node v24. **No** git, **no** internet.

**All 13 task families** scored reward **1.0** on it with a MockPolicy episode (`SANDBOX=agentcore python3 smoke_remote_families.py`), including the data-science families that used to need 3 custom images.

**Op latency (4 concurrent opens):** start_session 0.87–2.0 s · write ~82 ms · exec ~251 ms · read ~75 ms · stop ~92 ms.

## Why most tasks need no image

- **The sandbox contract is task-agnostic:** `write_file` / `exec` / `read_file`. The task's *seed files and checker travel with the episode*; they are written into the session at episode start, not baked into an image.
- **The checker needs only what the environment already has.** Math, code, SQL, RAG and tool-call checkers are stdlib Python (plus `sqlite3`); the data-science checkers need pandas/scipy/sklearn, which are preinstalled.
- **Nothing persists between episodes**, so there is no environment to build: each session starts clean and is stopped when its episode ends.
- **An image only encodes the environment**, never the task. You need one only when the environment itself differs (see the previous slide).

## Image routing — how an episode finds its sandbox

```python
task = task_from_dict(sample.metadata["task"])        # family + spec from the data row
image = task.image                                    # ClassVar per family: "cpu", "datasci", "tb-<task>", ...
async with make_sandbox(os.environ["SANDBOX"], image=image) as sb: ...
# SANDBOX=agentcore-runtime only:
def _runtime_arn_for_image(image):
    arn = json.loads(os.environ.get("AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE", "{}")).get(image)
    return arn or _next_runtime_arn()                 # else round-robin over ..._ARNS
```

- `metadata.task` → `task.image` → `make_sandbox(image=)` → `_runtime_arn_for_image` → Runtime ARN.
- **AgentCore never sees the task.** It sees one ARN (an image) and one session id. **Runtimes track images, not tasks.**
- On the default Code Interpreter the tag is ignored: every task shares the one managed environment.
- Terminal-Bench is the same mechanism at scale (a ~229-entry map), but **strict**: an unmapped task is an error, never a fallback.

## What `acs_tasks_cpu` is — the opt-in own Runtime

`acs_tasks_cpu` is the Runtime that `demo_cpu_tasks` deploys **only if you opt in** with `SANDBOX=agentcore-runtime` (today; it becomes the default, *in progress*, §5):

- image `acs-tasks-cpu` = `python:3.12-slim` + the sandbox server `sandbox_server_agent.py` (stdlib only);
- one fresh microVM session per episode, reached with `invoke_agent_runtime`;
- `AGENTCORE_SANDBOX_RUNTIME_ARNS` fans episodes out round-robin over K Runtimes (the Runtime-count ablation).

```bash
bash 5_run.sh                      # today's default: Code Interpreter, 2/3/4/6 no-ops
SANDBOX=agentcore-runtime bash 2_build_image.sh   # then 3_push_ecr, 4_deploy
SANDBOX=agentcore-runtime bash 5_run.sh     # then 6_cleanup
```

**Why you would:** extra packages, your own server logic, or a longer idle timeout. It is stdlib-only, so `DATASET=libs` scores 0 there unless you add the libraries. `compare_backends.py` shows the same episode gives the same reward on both backends.

## `demo_cpu_tasks` — use case · why · design

```{=latex}
\small
```

**Use case covered**

- RLVR on literature datasets: code, math, text-to-SQL, RAG, tool calling, data science (`DATASET=`).

**Why this demo**

- Most tasks need **no image**: the Code Interpreter already has the checker deps. The lowest-ops entry point; every later demo adds one thing the default can't do.

**Design considerations**

- Code Interpreter default today: steps 2/3/4/6 are no-ops (no build, deploy or cleanup); own Runtime opt-in (`SANDBOX=agentcore-runtime`), becoming the default (*in progress*, §5).
- One harness for all tasks: self-describing rows + `task.image` routing. `DATASET=a,b,c` mixes; GRPO normalizes per group, so reward scales never mix.
- Heartbeat is a no-op here → the lever is the session timeout; the invoke-rate quota (30/s) binds first.
- Rejected: a 3-image pool for pandas / scipy / sklearn (the old imagepool); the Code Interpreter has them.

## `demo_cpu_tasks` — train on the default runtime

One selector: `DATASET=mbpp` (default) · `gsm8k` · `math500` · `deepscaler` · `humaneval` · `gretelsql` · `hotpotqa` · `xlam` · `mcp` · `libs` · zoo · `a,b,c` · `mixed`.

```bash
bash 1_prepare_data.sh
DATASET=gsm8k bash 5_run.sh                    # no build, no push, no deploy
DATASET=libs  bash 5_run.sh                    # pandas/scipy/sklearn, no image
```

**Metrics on the Code Interpreter (2026-09-25):**

- `gsm8k`: `raw_reward` 0.81–1.0 per step; **`eval/gsm8k` 0.90**; 425 sessions, **0 infra errors**.
- `libs`: `raw_reward` ≈ 1.0; **`eval/libs` 1.0**; 1883 sandbox ops, all ok; step ~45–53 s.
- Open p50 ~1.8 s / p95 ~3.5 s (largest cost); `write_file` 69 ms; `libs` exec 1.34 s. Both near-saturated: proves the path, not a curve.

**Earlier, own CPU Runtime** (09-18/19): `mbpp` 0.69 (trained) · `math` 1.00 · `math500` 0.89 · `deepscaler` 0.78 · `humaneval` 0.85 · `gretelsql` 0.60 · `hotpotqa` 0.65 · `xlam` 1.00 · `mcp` 1.00 · `mixed` 0.67.

## Quotas and concurrency on the default runtime

- The **invoke rate binds first**: at ~8 ops and ~20 s per episode, 30/s allows ~75 episodes in flight. Raise it before `CONC ≳ 32`.
- **Session timeout, not heartbeat:** a Code Interpreter session ends at the `sessionTimeoutSeconds` set at start, whatever traffic it gets. The demo sets `AGENTCORE_CI_SESSION_TIMEOUT=3600` (max 8 h); sessions stop at episode end, so it costs nothing. `HEARTBEAT=1` is a warned no-op here.

```{=latex}
\small
```

| Quota (us-west-2 default) | Code | Default | This demo |
|-------------|------|----|----------|
| Concurrent Code Interpreter sessions | L-CAF6F552 | 1000 | `CONC × engines` = 4 × 2 = **8** |
| StartCodeInterpreterSession rate | L-6B09F374 | 30/s | about one per episode |
| InvokeCodeInterpreter rate | L-04A719E9 | 30/s | ~6–10 per episode |

# 7 · Custom images at scale — Terminal-Bench

## `demo_cpu_terminalbench` — use case · why · design

```{=latex}
\small
```

**Use case covered**

- Terminal / SWE agents where each task pins its **own OS image** (Terminal-Bench, ~229 single-container tasks).

**Why this demo**

- Custom images **at scale**: one image + Runtime per task, strict routing. Not in `demo_cpu_tasks`: there tasks share one environment; here the environment *is* the task.

**Design considerations**

- **Fresh microVM per episode:** tasks mutate the whole OS, so the session is the isolation boundary; never reused.
- Oracle validation (null FAILS, reference PASSES) drops broken tasks: 218 deployed → 166 usable.
- Look-ahead pre-open hides the cold start at the price of idle microVMs (`TB_PREOPEN_BATCHES`); always-on heartbeat keeps them alive.
- Single-container tasks only; 64 ahead ≪ the 5000-session quota. `warmstart/` probes both mechanisms without a GPU.

## `demo_cpu_terminalbench` — one image + Runtime per task

Terminal/SWE agents on **Terminal-Bench**: each task pins its own OS, so each gets its **own image + Runtime** (~229) and every episode a **brand-new microVM**.

- **Strict routing:** `task.image` → `tasks/arn_by_image.json`; an unmapped task is an error.
- **Oracle check first:** the null solution must FAIL and the reference must PASS on the deployed image.
- **Stop on exit:** `StopRuntimeSession` frees the quota slot when the episode ends.

```bash
bash 1_prepare_data.sh && bash 2_build_image.sh  # one image per task
bash 4_deploy_runtime.sh && python3 validate_oracle.py && bash 1_prepare_data.sh
bash 5_run.sh && bash 6_cleanup.sh
```

**Metrics.** 218/229 Runtimes deployed, **166 usable** (135 train + 31 eval). Smoke (2026-09-23): 382 episodes, **0 infra errors**, 100 % warm hit; reward sparse (4/382 solves, eval 0.0).

# 8 · Cutting launch cost — pre-open and heartbeat

## Pre-open — boot the next batches while this one runs

slime hands the pool only the current batch, so the pool alone cannot look ahead. A **look-ahead data source** (`tb_lookahead_source.py`, via `--data-source-path`) replays slime's deterministic prompt order and tells the pool the episodes of the **next `TB_PREOPEN_BATCHES` rollouts**:

- `TB_PREOPEN_BATCHES=2` (default) → `2 × RB × NS` = **64** sessions kept booted ahead; `1` → 32; `0` → off.
- `TerminalBenchPool._refill` boots per-task sessions; a worker takes any ready session for its task.
- `[demo] open_wait_ms` = the cold start still on the critical path; `preopen_idle_ms` = the price.

**Smoke:** `open_wait_ms` p95 **0** (boot fully hidden), open p50 0.66 s, but `preopen_idle_ms` p50 7.3 min vs episode p50 21 s → **~95 % of pre-opened microVM time idle**. Next: `TB_PREOPEN_BATCHES=1`.

## Warm-start probe (`terminalbench/warmstart/`)

Host-only, ~7 min: 24 MockPolicy `sql` episodes in batches of 6, **cold** (open at episode start) vs **preopen** (next batch booted during this one); all rewards 1.0. `BACKENDS=runtime,codeinterpreter bash warmstart/run_warmstart.sh`.

**Reading:** pre-open takes the boot off the critical path; only the first batch waits. The price is idle held session time, the same trade-off as `preopen_idle_ms` at full scale.

```{=latex}
\small
```

| Backend | Mode | Wait p50 | Waited for boot | Idle held | Makespan |
|---------|------|----:|-----:|----:|------:|
| own Runtime | cold | 774 ms | 24/24 | 0 s | 31.9 s |
| own Runtime | preopen | **5 ms** | 8/24 | 75 s | 25.5 s (**−20 %**) |
| Code Interpreter | cold | 1715 ms | 24/24 | 0 s | 28.7 s |
| Code Interpreter | preopen | **0 ms** | 6/24 | 70 s | 24.3 s (**−15 %**) |

## Heartbeat — surviving the weight-sync stall

While slime syncs weights no ops reach the sandbox, and **your own Runtime** reclaims an idle session after its idle timeout. A `SessionKeeper` outside the harness pings any session idle past `fraction × timeout` with `exec("true")`. Probe: 3 sessions idle **150 s** against a **60 s** timeout.

**Silent reclaim is the trap:** the id still answers, nothing errors, the files are gone. Terminal-Bench's keeper is always on; it is **on by default** for every remote own Runtime (`HEARTBEAT=0` = baseline), with a 3600 s idle timeout as the backstop; on the Code Interpreter, set a long session timeout.

```{=latex}
\small
```

| Backend | Heartbeat | Survived | Resume |
|---------|---------|----:|--------------------|
| own Runtime | on (14 pings) | **3/3** | 104 ms, files intact |
| own Runtime | off | 0/3 | **silently reclaimed**: same id, empty sandbox |
| Code Interpreter | on / off | 0/3 | expires at `sessionTimeoutSeconds` |

# 9 · Side note — a warm CPU pool, and what a shared Instance isolates

## Side note: `demo_cpu_instance_pool` — a warm pool

**No longer a main demo.** Capacity-provider **Instances are AgentCore's second compute type** (§3); what is **ours** is the pooling on top: one shared `runtimeSessionId` per warm m5 + our per-episode `sub_session`.

- **Use case:** a ~4 GB image (torch + sci stack) > the ~2 GB microVM cap → CPU Instance (`m5.xlarge`), cold boot ~60–83 s. **The usual fix is not a pool:** slim the image or install heavy deps at boot, so it fits a microVM.
- **The trade:** `SANDBOX_POOL=N` (default 2) warm m5s, least-loaded placement, N boots per run, not per episode; the VM boundary is lost → **trusted checkers only**.
- Rejected: a fresh `runtimeSessionId` per episode (375 m5 boots in 10 steps, wait ratio > 0.97).

```bash
bash 1_prepare_data.sh && bash 2_build_image.sh && bash 3_push_ecr.sh
bash 4_deploy_runtime.sh && SANDBOX_POOL=2 bash 5_run.sh && DELETE_CP=1 bash 6_cleanup.sh
```

**Metrics.** RL smoke (2026-09-25, 10 steps): 406 episodes, **2 cold boots** per run, warm op p50 **117 ms**, step 38–219 s (was ~730 s), **`eval/deepscaler` 0.7875**.

## Shared-Instance isolation (1/2) — achieved vs shared

**Achieved** per exec (`sandbox_farm_jail.py`): **22/22** probe checks pass live on a warm m5; 13/13 families still 1.0; **~+50 ms / exec**.

- **Landlock jail:** write only the episode's own dir (fresh, deleted on close); siblings, `/tmp`, `/dev/shm` → `EACCES`; every TCP `connect` / `bind` denied (server port, IMDS, internet).
- **Processes:** cross-domain signals, `ptrace`, `/proc/<pid>/{environ,cwd,mem}` blocked; the whole tree is killed at exec end; watchdog kill > 256 procs / 4 GiB RSS.

```{=latex}
\scriptsize
```

| Still shared | Consequence |
|----------|------------------------------|
| Kernel | an exploit or Landlock bypass defeats all of the above; no VM boundary |
| uid + PID namespace | all uid 0 in one userns; `ps` shows other episodes' command lines |
| UDP / DNS · netns | not covered by Landlock; one network stack, no per-task network |
| CPU · RAM | no cgroup: one episode can saturate 4 vCPU; the RAM watchdog only *polls* (a spike can OOM a neighbour) |
| Blast radius | a server crash, OOM or full disk hits every episode on the Instance |

## Shared-Instance isolation (2/2) — pitfalls

- **All capabilities dropped → per-episode uids fail:** `setuid` / `chown` / `unshare` / cgroups `EPERM`; our uid image failed every open (HTTP 500). Local docker keeps the caps and hides it: test with `--cap-drop ALL`.
- **Landlock depends on the kernel ABI** (fs ≥ 1, TCP ≥ 4, signal scoping ≥ 6; m5 kernel 6.18 = ABI 7). Older kernels degrade feature by feature, silently: rerun `isolation_probe.py --live` after an image or instance change.
- **Knobs that silently weaken it:** `SANDBOX_FARM_SHM=allow` (shares `/dev/shm`, needed by `multiprocessing`), `SANDBOX_FARM_NET=allow`, `SANDBOX_FARM_LANDLOCK=0`.
- **No process survives across execs**, even within one episode; **OS-mutating tasks** (apt, systemd, `/etc`) can't run in the jail at all → fresh session.
- **The GPU farm is weaker still:** no Landlock port, a cooperative memory cap, one shared card that can wedge for everyone.
- **Idle cost:** warm pool Instances bill while idle until the idle timeout reclaims them.

**Bottom line:** this stops **reward hacking** (reading a sibling's answer, killing a neighbour's checker, a leftover daemon); it is **not a boundary against hostile code**. **Full isolation = one session per episode:** own microVM (~1 s) or own Instance (60–80 s m5 / 2–3 min g6).

# 10 · Scaling the tiers independently

## `demo_cpu_scale_lanes` — use case · why · design

```{=latex}
\small
```

**Use case covered**

- High-throughput rollout: 256 episodes in flight across 4 inference engines and 256 sandboxes.

**Why this demo**

- Shows `throughput = min(inference, sandbox, harness)` with the three tiers as **independent pools**.
- Harness workers and microVMs scale for different reasons (Python concurrency vs quota and cost), so they are two knobs, not one.

**Design considerations**

- Per episode: a free worker pulls it, borrows **any free microVM**, then takes the **least-loaded engine** (sticky for its turns). Nothing is pinned.
- 256 sessions open once at 20/s (quota 25/s); client pools sized to 256 (boto3 default 10, executor ≤ 32).
- The log names the binding tier: `qdepth` (harness), `sandbox_wait_ms` (sandbox), `engine_inflight` (inference).

## `demo_cpu_scale_lanes` — 4 engines : 256 workers : 256 microVMs

**Knobs.** `HARNESS_WORKERS` (256) · `SANDBOX_SESSIONS` (256; 0 = fresh session per episode) · `ROLLOUT_GPUS` (4) · `SANDBOX_OPEN_RATE` (20/s) · `RB × NS` = 32 × 8 = 256 episodes/step.

```bash
python3 harness_scaling_demo.py   # offline throughput sweep
bash 1_prepare_data.sh && bash 2_build_image.sh && bash 3_push_ecr.sh
bash 4_deploy_runtime.sh && bash 5_run.sh && bash 6_cleanup.sh
```

**Isolation.** A session serves **one episode at a time** from a free-list, in a fresh `ep-<hex>` dir that is `rm -rf`'d before the session is handed out again; different sessions are different microVMs. `/tmp`, packages and stray processes are *not* reset, which is safe for DeepScaleR (writes only its workdir); OS-level tasks use `SANDBOX_SESSIONS=0`.

**Metrics.** Live pool check, no trainer (W = M = 64, 192 episodes, real Runtimes): 1280 ops ok; engines 48 / 48 / 47 / 49; a worker used up to 2 sessions and a session served up to 2 workers (the tiers mix). [PENDING: 4 : 256 : 256 RL smoke]. The old 4 × 4 × 4 lane topology passed on AWS (2026-09-23, eval/deepscaler 0.358).

## `demo_cpu_scale_lanes` — where the agent loop runs (vs DSec [31])

```{=latex}
\small
```

**Why DeepSeek moved it out.** DeepSeek's GPU jobs are routinely **preempted**. With the loop inside the GPU pod, the loop died but the sandbox survived, so recovery **replayed a command log**. From V4.1 on, the harness runs on DSec, **outside the GPU pool**: an agent sandbox (scaffold + tools) and a worker container (rollout state). A preempted trainer **reconnects**, with no replay.

| | DSec (V4.1+) | `demo_cpu_scale_lanes` |
|---|---|---|
| Harness runs | agent sandbox, with its tools | **slime's rollout process** (256 workers) |
| Tools run | same agent sandbox | 256 pooled AgentCore microVMs |
| Token record | worker container | slime's in-process `AnthropicAdapter` |
| Trainer lost | reconnect, continue | in-flight episodes abort: one batch |
| Trainer stalls | pause/resume the sandbox | **heartbeat** (default on) |

## `demo_cpu_scale_lanes` — why the agent loop stays in the trainer

**Why we keep it in-process.**

- **No preemption here.** The trainer runs on a dedicated 8×L40S box. A stall is the real risk, and the heartbeat covers it; a crash costs one batch, not correctness.
- **Moving only the harness saves nothing.** The token record and the SGLang engines die with the slime job anyway. DSec's gain also needs an external record and a "reconnect to episode X" path **in slime**.
- **Harness in the tool sandbox has costs.** The policy's commands run next to its transcript and reward, so it could tamper with them. Every microVM needs a network path back to inference (VPC + security-group rule). It would also re-tie the harness tier to the sandbox tier, which is the coupling this demo removes.
- **When to move it:** preemptible or Spot GPUs, or multi-hour episodes. Then put the harness plus its token record in its own service, and have slime fetch episodes by ID. `demo_cpu_branching` already runs a loop (the Claude Code CLI) inside the Runtime session.

# 11 · GPU demos

## `demo_gpu_single` — use case · why · design

```{=latex}
\small
```

**Use case covered**

- GPU-in-the-loop verification: the checker compiles and runs CUDA kernels (KernelBench).

**Why this demo**

- GPUs exist only on AgentCore **Instances** (not microVMs, not the Code Interpreter): the simplest GPU sandbox.

**Design considerations**

- g6 capacity provider + custom Runtime; **warm-shared** session (one g6 per run, per-episode `sub_session`).
- `SNAPSHOT_EVERY=1`: app-level workspace snapshot/resume; there is no usable VM/GPU snapshot on AgentCore.
- `LOCAL_GPU=1` runs the same image as a local farm when no g6 is available.
- Rejected: one g6 per episode (a ~207 s cold boot each).

## `demo_gpu_single` — KernelBench on one warm g6

The policy writes a CUDA `ModelNew`; the checker runs `allclose` + speedup on `cuda` in your custom GPU Runtime on a **g6** Instance. The Code Interpreter has no GPU (`cuda_available=False`), so a custom Runtime is the only way. All episodes share **one warm g6** (`agentcore-ray-farm`), each on a fresh `sub_session`. Train `level_1` (100), eval 20 `level_2` fusions.

**Knobs.** `B4_FARM_MAX_CONCURRENT` · `LOCAL_GPU=1` (on-box farm, no g6) · `SNAPSHOT_EVERY=1` (app-level snapshot/resume at the episode's EXIT, the quiescent point before weight sync).

```bash
bash 1_prepare_data.sh && bash 2_build_image.sh && bash 3_push_ecr.sh
bash 4_deploy_runtime.sh          # Runtime on the g6 capacity provider
bash 5_run.sh                     # SNAPSHOT_EVERY=1 for snapshot/resume
DELETE_CP=1 bash 6_cleanup.sh     # the g6 is the real cost
```

**Metrics.** **Trained** (committed convergence curve); smoke `eval/kernelbench` **0.525**; custom-Runtime probe 0.65; snapshot smoke (local farm) 0.625, live g6 + store round trip unverified. 2026-09-25 re-run: g6 cold boot 207 s, warm opens 150–190 ms; [PENDING: 2026-09-25 remote g6 smoke outcome].

## `demo_gpu_farm` — use case · why · design

```{=latex}
\small
```

**Use case covered**

- KernelBench on GPUs **shared by episodes**: pack two per card, or one whole GPU each.

**Why this demo**

- Session = whole instance, so AgentCore can't split a GPU or give one GPU per session → a Ray farm in one shared session assigns GPUs per `sub_session`. One knob: `FARM=fractional` (default, packing) | `FARM=whole` (the card as its own fault domain).

**Design considerations**

- `fractional`: `num_gpus=0.5` actors (0.5 + 0.5 on GPU 6), one warm CUDA context each; cooperative VRAM cap, not driver-enforced.
- `whole`: `B4_GPU_FRACTION=1.0` on GPUs 6,7 → 2 slots; no VRAM guard needed.
- **Shared kernel + shared GPU: trusted workloads only.** No MIG: more co-tenants ⇒ more `nvcc` load, which can wedge the card; 0.25 (`CONC=4`) is opt-in.
- Open gate == slot count (`B4_FARM_MAX_CONCURRENT`); `CONC ≤ 2`, so no checkout blocking.

## `demo_gpu_farm` — `FARM=fractional` | `FARM=whole`

GPU **sharing** with no AWS GPU: a local sandbox Runtime container owns host GPU 6 (or 6,7); a Ray farm inside it runs long-lived, warm-CUDA actors. One shared `runtimeSessionId` + a per-episode `sub_session` keyed to its actor (AgentCore maps 1 session → 1 instance). slime keeps GPUs 0–5.

```bash
bash 1_prepare_data.sh && bash 2_build_image.sh
FARM=fractional bash 4_deploy_runtime.sh   # local farm: 0.5 + 0.5 on GPU 6 (default)
FARM=whole      bash 4_deploy_runtime.sh   # or: one whole GPU each on GPUs 6,7
FARM=fractional bash 5_run.sh && bash 6_cleanup.sh   # FARM must match step 4
```

**Knobs.** `FARM` · `B4_FARM_GPUS` · `B4_GPU_FRACTION` · `B4_FARM_MAX_CONCURRENT` · `B4_PER_SANDBOX_MEM_FRACTION`.

**Metrics.** `fractional` (2026-09-25 smoke, as `demo_gpu_farm_fractional`): ~83 min; `raw_reward` 0.19–0.75; **`eval/kernelbench` 0.65** (0.775 on 2026-09-21); 359 episodes split **181 / 178** across the two actors, lease wait 0; exec p50 9.0 s / p95 19.8 s; open 4.2 ms. `whole` (2026-09-21, earlier names): **0.65 / 0.70**, `open` == `close`, no leaked leases. [PENDING: merged `demo_gpu_farm` smoke, both `FARM=` modes].

## Isolation on a shared GPU Instance — trusted only

The GPU farm (`sandbox_ray_farm.py`) applies a **uid + `0700` dir + path-confinement + kill-on-close layer** (the CPU pool's uid mode; the CPU pool's Landlock jail is not ported yet), plus:

- **Separate CUDA contexts** per actor: no reading another episode's device memory (driver-enforced).
- **Cooperative memory cap:** `B4_PER_SANDBOX_MEM_FRACTION` is applied in-process (`sitecustomize.py`), not by the driver.
- **Per-uid warm caches:** no slot can plant a kernel another slot loads; `SANDBOX_ISOLATE_HOME=1` makes them per episode.
- **No MIG on L4 / L40S:** one bad kernel can wedge the card for everyone (seen on L4); `FARM=whole` removes GPU sharing, **not** the shared kernel / host.
- **Where the uid layer applies:** it needs CAP_SETUID / SETGID / CHOWN / KILL. The local farm's Docker root has them; **the AgentCore g6 container is uid 0 with no capabilities**, so it falls back to one shared uid (dirs, path confinement, kill still apply); `ping` / `open` report `uid_isolation`. Live (local farm, GPU 6): separate uids, B denied A's files, 0.5 cap held, close left no process.
- Must not share a kernel or GPU? One `runtimeSessionId` per episode, one Instance each.

# 12 · Experimental and summary

## `demo_cpu_branching` (experimental) — use case · why · design

```{=latex}
\small
```

**Use case covered**

- A real Claude Code CLI harness whose episode fans out into sub-agents and compactions.

**Why this demo**

- Multi-trajectory episodes: one rollout → many samples, which every other demo treats as one.

**Design considerations**

- Rollout-level GRPO: fragments share a `rollout_id` and are grouped per rollout.
- CLI in the sandbox (node + CLI image), model calls dial back to the harness proxy (VPC mode).
- The CLI needs the tool-call parser (`qwen25`) and a per-turn cap above Qwen3's `<think>` (2048); without them it stops after one turn or exits 1.

## `demo_cpu_branching` — EXPERIMENTAL

One episode → **many training samples** (fan-out), with the real **Claude Code CLI** as the in-sandbox agent. Each model call dials back to the harness's proxy (`ADAPTER_PORT` 18021, VPC mode), and every fragment carries a shared `rollout_id` for grouping. In Agent Lightning only 36% of rollouts stay one sample [1].

**Knobs.** `BRANCHING_FORK_THRESHOLD` (512) · `AGENT_TIME_BUDGET_SEC` (900) · `ADAPTER_PORT`.

```bash
bash 1_prepare_data.sh && bash 2_build_image.sh && bash 3_push_ecr.sh
bash 4_deploy_runtime.sh          # VPC Runtime + SG->SG dial-back rule
bash 5_run.sh && bash 6_cleanup.sh
```

**Status.** Three blockers found and fixed: the CLI's root guard (`IS_SANDBOX=1`), the missing `qwen25` tool parser (every episode ended after one turn, reward 0), and CLI `exit=1` at the 1024-token per-turn cap (now 2048). Sandbox, CLI, dial-back, parser and checker verified without a trainer. **RL smoke 2026-09-25:** 68 episodes, 0 CLI errors, `fan_out` p50 4 (max 37), reward 0.61 / 0.48, `eval/swe` 0.47. But every advantage was 0 (grouped by per-rollout `index`, not the prompt group); fixed with a regression test, learning [PENDING: branching re-run with the advantage fix].

## Summary — 5 demos + 1 experimental + 1 side note

```{=latex}
\scriptsize
```

| Demo | Shows | Isolation unit | Smoke headline |
|------------------|-----------|-----------|-----------|
| `demo_cpu_tasks` | literature tasks on the **default** runtime | microVM per episode | gsm8k 0.90; libs 1.0 |
| `demo_cpu_terminalbench` | ~229 images; pre-open + heartbeat | fresh microVM per episode | wait p95 0; pre-open −15–20 % |
| `demo_cpu_scale_lanes` | 4 engines : 256 workers : 256 microVMs, decoupled | pooled microVM, one episode at a time, fresh workdir | pool check ok; [PENDING: RL smoke] |
| `demo_gpu_single` | KernelBench on one warm g6 | `sub_session` on a shared g6 | trained; smoke 0.525 |
| `demo_gpu_farm` | `FARM=` fractional / whole | shared kernel + card / own GPU | eval 0.65 / 0.65–0.70 |
| *`demo_cpu_branching`* | CLI fan-out (experimental) | microVM | fans out, eval 0.47; advantage fix re-run pending |
| *`…instance_pool`* | side note: warm m5 pool | jail, shared kernel | 2 boots per run |

## Takeaways

- **Start on the Code Interpreter.** Most literature tasks need no image: no Dockerfile, no ECR, no Runtime.
- **Add an image only when the environment differs**: a pinned OS, > 2 GB, CUDA, or a missing package. Route by `task.image`; AgentCore only ever sees an image and a session.
- **Hide boots with pre-open, bridge stalls with heartbeat** on your own Runtimes; on the Code Interpreter, set the session timeout.
- **Share an Instance only for trusted workloads**: the isolation layer (Landlock jail, private dir, kill-on-close) is not full isolation; kernel, network, `/proc`, CPU/RAM and the GPU stay shared. **Need full isolation? One microVM or Instance per session.**

## References

```{=latex}
\tiny
```

**Papers & code.** [1] Agent Lightning <https://arxiv.org/abs/2508.03680> · [2] DeepSeek-R1 <https://arxiv.org/abs/2501.12948> · [3] Qwen3 <https://arxiv.org/abs/2505.09388> · [5] Kimi K2 <https://arxiv.org/abs/2507.20534> · [7] KernelBench <https://huggingface.co/datasets/ScalingIntelligence/KernelBench> · [9] slime <https://github.com/THUDM/slime> · [31] DSec (DeepSeek Elastic Compute) <https://arxiv.org/abs/2609.22978>

**AgentCore docs.** [D1] What is AgentCore <https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/what-is-bedrock-agentcore.html> · [D2] Runtime <https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agents-tools-runtime.html> · [D3] How it works <https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-how-it-works.html> · [D4] Lifecycle <https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-lifecycle-settings.html> · [D5] Instances (GPU) <https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-instances-how-it-works.html> · [D6] Code Interpreter <https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/code-interpreter-tool.html> · [D8] InvokeAgentRuntime <https://docs.aws.amazon.com/bedrock-agentcore/latest/APIReference/API_InvokeAgentRuntime.html> · [D9] CreateAgentRuntime <https://docs.aws.amazon.com/bedrock-agentcore-control/latest/APIReference/API_CreateAgentRuntime.html> · [D10] NetworkConfiguration <https://docs.aws.amazon.com/bedrock-agentcore-control/latest/APIReference/API_NetworkConfiguration.html> · [D11] VPC endpoints <https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/vpc-interface-endpoints.html> · [D12] StopRuntimeSession <https://docs.aws.amazon.com/bedrock-agentcore/latest/APIReference/API_StopRuntimeSession.html> · [D13] Observability <https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/observability.html> · [D14] AgentCore quotas <https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/bedrock-agentcore-limits.html> (values: Service Quotas, this account, us-west-2, 2026-09-26) · [D15] ECR quotas <https://docs.aws.amazon.com/AmazonECR/latest/userguide/service-quotas.html>

**Tasks.** MBPP <https://huggingface.co/datasets/google-research-datasets/mbpp> · HumanEval <https://huggingface.co/datasets/openai/openai_humaneval> · GSM8K <https://huggingface.co/datasets/openai/gsm8k> · MATH-500 <https://huggingface.co/datasets/HuggingFaceH4/MATH-500> · DeepScaleR <https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset> · Gretel SQL <https://huggingface.co/datasets/gretelai/synthetic_text_to_sql> · HotpotQA <https://huggingface.co/datasets/hotpotqa/hotpot_qa> · xLAM <https://huggingface.co/datasets/minpeter/xlam-function-calling-60k-parsed> · ToolACE <https://huggingface.co/datasets/Team-ACE/ToolACE> · Terminal-Bench <https://github.com/laude-institute/terminal-bench>

**Repo.** `code/demos/README.md` (catalogue + step scripts) · `docs/report/agentcore_slime_report.md` (full results) · `docs/agentcore-networking.md` · `docs/gpu-snapshot.md`

---

*Thank you — Questions?*
