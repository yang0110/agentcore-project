# AgentCore as the Sandbox in Agentic RL

**Kaige Yang** — Senior Applied Scientist, 亚马逊云科技 (AWS)

September 25, 2026

---

## Outline

1. **Agentic RL** — architecture and pipeline: trainer, inference, harness, sandbox.
2. **AgentCore foundations** — Runtime vs Code Interpreter, microVM vs Instance, session lifecycle, limits.
3. **Using AgentCore in slime** — one hook, no fork; deploy, wire and network a Runtime.
4. **Demo: Terminal-Bench** — custom images at scale; pre-open and heartbeat hide the cold start.
5. **Demo: scale lanes** — scaling inference, harness and sandbox tiers independently.
6. **Other demos** — the same hook on other substrates.
7. **Conclusion** — DeepSeek DSec vs AgentCore, and takeaways.


# 1 · Agentic RL — architecture and pipeline

## Agentic RL — architecture

![Agentic RL: trainer, inference, harness, sandbox](figures/lanes_fig02_rl-loop.pdf){ height=42% }

- **Trainer** (slime: Megatron, GRPO) updates the policy; **inference** (SGLang engines) serves it.
- **Harness** runs the agent loop; a proxy records the exact `token_ids` + logprobs (**TITO**), so training stays on-policy.
- **Sandbox** (AgentCore) runs the tool calls and the **checker: the sandbox is the reward source**.

## Agentic RL — pipeline

![slime pipeline: rollout → data buffer → training → weight sync](figures/fig15_slime-architecture.png){ width=62% }

1. **Seed:** the harness opens a sandbox session and writes the task's files.
2. **Act → observe:** policy turn → tool call runs in the sandbox → output fed back, for several turns.
3. **Check:** a hidden checker runs in the sandbox → binary reward (RLVR, no LLM judge).
4. **Pack:** one loss-masked `Sample` per episode → data buffer.
5. **Train + sync:** a GRPO step, weights pushed to SGLang; rollout runs **async** with training.

# 2 · AgentCore foundations

## AgentCore — Runtime, Code Interpreter, two substrates

- **Runtime:** your image, serverless, **per-session isolated compute**, called with `InvokeAgentRuntime` (SigV4).
- **Code Interpreter:** a managed CPU sandbox, no image to build.
- Same loop code on both substrates; only the Runtime ARN differs. **Both demos use microVMs.**

| | **microVM** (serverless) | **Instance** (EC2 via a capacity provider) |
|---|---|---|
| Compute | own guest kernel, **CPU only** | `m5` (CPU) · `g6` / `g5` (**GPU**, whole cards, no MIG) |
| Image | **arm64**, ≤ ~**2 GB** | **amd64**, instance disk |
| Session length | ≤ 8 h | ≤ 14 days |

: microVM vs Instance substrates

## Runtime · session · lifecycle

- **1 : N : 1.** Runtime = one image, one ARN; session = one `runtimeSessionId` = one microVM / instance. **Runtimes track images, not tasks.**
- **Launch:** no "create session" API; the first invoke under a new id cold-starts the compute.
- **End:** idle timeout, max lifetime, or `StopRuntimeSession`.

```{=latex}
\small
```

| Substrate | Cold launch (measured) | Reused open |
|---|---|---|
| Code Interpreter | p50 **1.83 s**, **every episode** | – |
| Own Runtime, microVM | **≈ 0.77 s** (stdlib) · **≈ 15 s** (447 MB) | pre-open wait **5 ms** |
| Instance `m5` · `g6` | **≈ 60–80 s** · **≈ 123–207 s** | 0.12–0.33 s |

: Measured launch cost by substrate

## One RL step in AgentCore terms — a worked example

**Batch:** 16 prompts, **each needs its own image**; GRPO samples **8 rollouts per prompt**.

- **AgentCore never sees prompts or rewards**, only an ARN + session id; the harness maps them.
- **Limits:** 128 opens at 25 /s ≥ **5 s**/step; 128 ≪ 5000 concurrent; Runtimes = **distinct images in the dataset** (≤ 1000).
- **Cost:** 128 cold boots per step (0.8–15 s each) unless pre-opened (§4).

```{=latex}
\footnotesize
```

| RL concept | AgentCore object | Count for this batch |
|------------|--------------------------|--------------------------------------------------|
| Prompt env | 1 image → 1 **Runtime** (ARN) | **16 Runtimes**, deployed once, reused every step |
| Rollout | 1 **session** (session id) | 16 × 8 = **128 sessions**, new every step |
| Its compute | 1 **microVM** per session | **128 microVMs** in flight, stopped at episode end |
| GRPO group | 8 sessions on **one** Runtime | 8 separate microVMs, nothing shared |

: One batch mapped onto Runtimes, sessions and microVMs

## Limits that shape an RL sandbox

```{=latex}
\footnotesize
```

| Limit | Value | What we do |
|--------------|------------|----------------------|
| **Idle timeout** | 15 min default (we set 3600 s) | reclaim is **silent** (empty sandbox) → heartbeat |
| Max lifetime | microVM **8 h** · Instance 14 d | re-seed long episodes |
| **New sessions** | **25 /s** (adj.) | caps the cold-start burst → pre-open ahead |
| Concurrent sessions | **5000** (adj.) | one per in-flight episode |
| Code Interpreter | **30 /s** start and invoke | ≈ 75 episodes in flight |
| **Runtimes per account** | **1000**; create ≤ 50 /s | ≤ ~1000 custom images (Terminal-Bench ~229) |
| Image | microVM **2 GB**, arm64 | bigger → slim it or use an Instance |
| Per request | **15 min** · 100 MB | long `exec` runs async, polled |
| Privileges | **no Linux capabilities** | no per-episode uid / cgroup inside one session |

: AgentCore limits and how we handle them

# 3 · Using AgentCore in slime

## Hook AgentCore into slime — no fork

- `--custom-generate-function-path` → `generate.generate` (runs the harness, records turns); `--custom-rm-path` → the reward.

```python
async def generate(args, sample, sampling_params):        # slime hook
    task = task_from_dict(sample.metadata["task"])
    async with make_sandbox(os.environ["SANDBOX"]) as sb:  # session opened / closed here
        traj = await run_episode(task, policy, sb)
    return build_token_stream(traj, sample)               # loss-masked Sample
```

- Each sandbox op = one `invoke_agent_runtime(ARN, session_id)` in a thread → **N episodes concurrently**.
- `__aexit__` ends the session when the episode returns: one teardown path.
- `SANDBOX=agentcore` → Code Interpreter · `SANDBOX=agentcore-runtime` → your own Runtime.

## Deploy, wire and network a Runtime

1. **Build + deploy:** image (**arm64** microVM / **amd64** Instance) → ECR → `CreateAgentRuntime`. The image serves `sandbox_server_agent.py`: `write_file` / `exec` / `read_file`.
2. **IAM:** an execution role (ECR pull, logs) + an operator role with `InvokeAgentRuntime`. Instances also need `CreateCapacityProvider`.
3. **Wire** (`deploy/runtime.env`): `AGENTCORE_SANDBOX_RUNTIME_ARN` · `…_ARNS` (round-robin) · `…_ARN_BY_IMAGE` (per image).

**Networking — pull, don't expose.** SigV4 invoke, **no inbound port**. Only dial-back (in-session harness → trainer) needs ingress: one SG→SG rule under `networkMode=VPC`, plus interface endpoints (`ecr.api`, `ecr.dkr`, `s3`, `logs`, `sts`, `bedrock-agentcore`); a missing one fails create with a **null reason**.

Each demo wraps this as `2_build_image.sh` → `3_push_ecr.sh` → `4_deploy_runtime.sh`.

# 4 · Demo: Terminal-Bench — custom images at scale

## Terminal-Bench — the problem

- **~229 terminal / SWE tasks, each its own environment**: its own OS, packages and a deliberately broken state. Reward = 1 only if every hidden `pytest` passes.
- The environment *is* the task → **one arm64 image + one Runtime per task**, routed by task id.
- Tasks mutate the whole OS (apt, `/etc`, services) → **a fresh microVM per episode**; the session is the isolation boundary.

![Image 1:1 Runtime 1:N session 1:1 microVM](figures/tb_fig01_cardinality.pdf){ height=30% }

**229 tasks → 218 deployable** (arm64, ≤ 2 GB) **→ 166 oracle-usable → 135 train / 31 eval.**

## Terminal-Bench — architecture of one episode

![One Terminal-Bench episode: route → fresh session → agent loop → grade → stop](figures/tb_fig03_episode-workflow.pdf){ height=46% }

- **Route → fresh session → agent loop → grade → stop.** The first invoke with a new `runtimeSessionId` **is** the boot.
- Hidden tests are injected only **after** the agent finishes; `StopRuntimeSession` terminates the microVM.
- One step = RB 4 × NS 8 = **32 episodes = 32 fresh microVMs**.

## Terminal-Bench — key challenges

```{=latex}
\small
```

| Challenge | Why it hurts |
|------------------|------------------------------------------------------|
| **Cold start every episode** | ≈ 15 s for a 447 MB image; 32 per step ≈ **8 min** of serial boot if on the critical path |
| **Bursty opens** | all 8 GRPO siblings of a task start at once; account cap **25 new sessions/s** |
| **slime can't hide it** | only `CONC × engines` = 8 episodes visible; step N+1 drawn only after step N; slime knows nothing about sandboxes |
| **Silent idle reclaim** | a session idle ≈ 15 min is reclaimed (same id, empty sandbox), and weight sync can be that long |
| **Broken / oversized tasks** | 7 don't build on arm64, 4 exceed the 2 GB cap, 47 fail their own reference solution |
| **Runtime cap** | 1000 Runtimes per account → at most ~1000 images |

: Terminal-Bench key challenges

## Terminal-Bench — our solutions

```{=latex}
\small
```

| Challenge | Solution |
|------------------|------------------------------------------------------|
| Isolation vs reuse | **Fresh microVM per episode**; reuse + reset rejected (OS state leaks into the reward) |
| Many images | **One image + Runtime per task**, strict routing (`tasks/arn_by_image.json`; unmapped = error) |
| Broken tasks | **Oracle check**: untouched env must FAIL, reference must PASS → 166 usable |
| Cold start | **Look-ahead pre-open**: boot the next 2 batches' sessions before slime asks |
| Idle reclaim | **Heartbeat** (`SessionKeeper`) pings every open session |
| Infra failures | **Retry on a new id** (3×); logged as `infra_error`, never scored as a wrong answer |

: Terminal-Bench challenges and our solutions

## Pre-open — boot the next batches while this one runs

1. **Look-ahead data source** (`tb_lookahead_source.py`, a `--data-source-path` plugin) replays slime's deterministic prompt order → the next 2 rollouts' prompts. **No slime change.**
2. **Bounded window:** `TB_PREOPEN_BATCHES=2` → **P = 64** sessions booted ahead, pooled per task; each claim starts the next boot. `1` → 32, `0` → off.
3. **Sizing:** boot hidden when P · E / W ≥ boot time; beyond that is idle cost. Tune by `open_wait_ms` (≫ 0 → raise) vs `preopen_idle_ms` (the price).

```{=latex}
\small
```

| Warm-start probe (24 episodes, batches of 6) | Mode | Wait p50 | Waited | Makespan |
|---------|------|----:|-----:|------:|
| own Runtime | cold → **preopen** | 774 → **5 ms** | 24 → 8 /24 | 31.9 → 25.5 s (**−20 %**) |
| Code Interpreter | cold → **preopen** | 1715 → **0 ms** | 24 → 6 /24 | 28.7 → 24.3 s (**−15 %**) |

: Warm-start probe: cold open vs pre-open

## Heartbeat — surviving the weight-sync stall

- During weight sync no op reaches the sandbox → your own Runtime **silently reclaims** idle sessions (same id, empty sandbox, no error).
- `SessionKeeper` pings any session idle past `fraction × timeout` with `exec("true")`.
- **On by default** for own Runtimes (`HEARTBEAT=0` = off), idle timeout 3600 s as backstop. Code Interpreter: set a long session timeout instead.

```{=latex}
\small
```

| Backend | Heartbeat | Survived 150 s idle (60 s timeout) | Resume |
|---------|---------|----:|--------------------|
| own Runtime | on | **3/3** | 104 ms, files intact |
| own Runtime | off | 0/3 | **silently reclaimed** |
| Code Interpreter | on / off | 0/3 | expires at `sessionTimeoutSeconds` |

: Heartbeat vs silent idle reclaim

## Terminal-Bench — results (smoke, 2026-09-23)

```{=latex}
\small
```

| Metric | Value |
|------------------------------------|-------------------------------------------------------------|
| Episodes / infra errors | 382, each on its own fresh microVM / **0** |
| Boot on the critical path (`open_wait_ms`) | **p50 0 · p95 0** — pre-open hides it fully |
| Session open | p50 0.66 s · **100 % warm hits** · 0 retries |
| Pre-opened VM idle p50 | **7.3 min** for a 21 s episode → ≈ 95 % idle → try `TB_PREOPEN_BATCHES=1` |
| Solves | **4 / 382** train, eval 0.0; 42 % pass ≥ 1 hidden test → sparse; next: curriculum / partial credit |

: Terminal-Bench smoke results

# 5 · Demo: scale lanes — scaling the tiers independently

## What to scale — three independent tiers

**throughput = min(inference, harness, sandbox)**: each tier is its own pool, sized for its own reason, and joined per episode. Pinned together (worker = engine = microVM), the slowest pin sets the pace. **Task:** DeepScaleR on a generic stdlib image.

```{=latex}
\small
```

| Tier | Scales with | Limited by | Signal that it binds |
|-----------|------------------|------------------------|------------------------|
| **Inference** (SGLang) | GPUs / engines | decode throughput | `engine_inflight` |
| **Harness** (agent loops) | Python concurrency | event loop, client pools | `qdepth` |
| **Sandbox** (AgentCore) | sessions in flight | **25 new/s**, 5000 concurrent, cost | `sandbox_wait_ms` |

: The three tiers to scale


## How many — engines, harness, sandboxes

**Client pools sized to 256** (boto3 default 10, a silent cap). Shrink one tier to make it the ceiling, then read its signal.

```{=latex}
\footnotesize
```

| Tier | Default | Knob | Why this number |
|-----------|------------------|------------------|--------------------------------------------|
| **SGLang** | **4 engines**, TP1, GPUs 4–7 | `ROLLOUT_GPUS` (4) | Qwen3-4B fits one L40S; other 4 GPUs train |
| **Harness** | **256** dispatchers | `HARNESS_WORKERS` (256) | = one step's batch, RB 32 × NS 8 = 256; CONC = 64 per engine |
| **Sandbox** | **256** fresh sessions in flight | `SANDBOX_SESSIONS` (0 = fresh per episode; N = N pooled) | one per in-flight episode |
| Open rate | **20 /s** | `SANDBOX_OPEN_RATE` | under the 25 /s quota → 256 open in ≈ 13 s |

: Tier sizing in scale lanes

## scale lanes — architecture

![scale lanes: 4 engines : 256 dispatchers : 256 fresh sessions, agent loop in the session](figures/lanes_fig13_session.pdf){ height=55% }

- Per episode: a free dispatcher takes it, opens a **fresh session**, starts the agent loop **inside** it; the loop dials back to the **least-loaded engine** (sticky for its turns). Nothing is pinned.

## Where the agent loop runs — in the sandbox session

- **One agent loop per sandbox, in the same session** (`HARNESS_WHERE=session`, default on own Runtimes), like DSec.
- **Per episode:** write + start `session_runner.py` → long-poll → read transcript → **kill the runner, then the hidden grader** runs (the loop can't touch its reward).
- **Model calls dial back** (VPC, SG→SG) to the engine's `AnthropicAdapter`, which records the exact tokens (TITO).
- **So the harness tier scales with the sandbox tier**: trainer-side dispatchers only start and poll.
- **Fallback** (no `python3` or no path back, e.g. Code Interpreter) → loop in the trainer; logged as `harness_where`.

## scale lanes — run, isolation, results

```bash
python3 harness_scaling_demo.py   # offline throughput sweep
bash 1_prepare_data.sh && bash 2_build_image.sh && bash 3_push_ecr.sh
bash 4_deploy_runtime.sh && bash 5_run.sh && bash 6_cleanup.sh
```

- **Isolation:** default = fresh microVM per episode, stopped after grading. `SANDBOX_SESSIONS=N` pools N microVMs with a fresh `ep-<hex>` workdir only (`/tmp`, packages not reset → fine for DeepScaleR).
- **Session mode, live:** verified per episode (good solution 1.0, cheat 0.0).
- **Live pool check** (64 : 64, 192 episodes): 1280 ops ok; engines 48 / 48 / 47 / 49.
- **RL smoke:** the 4 × 4 × 4 topology passed on AWS (eval/deepscaler 0.358).

# 6 · Other demos in the codebase

## Other demos — same hook, other substrates

Each is a standalone folder in `code/demos/` with the same `1_prepare_data.sh` … `6_cleanup.sh` steps.

- **`demo_cpu_tasks`** — literature tasks (MBPP, GSM8K, MATH-500, DeepScaleR, HumanEval, SQL, HotpotQA, xLAM, MCP, mixes) on the **managed Code Interpreter**, no image.
- **`demo_cpu_swebench`** — one image + Runtime per **SWE-bench Verified** instance on **m5 Instances** (images > 2 GB microVM cap); reuses the Terminal-Bench pool.
- **`demo_gpu_single`** — **KernelBench** CUDA RL on a **g6 GPU Instance**, one warm custom Runtime; optional app-level snapshot.
- **`demo_gpu_farm`** — **one GPU instance serving many sessions** via a Ray farm (`FARM=fractional` 0.5 + 0.5 or `whole`); trusted-only.
- **`demo_cpu_instance_pool`** *(side note)* — a warm **m5** pool for a ~4 GB image, one Landlock jail per episode; shared kernel → not full isolation.
- **`demo_cpu_branching`** *(experimental)* — the **real Claude Code CLI** as harness; one episode fans out into many samples.

# 7 · Conclusion

## DeepSeek DSec vs AgentCore {frameoptions="t"}

```{=latex}
\begin{figure}\vspace*{-4mm}\centering\includegraphics[height=0.80\textheight,keepaspectratio]{figures/dsec_vs_agentcore.png}\caption{DeepSeek DSec vs AgentCore as an RL sandbox}\end{figure}
```

## What we take from DSec

- **Their median sandbox lives about as long as our idle timeout** → heartbeat on by default; idle timeout 3600 s.
- **New metrics, every demo:** `perf/sandbox_ready_span_s`, `perf/sandbox_open_peak_per_s` (vs 25/s), per-session lifetime + longest idle gap (`<log>.sessions.csv`).
- **Terminal-Bench, re-parsed:** 414 sessions, lifetime p50 **471 s** / p95 **1841 s**; 55 outlived 900 s (kept by heartbeat); peak **36 opens/s**, **0 throttles**.
- **Not exposed by AgentCore:** image layers, session pause, thousands of creates/s. **Copied:** agent loop in the sandbox session (§5). **Still to copy:** per-task network allowlist.

## Takeaways

- **Start on the Code Interpreter**: most literature tasks need no image.
- **Add an image only when the environment differs**; route by `task.image`, AgentCore sees only an image and a session.
- **Hide boots with pre-open, bridge stalls with heartbeat** on your own Runtimes.
- **Scale the tiers independently**; diagnose the binding tier before adding capacity.
- **Isolation = the session**: fresh microVM when tasks mutate the OS; pooled microVM + fresh workdir for workdir-only tasks.

---

*Thank you — Questions?*
