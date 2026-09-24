# Scaling lanes on AgentCore

**Kaige Yang** — Senior Applied Scientist, AWS

September 24, 2026

---

## Summary

| | |
|--------------|------------------------------------------------------------------------------|
| **Goal** | Scale an agentic-RL rollout (slime GRPO, Qwen3-4B) by scaling **all three tiers** together: harness, inference, sandbox |
| **Challenge** | Throughput = **min(inference, sandbox, harness)**. A single harness is a hidden ceiling, the router scatters an episode's turns, and a fresh session per episode puts an open on every episode |
| **Solution** | **Lanes, 1 : 1 : 1:** W warm harness workers, each pinned to its own SGLang engine and its own **persistent AgentCore microVM**, with a fresh workdir per episode |
| **Result** | Smoke run on AWS (4 × 4 × 4): 400 episodes, lanes verified 1:1:1, **4 session opens instead of 400**, sandbox p95 **431 → 120 ms**. Rollout time unchanged: this task is **inference-bound** |

: The deck in one table

Source: `code/demos/demo_cpu_scale_lanes/`, `docs/harness-scaling.md`.

# 1 · AgentCore, seen from agentic RL

## Where AgentCore sits in the RL loop

An agentic-RL rollout is produced by **three replicated tiers**. AgentCore is the **sandbox** tier.

| RL tier | Role in the episode | In this demo |
|------------------|------------------------------------------------|-----------------------------------------|
| **Harness** | Runs the agent loop: prompt → model → tool call → observation → … → reward | `MiniClaudeHarness`, in the slime rollout process |
| **Inference** | The **policy under training**; every model call is a sample we train on | SGLang engines, weights synced from slime each step |
| **Sandbox** | The **environment**: executes the agent's tool calls, holds its files, runs the checker | **AgentCore Runtime microVMs** |

: The three tiers

- slime only sees `generate()` → a scored trajectory. Tiers are wired in through its **custom generate hook**, no slime fork.
- **Throughput = min(inference, sandbox, harness)**: scale one tier alone and the smallest one caps the rollout.

## Three AgentCore concepts, and what each means for RL

| AgentCore | What it is | RL meaning |
|----------------------|------------------------------------------------|------------------------------------------|
| **Runtime** (ARN) | A template: one container image + entry point | An **environment type** (tool set, OS, packages). One per image, not per task |
| **Session** (`runtimeSessionId`) | The routing key; same id → same VM | An **episode's handle**: sticky for the episode's life |
| **microVM** | Firecracker VM (2 vCPU, 8 GB), 1 : 1 with a session | The **isolation boundary**: a new session is a clean reset |

: Runtime 1 : N sessions, session 1 : 1 microVM

- **No "create session" call:** the first invoke with a new id **is** the boot (cold ≈ seconds, warm hit ≈ 0.5 s).
- **Reset = a new session.** `StopRuntimeSession` terminates and sanitises the VM.
- One Runtime serves many concurrent sessions, so **GRPO siblings share a Runtime but never a VM**.

## The lifecycle, read as rollout costs

| AgentCore behaviour | Consequence for an RL rollout |
|--------------------------------------------|---------------------------------------------------------|
| Boot on first invoke | A fresh VM per episode puts an **open on every episode**; bursty at step start |
| **15-min idle reclaim**, 8 h max lifetime [D3] | A VM kept across steps must be pinged through weight sync (heartbeat) |
| 5000 active sessions, **25 new/s** [Q2] | Concurrency is cheap; the **open rate** is the limit for bursts |
| Image ≤ 2 GB, arm64, **CPU only** [Q1] | Tool/verifier environments fit; GPU sandboxes need Instances |
| Per-session isolation | Choose per task: **fresh VM** (OS-level tasks) or **persistent VM + fresh workdir** (file-level tasks) |

: Lifecycle → cost

- This deck is the **persistent VM** end of that choice; the Terminal-Bench deck is the **fresh VM** end.

# 2 · Setup

## The task and the episode

- **Task:** DeepScaleR math posed as a **coding task**. The agent writes `solution.py` that sets `answer`; a hidden `run_checks.py` compares it (tol 1e-4) → reward 0/1.
- **Harness:** `MiniClaudeHarness`, 3 tools (`bash`, `read_file`, `write_file`), ≤ 4 turns, ≤ 1024 tokens per turn.
- **Trainer:** slime GRPO, stock driver (Mode A), Qwen3-4B. RB 4 prompts × NS 8 samples = **32 episodes per step**; 10 train steps + eval.
- **GPUs:** 4 actor GPUs (0–3) + 4 rollout GPUs (4–7) = **4 SGLang engines, TP1**.
- **Sandbox:** 2 AgentCore CPU Runtimes (`acs_sandbox_cpu`, `acs_sandbox_cpu_1`).

**Why this task:** its checker only reads files **inside the episode's workdir**, so episodes don't leak state through the OS. That is what makes a persistent VM safe (§4).

# 3 · The challenge: three tiers, three load balancers

## Each tier balances load differently

| Tier | State | Default load balancing | Granularity |
|--------------|-------------------------------|-------------------------------------------------|-------------------|
| **Inference** | Stateless per request | slime's SGLang router, least-loaded / cache-aware | per **request** |
| **Sandbox** | Stateful (the episode's files) | New session per episode, round-robin over K ARNs | per **episode** |
| **Harness** | Warm (connection + trajectory) | Whoever calls `generate()` drives it | per **episode** |

: The default routing

The asymmetry is forced by state: an engine holds nothing between calls, a sandbox holds the episode's files, a harness worker holds a live connection and the trajectory.

## What goes wrong by default

1. **The harness is a hidden ceiling.** One process driving every episode is bounded by its own concurrency. Add engines and Runtimes and throughput still flat-lines.
2. **The router scatters an episode.** Per-request LB sends turn 2 to a different engine than turn 1, so the multi-turn prefix is re-prefilled instead of cache-hit.
3. **A session per episode opens a VM per episode.** 400 episodes = 400 opens (p50 0.41 s, max 0.89 s), and the extra round-trips widen the sandbox tail (p95 431 ms).

Scaling "more of everything" doesn't fix any of these: each tier has to be scaled **and** routed deliberately.

# 4 · The solution: lanes

## One lane = one worker : one engine : one microVM

![](figures/lanes_fig01_lanes.pdf){width=92%}

*Figure 1: LANES = 4 → 4 workers : 4 engines : 4 persistent microVMs.*

- `LANES=N` sets all three tiers to N. `HARNESS_WORKERS` alone moves only the harness tier.

## Four mechanisms (`harness_pool.py`)

1. **Warm worker pool, pull-based.** W asyncio workers pull whole episodes from a shared queue (work-stealing): a worker stuck on a slow episode simply doesn't pull more.
2. **Engine pin.** At start-up each worker gets **its own `AnthropicAdapter`** pointed at engine `w mod E` (engines discovered from the router). All its turns hit one engine, so prefix-cache hits stay local. No slime patch.
3. **Persistent lane microVM.** Each worker opens **one** session (`lane<w>-<hex>`, ARN `w mod K`) and keeps it for the run. Each episode gets a fresh `ep-<hex>` workdir, removed after. `LANE_SANDBOX=0` restores a fresh session per episode.
4. **Admission sized to the lanes.** `CONC = ⌈2W / E⌉`, so slime admits S = CONC × E = **2W** episodes: W being driven, W queued, so no worker idles between episodes.

## When a persistent VM is safe

| | scale_lanes (this deck) | Terminal-Bench |
|----------------------|-------------------------------------|-------------------------------------|
| What the agent changes | Files in its workdir | The whole OS: apt, `/etc`, services |
| What the checker reads | Only the episode's workdir | System state |
| Reset | `rm -rf ep-<hex>/` (ms) | New session = new microVM (boot) |
| Sessions per run | **W** (4) | **one per episode** (382) |
| Cold-start fix | Open once per lane | Look-ahead pre-open |

: Same AgentCore, opposite isolation choices

**The honest caveat:** the workers are asyncio tasks in one process. They lift the **IO ceiling** (episodes are mostly waiting on HTTP), not the **CPU ceiling**: the GIL still serialises tokenisation and trajectory flattening. If profiling shows CPU-bound, move to processes, not more coroutines.

# 5 · Results

## Offline: the scaling law, measured (`harness_scaling_demo.py`)

| Harness-only sweep, W | 1 | 2 | 4 | 6 | **8** | 12 |
|---------------------------------|------|------|------|------|------|------|
| Throughput (ep/s) | 5.3 | 10.5 | 21.0 | 27.8 | **34.3** | 34.2 |
| Binding tier | harness | harness | harness | harness | **inf.** | inf. |

: Harness alone: plateau at the inference ceiling (40 ep/s)

| Balanced sweep, N + N + N | 1 | 2 | 4 | 8 |
|---------------------------------|------|------|------|------|
| Throughput (ep/s) | 5.3 | 10.5 | 20.9 | **41.4** |

: All three tiers together: 8× tiers → 7.8× throughput

Real `run_episode`, modelled tier ceilings, measured wall-clock. Raising W alone stops paying once another tier binds; scaling **jointly** stays near-linear.

## AWS smoke run: lanes work (4 × 4 × 4, 2026-09-23)

| Metric | Value |
|------------------------------------|-------------------------------------------------------------|
| Episodes / job | 400 over 10 steps + eval; job succeeded (≈ 24 min) |
| Lanes 1:1:1 (`check_stickiness`) | ✓ every episode on its lane's engine **and** lane's VM |
| Episodes per lane 0–3 | 103 / 101 / 99 / 97 (pull queue keeps them balanced) |
| Session opens | **4** for the whole run (1.3–6.4 s each; 2 cold, 2 warm) |
| Warm sandbox ops (n = 2959) | p50 **80 ms**, p95 **120 ms** (exec 98 · write 75 · read 73 ms p50) |
| Queue depth | p50 3, max 7 (≈ W queued, as admission intends) |
| Reward (train, per step) | 0.19 – 0.50, mean 0.36 |
| Held-out `eval/deepscaler` | **0.358** at step 9 |

: Smoke run (RB = 4, NS = 8, Qwen3-4B)

## Before / after: fresh session per episode vs lanes

| | Fresh session per episode (2026-09-18) | Lanes (2026-09-23) |
|--------------------------|--------------------------------|--------------------------------|
| Engines | 2 × TP2 | 4 × TP1 |
| Sandbox sessions opened | 400 (p50 0.41 s, Σ 177 s) | **4** (Σ 12 s) |
| Sandbox op p50 / **p95** | 90 / **431** ms | 80 / **120** ms |
| Rollout time per step | 61 s | 64 s |
| Train reward / eval | 0.33 / 0.329 | 0.36 / 0.358 |

: Same task and harness; engine layout differs, so rollout time is not a clean A/B

- The sandbox side improved as designed: **100× fewer opens, 3.6× lower p95**.
- **Rollout time didn't move,** and that is the real finding (next slide).

## This workload is inference-bound

- An episode takes ≈ **8 s** (64 s rollout ÷ 8 episodes per worker). The sandbox share is ≈ **0.65 s** (7.4 ops × 88 ms).
- So **≈ 90 % of an episode is SGLang generation**; the sandbox was never the bottleneck, and shaving it barely moves the step.
- The 4 engines are also **under-used**: with 1 lane per engine, each engine serves ~1 request at a time.

**Lesson:** a sandbox scaling demo needs a task where the **sandbox does the work**. Math is the wrong task to show AgentCore's elastic compute.

# 6 · Next: 256 sandboxes on 4 engines

## Choosing a task the engines can keep up with

With N sandboxes in flight, each episode generating G tokens over T seconds, the engines keep up only if

**N · G / T ≤ C** (C = aggregate decode rate of the 4 engines).

- Estimate C ≈ 10 k tok/s (Qwen3-4B, TP1 on L40S, batch ~64 per engine; **to be measured**). With N = 256, G ≈ 1.5 k → **T ≥ 40 s**. Target 60–90 s of real sandbox CPU per episode for 2× headroom.
- **Task: slow-verifier coding,** e.g. performance optimisation. Short generations; each "run tests" is a heavy suite (property-based fuzzing, timed benchmark). Reward = tests passed + speed-up.
- One knob, `EXEC_BUDGET_S`, sets the sandbox : inference ratio, so we can **move the bottleneck on purpose** and show the crossover.

## Sizing the harness and slime

| Knob | Value | Why |
|----------------|--------------------------------|--------------------------------------------------|
| Harness workers W | **256** (one per sandbox) | A worker drives one episode at a time and mostly awaits exec: cheap |
| Layout | 4 lanes × 64 workers | Worker w → engine w mod 4 (prefix cache); own persistent VM each |
| Adapters | **4** (one per engine) | Not 256 threads/servers; workers share their engine's adapter |
| Harness CPU | ≈ 17 turns/s × 5–10 ms | ≤ 20 % of a core: one process is fine (profile to confirm) |
| `CONC` | ⌈2 · 256 / 4⌉ = 128 | slime admits 512: 256 driven + 256 queued |
| Batch | RB 64 × NS 8 = 512 / step | A queue refills stragglers at the end of each step |
| Opens | 256 at ≤ 25/s ≥ 11 s, over K = 4–8 ARNs | New-session rate quota [Q2]; 256 ≪ 5000 active |

: The 256-sandbox configuration

**Proof:** sweep N = 32 / 64 / 128 / 256 at E = 4: throughput should rise ~linearly while SGLang running-requests and per-call latency stay flat. Offline simulator first, then AWS.

## Takeaways

- **AgentCore maps cleanly onto the sandbox tier:** Runtime = environment type, session = episode handle, microVM = isolation boundary.
- **Isolation is a per-task choice:** persistent VM + fresh workdir for file-level tasks (this deck), fresh VM per episode for OS-level tasks (Terminal-Bench).
- **Lanes work on AWS:** 1:1:1 verified, 4 opens instead of 400, sandbox p95 431 → 120 ms, reward and eval unaffected.
- **Scale the bottleneck, not the tier you like:** math is inference-bound, so the next demo uses a sandbox-heavy task at 256 VMs to show where AgentCore's elastic CPU pays off.

## References

| Ref | Source | Supports |
|------|------------------------------------------------------------|--------------------------------------------|
| [Q1] | AgentCore quotas: <https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/bedrock-agentcore-limits.html> | Image ≤ 2 GB · 15-min idle timeout |
| [Q2] | `aws service-quotas list-service-quotas --service-code bedrock-agentcore` (us-west-2, 2026-09-23) | Active sessions 5000 (L-3E5722B2) · new sessions 25/s (L-8EE2AEA2) |
| [D3] | <https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-how-it-works.html> | Session = dedicated microVM; 15-min idle, 8 h max lifetime |
| Logs | `code/demos/_smoke_logs/lanes_tb_0923/scale_lanes.*`, `_smoke_logs/demo_cpu_harness_scale.*` | All measured numbers in §5 |

: References
