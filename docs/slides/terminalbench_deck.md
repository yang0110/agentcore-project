# Terminal-Bench on AgentCore

**Kaige Yang** — Senior Applied Scientist, AWS

September 24, 2026

---

## Summary

| | |
|--------------|------------------------------------------------------------------------------|
| **Goal** | RL-train Qwen3-4B (slime GRPO) on Terminal-Bench, every episode in its own AgentCore microVM |
| **Challenge** | Isolation needs a **fresh microVM per episode**, so every episode pays a **cold start**, in bursts, at every step |
| **Solution** | **Look-ahead pre-open:** replay slime's data order and boot the next 2 batches' sessions (P = 64) before slime asks; heartbeat keeps them alive |
| **Result** | Smoke run on AWS: 382 episodes, **0 infra errors**, `open_wait_ms` **p95 = 0** (no boot on the critical path). Cost: ≈ 95 % of microVM time idle; reward sparse (4 solves) |

: The deck in one table

Source: `code/demos/demo_cpu_terminalbench/`.

# 1 · Setup

## One image → one Runtime → one microVM per episode

![](figures/tb_fig01_cardinality.pdf){height=42%}

*Figure 1: image 1:1 Runtime 1:N session 1:1 microVM.*

- **Each Terminal-Bench task is its own environment** (OS, packages, a deliberately broken state), so each task gets its own arm64 image and its own AgentCore Runtime: **218 Runtimes**, routed by task id.
- **Session = microVM, 1:1.** A new `runtimeSessionId` boots a fresh Firecracker VM (2 vCPU, 8 GB) from the pristine image; `StopRuntimeSession` terminates and sanitizes it.
- The image is untouched except for one wrapper layer: a standalone Python + a stdlib HTTP server (`exec`, files, boot timestamps). Hidden tests are injected **after** the agent finishes.

## The dataset

| Stage | Tasks | Dropped because |
|----------------------------------|-----------|--------------------------------------------------------------|
| Terminal-Bench tasks | 229 | — |
| Deployable (arm64, ≤ 2 GB image) | **218** | 7 don't build for arm64 · 4 exceed the fixed 2 GB image cap |
| Oracle-usable | **166** | 47 reference solution fails on this substrate · 5 timed out · 0 pass untouched |
| Train / held-out eval | **135 / 31** | 80/20, disjoint by task, stratified by difficulty |

: From benchmark to training set

- **Oracle check:** keep a task only if the reference solution passes **and** the untouched environment fails; otherwise its reward is structurally 0 or 1.
- Images: p50 128 MB, p90 492 MB, max 2 GB (48 GB total). The heavy tail (ML models, TeX, toolchains) is what makes boot time vary.
- **Reward = 1 only if every hidden test passes.**

## How one episode runs

![](figures/tb_fig03_episode-workflow.pdf){height=56%}

*Figure 2: route → fresh session → agent loop → grade → stop.*

- There is no "create session" call: the first invoke with a new `runtimeSessionId` **is** the boot.
- One step = RB 4 prompts × NS 8 samples = **32 episodes = 32 fresh microVMs**; 4 harness workers run them.

# 2 · The challenge: cold start

## Why every episode pays a boot

**We can't reuse a warm VM.** Terminal-Bench tasks change the whole OS (apt, `/etc`, services, `/tmp`). Resetting a workdir would leak state into the next episode and corrupt the reward and the GRPO advantage. **The session is the isolation boundary**, so each episode needs a new one.

| Cost of a fresh session | Typical |
|------------------------------------------------------|-------------------------------------------|
| Cold boot: allocate VM, pull image, start server | **≈ 15 s** for a 447 MB image; grows with image size |
| Warm hit: AgentCore already had a spare VM up | ≈ 0.5 s |
| Per step, if on the critical path | 32 × 15 s ≈ **8 min** of serial boot |

: Cold-start cost

- **Bursty:** all 8 GRPO siblings of a task start at once, on the same Runtime.
- **Idle reclaim:** a session silent for ≈ 15 min is reclaimed, and the weight-sync pause can be that long.

## Why the trainer can't hide it on its own

- **slime only reveals a few episodes at a time:** at most `CONC × engines` = 8 `generate()` calls in flight.
- **It draws step N+1 only after step N finishes**, so every step's first episodes would wait for a boot.
- **slime knows nothing about sandboxes.** Tasks, Runtimes and sessions all live in our `TerminalBenchPool`, inside slime's rollout process.

| Option | Hides boot? | Keeps isolation? | Catch |
|--------------------------------------------|-------------|-----------------|--------------------------------------|
| Reuse one warm session per task + reset | ✓ | ✗ | OS state leaks → **rejected** |
| Standing warm pool per task | ✓ | ✓ | 218 tasks × NS idle VMs |
| Open on pickup | ✗ | ✓ | Boot on the critical path |
| Fully-async rollout | ✓ | ✓ | Off-policy (stale weights); slime mode change |
| **Look-ahead pre-open (used)** | ✓ | ✓ | Guesses can be wrong; booted VMs idle |

: Options

# 3 · The solution: look-ahead pre-open

## Four mechanisms

1. **Look-ahead data source.** `tb_lookahead_source.py` (a `--data-source-path` plugin) replays slime's deterministic order (sequential offset, `random.seed(seed + epoch)` shuffle) to list the **next 2 rollouts'** prompts, without moving slime's position. No slime change.
2. **Bounded pre-open window.** The pool keeps a fresh session booted for each of the next **P** = `TB_PREOPEN_BATCHES` × RB × NS = **64** upcoming episodes, pooled per task. A worker claims any ready session for its task; the claim moves the window on by one. Wrong guesses are stopped at the next draw.
3. **Heartbeat.** A `SessionKeeper` pings every open session at half the 15-min idle timeout, so none is reclaimed during training or weight sync.
4. **Retry on a new id.** A failed open retries on a **new** session (3×); an infra failure is logged as `infra_error`, never scored as a wrong answer.

Live microVMs ≤ W + P = 4 + 64 = **68**. The account quota is **5000** active sessions and 25 new/s [Q2]; throttled opens back off and retry.

## Across a step boundary

| Moment | What happens | Worker waits for a boot? |
|---------------------------------------------|------------------------------------------------------------------|--------------|
| Step N−1 running | Its draw already announced steps N and N+1 → 64 sessions booting | — |
| Step N drawn | Window = steps N, N+1: already booted, nothing new to open | — |
| Each claim | Next episode of step N+2 starts booting | **No** |
| Step N training | 64 booted VMs idle through training; heartbeat keeps them | — |
| Step N+1 drawn | All its episodes already booted | **No** |

: Look-ahead across a step boundary (W = 4, P = 64)

- **Sessions per Runtime are never configured.** A task sampled in a step gets NS = 8 sessions; others get 0.
- **The price is idle time:** each VM is booted about P · E / W seconds before it's needed (E = episode time).

## Sizing `TB_PREOPEN_BATCHES`

Boot is hidden when **P · E / W ≥ B** (B = boot time), and a window of at least one batch (P ≥ RB × NS) also covers each step's first episodes. Everything beyond that is paid idle time.

| `TB_PREOPEN_BATCHES` | P | Boot hidden? | Idle per VM ≈ P · E / W |
|--------------------------|------|------------------------------------|------------------------------|
| 0 — look-ahead off | ≈ 4 | Mid-step only; each step's first W wait | ≈ 20 s |
| 1 — one batch ahead | 32 | Yes, including step starts | ≈ 3 min |
| **2 — used in the smoke run** | **64** | Yes, with a full batch of slack | ≈ 6 min (measured: 7.3 min p50) |

: Choosing TB_PREOPEN_BATCHES

E = 21 s (measured episode p50), W = 4. Tune from the `[demo]` lines: `open_wait_ms` ≫ 0 → raise it; `preopen_idle_ms` ≫ boot time → lower it.

# 4 · Results

## Smoke run: learning (10 steps + eval, 2026-09-23)

| Metric | Value |
|--------------------------------|----------------------------------------------------------------------|
| Train solve rate, steps 1–10 | 0 · 0 · 0.088 · 0 · 0 · 0 · 0 · 0.026 · 0 · 0 |
| Full solves | **4 / 382** episodes (`acl-permissions-inheritance` ×3, `git-leak-recovery`) |
| Held-out solve rate (31 tasks × 2) | 0.0 |
| Partial progress | **42 %** of episodes pass ≥ 1 hidden test |
| Agent behaviour | Median 6 turns, 5 bash calls; 45 % hit the 8-turn budget |

: Learning (RB = 4, NS = 8, Qwen3-4B)

- **Sparse, not dead.** The tool loop works and the agent makes partial progress, but almost every GRPO group is all-0, so there is little gradient.

## Smoke run: sandbox and cost

| Metric | Value |
|------------------------------------|-------------------------------------------------------------|
| Episodes / infra errors | 382, each on its own fresh microVM / **0** lost (job ≈ 70 min) |
| Boot on the critical path (`open_wait_ms`) | **p50 0 · p95 0**; max 4.9 s (first 4 episodes only) |
| Session open (`open_ms`) | p50 0.66 s · p95 2.5 s · **100 % warm hits** · 0 retries |
| Episode time p50 / p90 | 21 s / 64 s (agent 9 s + grade 10 s) |
| Pre-opened VM idle p50 / max | **7.3 min** / 16.5 min; heartbeat kept all alive |
| microVM-seconds per train step | ≈ 13 000 (32 × 411 s); **≈ 95 % idle** |

: Sandbox and cost (W = 4, TB_PREOPEN_BATCHES = 2)

- **Every open was a warm hit.** AgentCore already had a spare VM up for each Runtime, so no ≈ 15 s cold boot occurred in this run. The per-image cold-start probe (`probe_coldstart.py`) is still to run.

## Takeaways and next steps

- **Works end to end:** per-task Runtimes, one fresh microVM per episode, oracle-checked tasks, 0 infra failures.
- **Cold start is solved, but over-provisioned.** Look-ahead removed all boot wait, but at P = 64 a VM idles ≈ 7 min for a 21 s episode. With warm hits at ≈ 0.7 s, **`TB_PREOPEN_BATCHES` = 1 (or 0)** should give the same wait at a fraction of the cost.
- **Learning signal is the bottleneck:** 4 solves in 382 episodes. Next: easy-first curriculum, more turns, or partial credit from `tests_passed`.

| Next | Why |
|-------------------------------------------|------------------------------------------------------|
| Re-run with `TB_PREOPEN_BATCHES` = 1 and 0 | Measure the cost/wait trade-off |
| `probe_coldstart.py` over all 218 Runtimes | Cold vs warm boot by image size (burst opens) |
| Curriculum / partial-credit reward | Mixed GRPO groups → real gradient |

: Next steps

## References

| Ref | Source | Supports |
|------|------------------------------------------------------------|--------------------------------------------|
| [Q1] | AgentCore quotas: <https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/bedrock-agentcore-limits.html> | 1000 Runtimes/account · image ≤ 2 GB · 15-min request timeout · 15-min idle timeout |
| [Q2] | `aws service-quotas list-service-quotas --service-code bedrock-agentcore` (us-west-2, 2026-09-23) | Active Session Workloads = 5000 (L-3E5722B2) · new sessions 25/s (L-8EE2AEA2) · image 2048 MB (L-0A9E32B3) |
| [D3] | <https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-how-it-works.html> | Session = dedicated microVM; 15-min idle, 8 h max lifetime |

: References
