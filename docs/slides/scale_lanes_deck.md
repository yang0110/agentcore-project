# Scaling the rollout on AgentCore: three decoupled tiers

**Kaige Yang** — Senior Applied Scientist, AWS

September 25, 2026

---

## Summary

| | |
|--------------|------------------------------------------------------------------------------|
| **Goal** | Scale an agentic-RL rollout by scaling harness, inference and sandbox **independently** |
| **Challenge** | Throughput = **min(inference, sandbox, harness)**; by default the three tiers are balanced independently |
| **Solution** | **Decoupled tiers:** 4 SGLang engines : 256 in-process harness workers : 256 persistent AgentCore microVMs, joined per episode |
| **Result** | Pool verified live at 64 × 64 (1280 ops ok, engines 48/48/47/49, tiers mix freely). Earlier 4 × 4 × 4 lanes on AWS: **4 opens instead of 400**, sandbox p95 **431 → 120 ms**. [PENDING: 4 : 256 : 256 RL smoke] |

: The deck in one table

# 1 · AgentCore, seen from agentic RL

## AgentCore is the sandbox tier of the RL loop

![](figures/lanes_fig02_rl-loop.pdf){width=96%}

*Figure 1: harness, inference and sandbox — three tiers, one loop.*

- slime sees only `generate()` → a scored trajectory; all three tiers plug in through its custom generate hook.

## Three concepts: environment, episode, isolation

![](figures/lanes_fig03_concepts.pdf){width=100%}

*Figure 2: Runtime 1 : N sessions, session 1 : 1 microVM.*

- No "create session" call: the **first invoke** with a new id boots the VM. `StopRuntimeSession` ends it.

## A session can be fresh per episode, or kept and reused

![](figures/lanes_fig04_session-use.pdf){width=86%}

*Figure 3: fresh per episode vs persistent (then per lane, now pooled).*

- **Fresh** for OS-level tasks (Terminal-Bench). **Persistent + fresh workdir, one episode at a time** for file-level tasks (this deck).
- A kept session must stay active: AgentCore reclaims it after **15 min idle** [D3].

# 2 · The challenge

## Throughput = min(inference, sandbox, harness)

![](figures/lanes_fig08_balancers.pdf){width=92%}

*Figure 4: by default each tier balances on its own.*

- **Harness** is a hidden ceiling: add engines and VMs and throughput still flat-lines.

# 3 · The solution: decoupled tiers

## 4 engines : 256 workers : 256 microVMs, joined per episode

![](figures/lanes_fig12_decoupled.pdf){width=64%}

*Figure 5: each tier is its own pool; nothing is pinned.*

- Free worker → any free microVM (fresh `ep-<hex>/`) → least-loaded engine. `RB × NS` = **256 episodes/step**.

## Why decoupled: each tier is sized on its own

| Knob | Tier | It binds when |
|---------------------|----------------|------------------------------------------------|
| `HARNESS_WORKERS` (256) | harness | `qdepth` > 0: episodes wait for a worker |
| `SANDBOX_SESSIONS` (256) | sandbox | `sandbox_wait_ms` > 0: workers wait for a microVM |
| `ROLLOUT_GPUS` (4) | inference | `engine_inflight` and `sglang_gen_ms` climb |

: One knob per tier; the log names the tier that binds

- Harness and sandbox scale for **different reasons** (Python concurrency vs microVM quota and cost), so they are two numbers, not one.

## Isolation of a reused microVM

- One session serves **one episode at a time**; a free-list hands it out, and it comes back only after `rm -rf` of the episode's workdir.
- Different sessions are different microVMs (AgentCore's isolation boundary).
- Not reset between episodes of one session: `/tmp`, packages, stray processes. Safe for DeepScaleR (writes only its workdir); OS-level tasks need `SANDBOX_SESSIONS=0` (a fresh session per episode).

## Opening 256 microVMs, and the client limits

- All sessions open once at start, paced at **20/s** (quota 25 new/s [Q2]): ready in ≈ 13 s + one boot (≈ 5 s).
- Data-plane invokes: ≈ 100/s at 256 episodes vs a 1000/s quota. Active sessions: 256 vs 5000.
- Client side: boto3 `max_pool_connections` = 272 (default 10) and a 288-thread executor (default ≤ 32), or the harness process becomes the ceiling.

## The step pipeline: training hides behind the rollout

![](figures/lanes_fig06_pipeline.pdf){width=88%}

*Figure 6: slime's async driver overlaps train N with rollout N+1.*

- Weight sync waits for the rollout to finish, so rollouts are **one step off-policy**.

## DeepSeek DSec moved the agent loop out of the trainer [31]

- **Why:** DeepSeek's GPU jobs are routinely **preempted**. With the loop in the GPU pod, the loop died but the sandbox survived, so recovery **replayed a command log**.
- **Where:** from V4.1 on, the harness runs on DSec, **outside the GPU pool**: an agent sandbox (scaffold + tools) and a worker container (rollout state). A preempted trainer **reconnects**, with no replay.

| | DSec (V4.1+) | This demo |
|------------------|------------------------------|--------------------------------|
| Harness | agent sandbox, with its tools | **slime's rollout process** |
| Token record | worker container | slime's in-process adapter |
| Trainer lost | reconnect, continue | abort in-flight: one batch |
| Trainer stalls | pause/resume sandbox | **heartbeat** keeps sessions |

: Agent-loop placement, DSec vs this demo

## What we did, and why: the harness stays in-process

- **No preemption here:** the trainer has a dedicated 8×L40S box. A stall is the real risk, and the heartbeat covers it. A crash costs one batch, not correctness.
- **Moving only the harness saves nothing:** the token record and the SGLang engines die with the slime job anyway. DSec's gain also needs an external record and a "reconnect to episode X" path **in slime**.
- **Harness in each microVM has costs:**
  - the policy's commands would run next to its own transcript and reward, so it could tamper with them;
  - every microVM needs a network path back to inference (VPC mode plus a security-group rule);
  - it re-ties the harness tier to the sandbox tier, which is the coupling this demo removes.
- **When to move it:** preemptible/Spot GPUs or multi-hour episodes. Then run the harness plus its token record as its own service, and have slime fetch episodes by ID.

# 4 · Results

## Offline: tiers must scale together

![](figures/lanes_fig07_sweep.pdf){width=88%}

*Figure 7: harness alone plateaus; all three tiers scale near-linearly.*

## Live pool check: the tiers mix freely

| Metric (W = 64, M = 64, 192 episodes, fake inference, real Runtimes) | Value |
|-----------------------------------|-----------------------------------------------------|
| Sandbox ops | **1280**, all ok; 64 opens, 64 stops |
| Episodes per engine | 48 / 48 / 47 / 49 |
| Sessions per worker (max) · engines per worker (p50) · workers per session (max) | 2 · 2 · 2 |
| `open_ms` p50 · `sandbox_wait_ms` p50 / p95 | 5.45 s · 0 / 11.8 s (opening ramp only) |

: 2026-09-25; no trainer. [PENDING: 4 : 256 : 256 RL smoke]

## Earlier on AWS: 4 × 4 × 4 lanes, 4 opens instead of 400

| Metric (smoke run, 2026-09-23, old pinned-lane topology) | Value |
|-----------------------------------|-----------------------------------------------------|
| Lanes 1 : 1 : 1 | ✓ all 400 episodes on their lane's engine **and** VM |
| Episodes per lane | 103 / 101 / 99 / 97 |
| Session opens | **4** for the whole run |
| Warm sandbox op p50 / p95 | **80 / 120 ms** |
| Reward (train) / eval | 0.36 / **0.358** |

: 4 × 4 × 4, 10 steps + eval, Qwen3-4B

## The sandbox got faster; the rollout didn't

![](figures/lanes_fig09_before-after.pdf){width=72%}

*Figure 8: AWS smoke, same task and harness (engines differ: 2 × TP2 vs 4 × TP1).*

- An episode takes ≈ 8 s; the sandbox part is ≈ **0.65 s**. **≈ 90 % is generation**: this task is inference-bound.

# 5 · Next: a sandbox-heavy task for 256 microVMs

## Pick a task where the sandbox does the work

The engines keep up only if **N · G / T ≤ C**: N sandboxes, G tokens per episode, T episode time, C ≈ 10k tok/s (4 engines, to measure).

![](figures/lanes_fig10_task-time.pdf){width=98%}

*Figure 9: DeepCoder: every check runs the full hidden suite.*

- **Probe** (1500 problems): 74 % pass with the gold solution; suite p50 **4.2 s**, 101 tests. Keep the **≥ 8 s** tail (13 %).

## Takeaways

- **AgentCore = the sandbox tier:** Runtime = environment, session = episode, microVM = isolation.
- **Decouple the tiers:** harness workers, microVMs and engines are three pools joined per episode; persistent microVMs gave 100× fewer opens and 3.6× lower sandbox p95.
- **Scale the bottleneck:** math is inference-bound; the 256-sandbox demo needs a sandbox-heavy task.

## References

| Ref | Source | Supports |
|------|------------------------------------------------------------|--------------------------------------------|
| [Q2] | `aws service-quotas list-service-quotas --service-code bedrock-agentcore` (us-west-2, 2026-09-23) | Active sessions 5000 · new sessions 25/s |
| [D3] | <https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-how-it-works.html> | Session = dedicated microVM; 15-min idle, 8 h max |
| [31] | DSec (DeepSeek Elastic Compute) <https://arxiv.org/abs/2609.22978>, §6.2–6.3 | Agent loop outside the preemptible GPU pool; pause/resume |
| Code | `demo_cpu_scale_lanes/`: `harness_pool.py`, `harness_scaling_demo.py` | Decoupled pool, sweep |
| Logs | `_smoke_logs/lanes_tb_0923/`, `_smoke_logs/demo_cpu_harness_scale.*` | Measured numbers |

: References
