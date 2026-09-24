# Scaling lanes on AgentCore

**Kaige Yang** — Senior Applied Scientist, AWS

September 24, 2026

---

## Summary

| | |
|--------------|------------------------------------------------------------------------------|
| **Goal** | Scale an agentic-RL rollout by scaling harness, inference and sandbox **together** |
| **Challenge** | Throughput = **min(inference, sandbox, harness)**; by default the three tiers are balanced independently |
| **Solution** | **Lanes:** 1 harness worker : 1 SGLang engine : 1 persistent AgentCore microVM |
| **Result** | Lanes verified on AWS; **4 session opens instead of 400**; sandbox p95 **431 → 120 ms**. Rollout time unchanged: the task is inference-bound |

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

## A session can be fresh per episode, or kept per lane

![](figures/lanes_fig04_session-use.pdf){width=86%}

*Figure 3: fresh per episode vs persistent per lane.*

- **Fresh** for OS-level tasks (Terminal-Bench). **Persistent + fresh workdir** for file-level tasks (this deck).
- A kept session must stay active: AgentCore reclaims it after **15 min idle** [D3].

# 2 · The challenge

## Throughput = min(inference, sandbox, harness)

![](figures/lanes_fig08_balancers.pdf){width=92%}

*Figure 4: by default each tier balances on its own.*

- **Harness** is a hidden ceiling: add engines and VMs and throughput still flat-lines.

# 3 · The solution: lanes

## One lane = one worker : one engine : one microVM

![](figures/lanes_fig01_lanes.pdf){width=90%}

*Figure 5: LANES = 4 → 4 × 4 × 4.*

- Workers **pull** whole episodes; slime admits 2W, so a worker never waits for work.

## One episode inside a lane

![](figures/lanes_fig05_episode.pdf){width=96%}

*Figure 6: every turn stays on one engine and one microVM.*

## The step pipeline: training hides behind the rollout

![](figures/lanes_fig06_pipeline.pdf){width=88%}

*Figure 7: slime's async driver overlaps train N with rollout N+1.*

- Weight sync waits for the rollout to finish, so rollouts are **one step off-policy**.

# 4 · Results

## Offline: tiers must scale together

![](figures/lanes_fig07_sweep.pdf){width=88%}

*Figure 8: harness alone plateaus; all three tiers scale near-linearly.*

## On AWS: lanes hold, 4 opens instead of 400

| Metric (smoke run, 2026-09-23) | Value |
|-----------------------------------|-----------------------------------------------------|
| Lanes 1 : 1 : 1 | ✓ all 400 episodes on their lane's engine **and** VM |
| Episodes per lane | 103 / 101 / 99 / 97 |
| Session opens | **4** for the whole run |
| Warm sandbox op p50 / p95 | **80 / 120 ms** |
| Reward (train) / eval | 0.36 / **0.358** |

: 4 × 4 × 4, 10 steps + eval, Qwen3-4B

## The sandbox got faster; the rollout didn't

![](figures/lanes_fig09_before-after.pdf){width=72%}

*Figure 9: AWS smoke, same task and harness (engines differ: 2 × TP2 vs 4 × TP1).*

- An episode takes ≈ 8 s; the sandbox part is ≈ **0.65 s**. **≈ 90 % is generation**: this task is inference-bound.

# 5 · Next: 256 sandboxes on 4 engines

## Pick a task where the sandbox does the work

The engines keep up only if **N · G / T ≤ C**: N sandboxes, G tokens per episode, T episode time, C ≈ 10k tok/s (4 engines, to measure).

![](figures/lanes_fig10_task-time.pdf){width=98%}

*Figure 10: DeepCoder: every check runs the full hidden suite.*

- **Probe** (1500 problems): 74 % pass with the gold solution; suite p50 **4.2 s**, 101 tests. Keep the **≥ 8 s** tail (13 %).

## Harness: 256 workers, 4 engines

![](figures/lanes_fig11_256.pdf){width=70%}

*Figure 11: 4 lanes × 64: one worker per microVM, 64 workers per engine.*

- **Proof:** sweep N = 32 → 256 at 4 engines; throughput rises while SGLang latency stays flat.

## Takeaways

- **AgentCore = the sandbox tier:** Runtime = environment, session = episode, microVM = isolation.
- **Lanes work:** 1 : 1 : 1 verified on AWS, 100× fewer opens, 3.6× lower sandbox p95.
- **Scale the bottleneck:** math is inference-bound; the 256-sandbox demo needs a sandbox-heavy task.

## References

| Ref | Source | Supports |
|------|------------------------------------------------------------|--------------------------------------------|
| [Q2] | `aws service-quotas list-service-quotas --service-code bedrock-agentcore` (us-west-2, 2026-09-23) | Active sessions 5000 · new sessions 25/s |
| [D3] | <https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-how-it-works.html> | Session = dedicated microVM; 15-min idle, 8 h max |
| Code | `demo_cpu_scale_lanes/`: `harness_pool.py`, `harness_scaling_demo.py` | Lanes, sweep |
| Logs | `_smoke_logs/lanes_tb_0923/`, `_smoke_logs/demo_cpu_harness_scale.*` | Measured numbers |

: References
