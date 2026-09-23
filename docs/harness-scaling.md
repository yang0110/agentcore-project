# Scaling the harness tier: `throughput = min(inference, sandbox, harness)`

## The claim

An async-RL rollout is produced by **three replicated tiers**, and steady-state rollout throughput
is the minimum of their capacities:

```
throughput  =  min( inference , sandbox , harness )
```

Every other demo in this repo holds the **harness in-process and single** and studies the *task* and
the *sandbox*. That deliberately hides the third tier. [`demo_cpu_scale_lanes`](../code/demos/demo_cpu_scale_lanes/)
fixes the task (one, end-to-end — DeepScaleR math) and makes the **harness the scalable tier**, so
the third term of the `min(...)` becomes visible and tunable.

> **Why this is easy to miss.** Deepen slime's async buffer and scale inference (more SGLang engines)
> + sandbox (more Runtimes), and throughput still flat-lines if a **single process** drives every
> episode. The harness is bounded by its own asyncio concurrency, the event loop, file descriptors,
> and RAM for in-flight trajectories. It is a real tier; it just usually sits off-screen.

## Scaling a harness ≠ scaling a sandbox

The two are architecturally different, and conflating them is the trap:

- A **sandbox** is **ad-hoc per episode** — a fresh session holds one episode's files and is thrown
  away at the end. "More sandbox" = round-robin new sessions across more Runtimes.
- A **harness** is **persistent / warm** — it owns a live model-API connection (here a reused
  `aiohttp` session to the adapter) and the episode's trajectory state. "More harness" does **not**
  mean "spin one up per episode" (that would pay warm-up cost every episode and defeat the point). It
  means run a **fixed pool of W warm workers** and **route** episodes to them.

So the harness tier is `W = HARNESS_WORKERS` warm workers, defaulting to 4 to sit symmetric with the
demo's 4 inference engines and 4 microVMs — the balanced point of the `min(...)`.

> **The demo's default: 4 × 4 × 4 lanes (2026-09-23).** `demo_cpu_scale_lanes` (formerly
> `demo_cpu_harness_scale`) now makes that balance literal: `LANES=4` gives **4 warm workers : 4
> SGLang engines (TP1, one GPU each on rollout GPUs 4–7) : 4 persistent AgentCore microVMs**, strictly
> 1:1:1 — worker `w` pins its inference to engine `w` and drives all its episodes on its own sandbox
> session (opened once, kept for the run, on ARN `w % 2`, fresh `ep-<hex>` workdir per episode).
> `LANE_SANDBOX=0` restores the per-episode sticky/round-robin sandbox described below, and the
> per-request router LB below is what you get without the engine pin. `metrics.py`'s
> `check_stickiness` verifies the lanes are 1:1:1. Verified offline only so far (sweep + a
> fake-microVM lane test), not yet on AWS.

## The three tiers route and load-balance differently

This asymmetry is the substance of the demo. Each tier is replicated, but *how a prompt/episode
reaches a replica* differs by tier because their **state** differs:

| Tier | Replicas | State | Load-balancer | Granularity |
|---|---|---|---|---|
| **inference** | `E` SGLang engines | **stateless** per request | slime's SGLang **router**, cache-aware | per **request** |
| **sandbox** | `K` Runtimes | **stateful** (episode's files) | `make_sandbox` **round-robin**, one fresh session/episode | per **episode**, sticky |
| **harness** | `W` warm workers | **warm** (conn + trajectory) | shared `asyncio.Queue`, free worker **pulls** | per **episode**, by availability |

- **Inference is per-request LB.** Because an engine holds no episode state, the router is free to
  send each `/generate` call to the least-loaded / best-cache engine. A single episode's turns can
  span multiple engines. Finest granularity, most flexible.
- **Sandbox is per-episode sticky LB.** A sandbox *does* hold state (the files the agent wrote), so it
  must stay pinned for the episode's life. We balance by round-robining **which** sandbox a new
  episode opens on across the `K` ARNs (`_next_runtime_arn`). Coarser granularity, forced by state.
- **Harness is per-episode pull LB.** A worker is warm but holds no *cross-episode* state, so any free
  worker can take any episode — but only whole episodes (it owns the connection + trajectory for the
  run). We use a **pull** queue (work-stealing), not push/hash, because multi-turn agentic episodes
  have wide, variable latency: pull balances by **availability**, so a worker stuck on a slow episode
  simply doesn't pull more, while push would keep piling onto it.

### Admission vs. drive

slime admits up to `S = CONC × E` episodes concurrently (its `--sglang-server-concurrency`). But only
`W` of them are actively **driven by a harness worker** at a time; the rest block on the pull queue.
So `W` is the harness ceiling. `run.sh` defaults `CONC = ceil(W / E)` so `S ≈ W` (balanced start).
The sweep is: raise `W`, watch throughput climb ~linearly (harness-bound), then plateau once it hits
`min(inference, sandbox)`.

## Seeing it (offline, no AWS/GPU)

`harness_scaling_demo.py` drives the **real** `run_episode` on a `LocalSandbox` through `W` warm
asyncio workers, behind modelled inference/sandbox concurrency ceilings, and **measures** wall-clock
throughput + the service time each tier actually incurred. Example sweep:

```
  W     ep/s   solved       bind   ceilings ep/s [inf | sbx | harness]
----------------------------------------------------------------------
  1      5.0   24/24     harness       40 |    80 |     5.0
  2      9.9   24/24     harness       40 |    79 |     9.9
  4     19.6   24/24     harness       40 |    78 |    19.7
  8     33.6   24/24     harness       40 |    78 |    39.5

throughput rises ~linearly with W while the harness binds, then plateaus at ~34 ep/s once
inference binds (~40 ep/s).  => throughput = min(inference, sandbox, harness).
```

The plateau height is set by whichever of inference/sandbox is scarcer; raising `W` past that point
buys nothing — you must scale that tier too. That is the **joint-scaling** argument: the three tiers
must be provisioned together, or the smallest one caps the whole rollout.

## Placement and the honest CPU caveat

`demo_cpu_scale_lanes` places the `W` workers **in-process** (asyncio tasks in the trainer process),
with **one shared in-process `AnthropicAdapter`** on localhost — so there is no harness Runtime to
deploy and no public adapter port to open (unlike the removed two-Runtime variant; see
[`harness-sandbox-decoupling.md`](harness-sandbox-decoupling.md)).

In-process warm workers lift the **IO-concurrency** ceiling, which is the one that binds for agentic
rollouts: an episode spends almost all its wall-clock **awaiting** — LLM turns over HTTP to the
adapter, tool exec over HTTP to the sandbox Runtime — and asyncio overlaps those awaits across `W`
episodes efficiently.

They do **not** lift the **CPU** ceiling. The GIL still serialises the CPU-bound work (tokenisation,
trajectory flattening in `finish_session`). If profiling shows the harness is CPU-bound rather than
await-bound, the next step is a *different placement* — separate processes, or dedicated harness
Runtimes — not more asyncio workers. That is deliberately out of scope: this demo characterises the
IO-bound regime, where a warm in-process pool is the correct, cheap answer.

## Files

- [`code/demos/demo_cpu_scale_lanes/agentcore_slime_demo/harness_pool.py`](../code/demos/demo_cpu_scale_lanes/agentcore_slime_demo/harness_pool.py) — the warm worker pool (queue, workers, per-episode drive) + the routing model in its module docstring
- [`.../harness_pool_generate.py`](../code/demos/demo_cpu_scale_lanes/agentcore_slime_demo/harness_pool_generate.py) — the slime `--custom-generate-function-path` hook that hands each episode to the pool
- [`.../harness_scaling_demo.py`](../code/demos/demo_cpu_scale_lanes/harness_scaling_demo.py) — the offline, measured scaling-law demo above
- [`code/demos/demo_cpu_scale_lanes/README.md`](../code/demos/demo_cpu_scale_lanes/README.md) — run/reproduce, knobs, Dockerfile/deploy
