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

So the harness tier is `W = HARNESS_WORKERS` warm workers (default **256**), sized **independently**
of the demo's 4 inference engines and its `M = SANDBOX_SESSIONS` = 256 persistent microVMs.

> **The demo's default: decoupled tiers, 4 : 256 : 256 (2026-09-25).** `demo_cpu_scale_lanes` runs
> **4 SGLang engines (TP1, rollout GPUs 4–7) : 256 in-process harness workers : 256 persistent
> AgentCore microVM sessions**, and none of them is pinned to another. Each episode is joined from
> three pools when it starts: a free worker pulls it, borrows **any free microVM** from a free-list
> (fresh `/workspace/ep-<hex>` workdir, `rm -rf` on return, one episode per session at a time), and
> picks the **least-loaded engine**, which then serves all of that episode's turns. `SANDBOX_SESSIONS=0`
> falls back to a fresh session per episode. `metrics.py`'s `check_stickiness` reports how freely the
> tiers mixed (sessions per worker, engines per worker, workers per session). Until 2026-09-25 the
> demo pinned 4 × 4 × 4 "lanes" (worker `w` : engine `w` : its own microVM); that topology passed an AWS
> smoke on 2026-09-23. The decoupled pool was verified live at W = M = 64 without a trainer (1280 ops
> ok, engines 48/48/47/49). [PENDING: 4 : 256 : 256 RL smoke.]

## The three tiers route and load-balance differently

This asymmetry is the substance of the demo. Each tier is replicated, but *how a prompt/episode
reaches a replica* differs by tier because their **state** differs:

| Tier | Replicas | State | Load-balancer | Granularity |
|---|---|---|---|---|
| **inference** | `E` SGLang engines | **stateless** per request (prefix cache per engine) | the episode picks the **least-loaded** engine (ties rotate); without it, slime's router balances per request | per **episode** (per request via the router) |
| **sandbox** | `M` pooled sessions on `K` Runtimes | **stateful** (episode's files) | a **free-list** of sessions; `SANDBOX_SESSIONS=0`: `make_sandbox` round-robin, fresh session/episode | per **episode**, sticky |
| **harness** | `W` warm workers | **warm** (conn + trajectory) | shared `asyncio.Queue`, free worker **pulls** | per **episode**, by availability |

- **Inference can be per-request LB.** Because an engine holds no episode state, slime's router is
  free to send each `/generate` call to the least-loaded / best-cache engine, so one episode's turns
  can span engines. The demo instead picks the least-loaded engine once per episode and keeps it,
  so the growing prompt stays in one engine's prefix cache.
- **Sandbox is per-episode sticky LB.** A sandbox *does* hold state (the files the agent wrote), so it
  must stay pinned for the episode's life. With the pool, an episode borrows any free session and
  returns it when done; slot `s` lives on `ARN[s % K]`. Without the pool, each episode opens a fresh
  session round-robined across the `K` ARNs (`_next_runtime_arn`). Coarser granularity, forced by state.
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
