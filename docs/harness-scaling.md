# Scaling the harness tier: `throughput = min(inference, sandbox, harness)`

> **Historical (2026-09-30).** This doc was written for demos and code dirs that were removed on 2026-09-30; the only live demo is [`code/demos/demo_cpu_terminalbench`](../code/demos/demo_cpu_terminalbench/README.md).

## The claim

An async-RL rollout is produced by **three replicated tiers**, and steady-state rollout throughput
is the minimum of their capacities:

```
throughput  =  min( inference , sandbox , harness )
```

Two consequences shape every demo in this repo:

1. **The harness must scale along with SGLang (inference) and the sandbox.** Deepen slime's async
   buffer, add SGLang engines, add sandbox sessions, and throughput still flat-lines if a few
   coroutines drive every episode. The harness is a real tier; it just usually sits off-screen.
2. **The harness should scale with the sandbox, not as part of the trainer**, as in DeepSeek's DSec
   (Elastic Compute): move the agent loop out of the trainer. Here the mini_claude loop runs **inside
   the episode's own sandbox session** (`in_session_harness.py` → `session_runner.py`) and dials the
   trainer's per-engine `AnthropicAdapter` back, so N sessions carry N loops.

**The decoupled harness tier is the default rollout topology in every demo (2026-09-28).** It came
from `demo_cpu_scale_lanes` (retired into `demo_cpu_tasks SCALE=1`), the same way `PREOPEN` and `HEARTBEAT` were merged into all demos
earlier. Each demo lib has `agentcore_slime_demo/harness_pool.py` (its module docstring is the
authoritative description); the Terminal-Bench / SWE-bench pool subclasses it and swaps only the
sandbox tier. The one exception is the experimental `demo_cpu_branching`, which runs the real Claude
Code CLI inside its sandbox microVM. The 4:256:256 run that used to be that folder is now the
**`SCALE=1` preset of `demo_cpu_tasks`** (below).

## Scaling a harness ≠ scaling a sandbox

The two are architecturally different, and conflating them is the trap:

- A **sandbox** is **ad-hoc per episode**: a fresh session holds one episode's files and is thrown
  away at the end. "More sandbox" = more sessions, across more Runtimes.
- A **harness dispatcher** is **persistent / warm**: it owns a reused `aiohttp` session and drives the
  episode's in-session loop (starts `session_runner.py`, long-polls it, grades). "More harness" does
  **not** mean "spin one up per episode". It means a **fixed pool of W warm workers** that episodes
  are **routed** to. The loop itself lives in the session, so its CPU and memory scale with the
  sandbox tier.

## The three tiers, sized independently and joined per episode

No worker owns an engine or a session. Each episode is joined from three pools when it starts:

| Tier | Replicas | Default | State | Load-balancer | Granularity |
|---|---|---|---|---|---|
| **harness** | `W = HARNESS_WORKERS` dispatch workers | RB×NS, one rollout step (32 at RB=4 NS=8) | warm (HTTP conn) | shared `asyncio.Queue`, a free worker **pulls** | per **episode**, by availability |
| **sandbox** | the demo's own sandbox | a fresh session per episode (`PREOPEN` / `HEARTBEAT` apply as before) | **stateful** (the episode's files) | taken **before** the engine; `SANDBOX_SESSIONS=M>0` (opt-in, `demo_cpu_tasks` own-Runtime only): M persistent sessions per image on a free-list, a fresh `/workspace/ep-<hex>` per episode (removed on return), one episode per session at a time; `SANDBOX_OPEN_RATE` (default 20/s) paces opens | per **episode**, sticky |
| **inference** | `E` SGLang engines | least-loaded (fewest in-flight episodes, ties rotate) | stateless per request (prefix cache per engine) | picked once per episode | per **episode**: all its turns hit one engine |

- **Inference is picked once per episode.** An engine holds no episode state, so slime's router could
  balance every `/generate` call; the pool instead picks the least-loaded engine once and keeps it, so
  the growing prompt stays in one engine's prefix cache. This replaced the old per-episode hash
  pinning (`router_trace.pin_engine`, `crc32` → engine) in the tasks / GPU demos (and the since-removed instance_pool), and
  the Terminal-Bench / SWE-bench `HARNESS_WORKERS=4` workers pinned `w → w % E`.
- **The sandbox is sticky and taken first.** It holds the agent's files, so it stays with the episode
  for its life. Taking it before the engine means an episode waiting for a session is not counted as
  engine load. Pooled sessions (`SANDBOX_SESSIONS`) are not for tasks that change the OS
  (Terminal-Bench, SWE-bench keep a fresh session per episode).
- **The harness is pull, not push.** Multi-turn agentic episodes have wide, variable latency. Pull
  balances by **availability**: a worker stuck on a slow episode simply doesn't pull more, while push
  or hash would keep piling onto it.

The pool returns **one bare Sample** per episode (the longest of `finish_session`'s list), the shape
slime's fully-async driver needs. That driver is the only one (the `MODE` knob is gone); Terminal-Bench /
SWE-bench's look-ahead pre-open data source slides with its per-group draws.

### Admission vs. drive

slime admits up to `S = CONC × E` episodes concurrently (`--sglang-server-concurrency`). Only `W` of
them are driven by a worker at a time; the rest wait on the pull queue. So `W` is the harness ceiling.
`demo_cpu_tasks` keeps `CONC=4` by default and `SCALE=1` sets `CONC = ceil(W / E)` so `S = W`;
Terminal-Bench / SWE-bench default `CONC = ceil(2W / E)` (W running + W queued, so a free worker never
idles). On the GPU farms the farm's few slots bound concurrency, so those runs are sandbox-bound by
design.

### Which tier binds

Every episode's `[demo]` record carries the tier fields: `worker`, `qdepth` (> 0 = **harness**-bound:
episodes waiting for a free worker, raise `HARNESS_WORKERS`), `sandbox_wait_ms` (> 0 = **sandbox**-bound;
Terminal-Bench / SWE-bench report `open_wait_ms`), `engine`, `engine_inflight` (climbing =
**inference**-bound), plus `sandbox_slot` / `session_episode` when pooled. `metrics.py`'s
`check_stickiness` (every demo) reports sessions per worker, engines per worker and workers per
session from `<log>.router.csv` (which gains a `sandbox_slot` column). A worker using several engines
or sessions is the expected, decoupled shape, not a violation; the only violation is one episode's
ops seen on two Runtimes.

## `SCALE=1`: the harness-scaling run (was `demo_cpu_scale_lanes`)

```bash
SCALE=1 bash code/demos/demo_cpu_tasks/5_run.sh
```

turns all three tiers up together: `DATASET=deepscaler RB=32` (× NS=8 = 256 episodes per step),
`GPUS_PER_ENGINE=1` (E=4 TP1 engines on the 4 rollout GPUs), `HARNESS_WORKERS=256`, and
`CONC = ceil(256/E) = 64` → S = 256 episodes in flight, i.e. 256 sandbox sessions, each carrying its own
mini_claude loop. The ckpt gets the `_scale` suffix and the log tag `-scale`; any knob still overrides
the preset (e.g. `SANDBOX_SESSIONS=64` to run on 64 pooled sessions).

Results so far (from the retired `demo_cpu_scale_lanes` folder, i.e. the harness-scaling run, now
`demo_cpu_tasks SCALE=1`; logs in `code/demos/demo_cpu_tasks/logs/scale/`):

- **256-worker RL smokes**, 592 episodes each over 2 steps × 256: 2026-09-26
  (`smoke_all_20260926_142657`) `rollout/raw_reward` 0.37 → 0.39, `eval/deepscaler` 0.25; 2026-09-28
  (`smoke_all_20260928_031307`) 0.31 → 0.37, `eval/deepscaler` 0.27.
- **Pool check without a trainer (2026-09-25)**: the real `HarnessPool` against the K=2 CPU Runtimes,
  4 fake engines, W = M = 64, 192 episodes: 1280 sandbox ops all ok, every workdir fresh, episodes per
  engine 48 / 48 / 47 / 49; a worker used up to 2 sessions and p50 2 engines.
- **Earlier 4 × 4 × 4 "lanes" topology** (worker `w` : engine `w` : its own microVM; AWS smoke
  2026-09-23): reward 0.19–0.50, `eval/deepscaler` 0.358. Replaced by the decoupled pool 2026-09-25.

## Seeing it offline (no AWS/GPU)

`cd code/demos/demo_cpu_tasks && python3 harness_scaling_demo.py` (seconds; knobs `WORKERS`,
`EPISODES`, `CINF`, `CSBX`, `TINF`, `TSBX`; output in `logs/scale/harness_scaling_offline.log`) drives the
episode on a `LocalSandbox` through `W` warm asyncio workers, behind modelled inference/sandbox
concurrency ceilings, and **measures** wall-clock throughput + the service time each tier incurred:

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

(It is an offline model: it runs the loop in-process on a local sandbox, not in a session.) The
plateau height is set by whichever of inference/sandbox is scarcer; raising `W` past that point buys
nothing, and that tier must scale too. That is the **joint-scaling** argument.

## Placement and the honest trainer-side caveat

Only the dispatch tier (W asyncio tasks) and one `AnthropicAdapter` per engine stay in the trainer
process. Dispatch is IO (start the runner, long-poll, grade), which asyncio overlaps well across W
episodes. The agent loop's own CPU and memory are in the sessions, so they scale with the sandbox tier.

What the trainer still serializes under the GIL is the adapters' CPU-bound work: tokenization and
trajectory flattening in `finish_session`. At large W (256 in `SCALE=1`) that is the part to watch. If
profiling shows it binds, the next step is a different placement of the adapters (separate processes),
not more dispatch workers.

## Files

- `code/demos/<demo>/agentcore_slime_demo/harness_pool.py` (every demo lib; e.g.
  [`demo_cpu_terminalbench`](../code/demos/demo_cpu_terminalbench/agentcore_slime_demo/harness_pool.py)): the pool
  (queue, workers, least-loaded engine, opt-in pooled sessions, tier fields) + the model in its docstring.
- `.../in_session_harness.py` + `session_runner.py`: the loop in the session.
- `demo_cpu_tasks` (Terminal-Bench rows) / `demo_cpu_swebench` `terminalbench_pool.py`: `HarnessPool` subclass with
  the fresh-session, pre-open, strict-routing sandbox tier.
- [`code/demos/demo_cpu_terminalbench/run.sh`](../code/demos/demo_cpu_terminalbench/run.sh): the `HARNESS_WORKERS` / `GPUS_PER_ENGINE` knobs.
  The removed `demo_cpu_tasks/run.sh` had the `SCALE=1` block and the
  `HARNESS_WORKERS` / `SANDBOX_SESSIONS` / `SANDBOX_OPEN_RATE` / `GPUS_PER_ENGINE` knobs.
- `code/demos/demo_cpu_tasks/harness_scaling_demo.py` (removed 2026-09-30, git history):
  the offline, measured scaling-law sweep above; `code/demos/demo_cpu_tasks/logs/scale/`: the logs of
  the runs above (the `demo_cpu_scale_lanes` folder itself was removed 2026-09-28).
