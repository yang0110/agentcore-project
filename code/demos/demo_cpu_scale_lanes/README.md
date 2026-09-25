# demo_cpu_scale_lanes — scaling the rollout on **three DECOUPLED tiers** (4 SGLang engines : 256 in-process harness workers : 256 AgentCore microVMs)

*(Formerly `demo_cpu_harness_scale`; until 2026-09-25 it ran 4 × 4 × 4 pinned "lanes", hence the folder
name.)* The async-RL rollout is produced by three replicated tiers, and this demo shows they **scale
independently**:

```
throughput  =  min( inference , sandbox , harness )
```

| Tier | Default | Knob | Where it runs |
|---|---:|---|---|
| **inference** | **E = 4** SGLang engines (TP1, rollout GPUs 4–7) | `ROLLOUT_GPUS` / `GPUS_PER_ENGINE` | the training box |
| **harness** | **W = 256** warm asyncio workers (`MiniClaudeHarness`) | `HARNESS_WORKERS` | **in the trainer process** |
| **sandbox** | **M = 256** persistent AgentCore Runtime sessions (microVMs) on K=2 ARNs | `SANDBOX_SESSIONS` | remote, AgentCore |

No tier is tied to another. A worker is not bound to an engine, and a microVM is not bound to a
worker: **each episode is joined from three independent pools** when it starts.

```
 slime ──► shared episode queue ──► any FREE harness worker (W=256, pull / work-stealing)
                                        │
                                        ├─► any FREE microVM session (M=256, asyncio.Queue of free slots)
                                        │     fresh workdir /workspace/ep-<hex>, rm -rf on return
                                        │
                                        └─► LEAST-LOADED SGLang engine (E=4), sticky for the episode's turns
```

So 256 workers and 256 microVMs are **two numbers, not one**. Set `HARNESS_WORKERS=256
SANDBOX_SESSIONS=64` and the sandbox tier binds (`sandbox_wait_ms` > 0); set `HARNESS_WORKERS=64
SANDBOX_SESSIONS=256` and the harness tier binds (`qdepth` > 0, 192 idle microVMs); drop `ROLLOUT_GPUS`
to 2 and inference binds (`engine_inflight` climbs, `perf/sglang_gen_ms_*` grows). The default 4 : 256 :
256 is the balanced point for a 256-episode step.

The task is fixed on purpose (one, end-to-end: **DeepScaleR** math), so only the tiers move. The other
demos keep the harness in-process and small and vary the *task* and the *sandbox*; this one makes the
**harness a scalable tier of its own**. If you deepen the async buffer and scale inference and sandbox
but keep a few coroutines driving every episode, the harness silently becomes the ceiling (its
concurrency, the asyncio loop, file descriptors, RAM for in-flight trajectories).

The harness is the **"mimic Claude Code" `MiniClaudeHarness`**: 3 tools (`write_file` / `bash` /
`read_file`), speaking the **Anthropic Messages API** to slime's in-process `AnthropicAdapter` (one
adapter per engine), with the tools running in the remote CPU sandbox Runtime. Reward is **RLVR**: a
hidden `run_checks.py` numeric-asserts the answer (tol 1e-4); no LLM judge.

- **Task:** DeepScaleR math, coding-shape (`solution.py` sets a module-level `answer`).
- **Dataset:** `agentica-org/DeepScaleR-Preview-Dataset`: `data/deepscaler_train.jsonl` (400) +
  `data/deepscaler_eval.jsonl` (40).
- **Placement:** in-process asyncio worker pool (see the GIL caveat below).

---

## How each episode is routed (this is the point of the demo)

| Tier | Replicas | State | Who load-balances | Granularity |
|---|---|---|---|---|
| **harness** | `W` warm workers | stateless but **warm** (shares one `aiohttp` session, `limit=0`) | a shared `asyncio.Queue`; whichever worker is **free pulls** the next episode | per **episode**, by availability |
| **sandbox** | `M` persistent sessions, slot `s` on `ARN[s % K]` | **stateful** per episode: `solution.py`, `run_checks.py` in a fresh `/workspace/ep-<hex>` | a second `asyncio.Queue` of **free sessions**; the episode borrows any one and returns it | per **episode**, by availability |
| **inference** | `E` SGLang engines | stateless per request (KV prefix cache per engine) | the episode picks the engine with the **fewest in-flight episodes** (ties rotate) and its adapter session stays on it | per **episode**: all of its turns hit one engine |

Order inside an episode: the worker (1) takes a free sandbox session first, then (2) picks the
least-loaded engine, so an episode never holds an engine slot while it waits for a microVM.

**Why pull, not push.** Multi-turn agentic episodes have wide, variable latency. A push or hash
assignment would hot-spot a worker (or a microVM) stuck on a slow episode while others idle. Pulling
from a free-list balances by availability, which is robust to that variance.

**Admission.** The stock driver (`MODE=A`) keeps one batch in flight, so a step is `RB × NS` episodes;
the default `RB=32, NS=8` gives **256**, enough to keep all 256 workers busy. slime admits up to
`S = CONC × E` episodes at once; `run.sh` defaults `CONC = ceil(W / E)` = 64, so `S = W`.

**Opening 256 microVMs.** The pool opens all `M` sessions when the first episode arrives, paced at
`SANDBOX_OPEN_RATE` = 20/s (the account limit is 25 new sessions/s, L-8EE2AEA2), so the pool is
ready in about 13 s plus one boot (~5 s). Every open logs `[demo] session_open=1 sandbox_slot=…
open_ms=…`, then `[demo] session_pool_ready=N of=M`. Episodes start as soon as the first sessions are
up. A session that fails to open shrinks the pool instead of failing the run. All sessions are
stopped at process exit. `SANDBOX_SESSIONS=0` switches back to one fresh session per episode.

**Client-side sizing** (so the harness process itself is not the silent ceiling): one thread-safe
boto3 client with `max_pool_connections = max(W, M) + 16` (the botocore default of 10 would cap
sandbox ops at 10), and the event loop's default executor enlarged to `max(W, M) + 32` threads (the
stock `asyncio.to_thread` pool caps at `min(32, cpus + 4)`).

**Quotas (us-west-2 defaults):** new Runtime sessions 25/s (L-8EE2AEA2) vs the 20/s open ramp; Runtime
data-plane invokes 1000/s (L-46ED137C) vs about 256 episodes × ~8 ops / ~20 s ≈ 100/s; Active Session
Workloads 5000 vs 256.

### Isolation of a reused microVM

One microVM session serves many episodes over the run, **one episode at a time**:

- A session is on the free list or checked out by exactly one episode; never two at once.
- Each episode gets a brand-new workdir `/workspace/ep-<16 hex>`. Its files are written there, the
  checker runs there, and the directory is `rm -rf`'d before the session goes back on the free list.
- Different sessions are different microVMs (AgentCore's isolation boundary); nothing is shared
  between them.

What is **not** reset between the episodes of one session: the rest of the microVM's filesystem,
`/tmp`, installed packages, and any process left running. That is safe here because a DeepScaleR
episode only writes `solution.py` in its own workdir and the checker only reads that workdir. **Tasks
that change the whole OS must use a fresh session per episode** (`SANDBOX_SESSIONS=0`, or see
`demo_cpu_terminalbench`). For per-episode uid isolation on a shared instance, see
`demo_cpu_instance_pool`.

If a session is idle-reclaimed (15 min), the next invoke on the same session id boots a new microVM
and the per-episode workdir is recreated on demand. `HEARTBEAT=1` keeps pooled sessions warm across the
weight-sync stall.

Full model + code: [`agentcore_slime_demo/harness_pool.py`](agentcore_slime_demo/harness_pool.py) and
[`../../../docs/harness-scaling.md`](../../../docs/harness-scaling.md).

### Honest ceiling caveat (in-process placement)

In-process asyncio workers lift the **IO-concurrency** ceiling only. Episodes spend almost all
wall-clock **awaiting** (LLM turns over HTTP to the adapter, tool exec over HTTP to the sandbox
Runtime), which asyncio overlaps well. They do **not** lift the **CPU** ceiling: the GIL still
serializes the CPU-bound parts (tokenization and trajectory flattening in the adapter's
`finish_session`). At W=256 that is the part to watch. Lifting it needs separate processes or
dedicated harness Runtimes, a different placement that is out of scope here.

### Validated without a trainer (2026-09-25)

A live pool check (a throwaway script, not committed) ran the real `HarnessPool` (opener, workers, free-session queue, `submit`) against the
real K=2 CPU Runtimes, with 4 fake engines and a fake harness that asserts every workdir is fresh.
W=64, M=64, 192 episodes:

- 1280 sandbox ops, all ok; 64 sessions opened and 64 `stop_session` ok; every workdir fresh.
- Episodes per engine 48 / 48 / 47 / 49.
- Decoupled: a worker used up to 2 sessions and p50 2 engines; a session served up to 2 workers.
- `open_ms` p50 5.45 s; `sandbox_wait_ms` p50 0, p95 11.8 s (only during the opening ramp).

[PENDING: live 4 : 256 : 256 RL smoke; the last AWS smoke (2026-09-23, reward 0.19–0.50, eval/deepscaler
0.358) ran the old 4 × 4 × 4 lane topology.]

---

## Step by step (numbered scripts; run from this dir on the HOST — container steps `docker exec` themselves)

```bash
bash 1_prepare_data.sh     # 1. container: deepscaler 400/40 (ships in data/; re-pulls the HF rows)
bash 2_build_image.sh      # 2. arm64 image from deploy/Dockerfile.sandbox.cpu
bash 3_push_ecr.sh         # 3. -> ECR acs-scalelanes-cpu (deploy/images.env)
bash 4_deploy_runtime.sh   # 4. Runtime acs_scalelanes_cpu -> deploy/runtime.env (AGENTCORE_SANDBOX_RUNTIME_ARNS)
bash 5_run.sh              # 5. container: 10 train + 1 eval, 4 engines : 256 workers : 256 microVMs; logs/<tag>.{log,csv,eval.csv,png}
bash 6_cleanup.sh          # 6. delete the Runtime
```

Without step 4, `5_run.sh` uses the baked K=2 `acs_sandbox_cpu` ARNs (pooled slot `s` on `ARN[s % K]`).
`HARNESS_WORKERS`, `SANDBOX_SESSIONS` and `ROLLOUT_GPUS` each move one tier alone.

The default needs 8 GPUs (4 actor + 4 engines). On a shared box, pin a subset and shrink the actor:
`SLIME_CUDA_VISIBLE=2,3,4,5,6,7 SLIME_NUM_GPUS=6 ACTOR_GPUS=2 ROLLOUT_GPUS=4 bash 5_run.sh`.
Stock driver only (`MODE=A`, pinned): the pool's fan-out LIST of Samples is incompatible with the
fully-async Mode B sort key, so `run.sh` rejects `MODE=B`.

**Offline scaling law (no AWS, no GPU)**, run in the container:
```bash
python3 harness_scaling_demo.py                            # sweep W: knee + plateau at min(inference, sandbox)
TIERS=1,2,4,8 EPISODES=24 python3 harness_scaling_demo.py  # balanced joint sweep: every tier = N, ~linear
REAL_SANDBOX=1 SANDBOX=agentcore-runtime AGENTCORE_SANDBOX_RUNTIME_ARN=<arn> \
  EPISODES=8 WORKERS=1,2,4 python3 harness_scaling_demo.py # sandbox tier = measured remote round-trip
```
The first sweep drives the real `run_episode` on a `LocalSandbox` through `W` warm asyncio workers
behind modelled inference/sandbox ceilings and prints which tier binds at each `W`. `REAL_SANDBOX=1`
opens one live remote sandbox **per warm worker** (reused across its episodes) and round-robins the
workers across the ARNs, so the `sandbox` ceiling is measured instead of the modelled `--tsbx`
(`--csbx` still applies). Keep `EPISODES` small — each episode is a real network call.

> **Adapter/tool-parser requirement (the A3/B3 dead-reward fix):** `run.sh` bakes in
> `--sglang-tool-call-parser qwen25 --sglang-reasoning-parser qwen3`. Without them the adapter cannot
> turn Qwen3's raw tool syntax into `tool_use` blocks, the mini-harness tool loop is dead, and reward
> is structurally 0. Do not remove them.

## Knobs (env vars; defaults in `run.sh`)

| Knob | Default | Tier / meaning |
|---|---|---|
| `HARNESS_WORKERS` | 256 | **W**, warm in-process harness workers |
| `SANDBOX_SESSIONS` | 256 | **M**, persistent microVM sessions shared by all workers; `0` = fresh session per episode |
| `SANDBOX_OPEN_RATE` | 20 | new sessions per second while the pool opens (quota 25/s) |
| `ROLLOUT_GPUS` / `GPUS_PER_ENGINE` | 4 / 1 | ⇒ **E** SGLang engines (4) |
| `ACTOR_GPUS` | 4 | train GPUs (TP2, so even) |
| `AGENTCORE_SANDBOX_RUNTIME_ARNS` | 2 pinned CPU ARNs | **K** sandbox Runtimes; pooled slot `s` lives on `ARN[s % K]` |
| `SLIME_CUDA_VISIBLE` / `SLIME_NUM_GPUS` | unset / 8 | pin ray to a GPU subset on a shared box |
| `CONC` | `ceil(W/E)` | episode-admission concurrency; `S = CONC × E` (default `W`) |
| `MODE` | `A` (pinned) | stock async driver only; `B` (fully-async) is **rejected** — the pool returns a fan-out LIST of Samples the Mode-B sort key can't order |
| `NR` | 11 | num-rollout ⇒ real steps `NR-1` (10 train + 1 eval); `NR×RB > 400` wraps into a second epoch |
| `RB` / `NS` | 32 / 8 | GRPO prompts-per-step / group size; `RB×NS` = 256 episodes per step keeps W=256 busy |
| `EVAL_INTERVAL` / `EVAL_NS` | 10 / 2 | held-out eval cadence / samples |
| `EPISODE_MAX_TURNS` | 4 | mini-harness turn budget per episode |
| `TEMP` | 1.0 | rollout sampling temperature |
| `SANDBOX` | `agentcore-runtime` | `agentcore-runtime` \| `agentcore` (Code-Interpreter). **`local` is rejected** (remote-only) |
| `HEARTBEAT` / `HEARTBEAT_IDLE` / `HEARTBEAT_PING_FRACTION` | 0 / 600 / 0.5 | `1` = a `SessionKeeper` pings each pooled microVM (and per-episode sessions) once idle past `fraction × idle` s, so they survive the weight-sync stall (was `demo_cpu_heartbeat`) |

## Customize the sandbox runtime build (Dockerfile)

The sandbox image is built from [`deploy/Dockerfile.sandbox.cpu`](deploy/Dockerfile.sandbox.cpu) —
the same CPU sandbox server (`sandbox_server_agent.py`, a `BedrockAgentCoreApp` exposing
`/invocations` write_file/exec/read_file) the other CPU demos use. Edit it, then rebuild and redeploy
with the step scripts (`FORCE_RECREATE=1` replaces an existing Runtime):

```bash
bash 2_build_image.sh && bash 3_push_ecr.sh && FORCE_RECREATE=1 bash 4_deploy_runtime.sh
```

The adapter and the harness workers run **in the trainer process** (localhost), so — unlike the
removed two-Runtime variant — there is **no** harness Runtime to build and no public adapter port /
security-group to open. Only the *sandbox* Runtime is remote.

## Metrics → CSV + PNG

`5_run.sh` runs `metrics.py` on its log and writes `logs/<tag>.{csv,eval.csv,png}`.

Watch:

- `rollout/raw_reward` (learning signal), `eval/deepscaler` (held-out solve rate), and the
  `perf/sandbox_ms_*` vs `perf/sglang_gen_ms_*` split.
- Per episode: `[demo] worker=… engine=… engine_inflight=… sandbox_session=… sandbox_slot=…
  session_episode=… sandbox_wait_ms=… qdepth=…`. Which tier binds:
  - `sandbox_wait_ms` > 0 after the opening ramp: the **sandbox** tier (raise `SANDBOX_SESSIONS`).
  - `qdepth` > 0: the **harness** tier (episodes waiting for a free worker; raise `HARNESS_WORKERS`).
  - high `engine_inflight` and growing `perf/sglang_gen_ms_*`: **inference** (raise `ROLLOUT_GPUS`).
- The **decoupling report** (`check_stickiness`, from `router.csv`): episodes per engine, and min/p50/max
  of sessions per worker, engines per worker and workers per session. Values above 1 show the tiers
  mixing freely. The only violation it flags is one session spanning two Runtimes.
- Finished samples carry `harness_worker`, `engine` and `sandbox_slot` metadata.

## Next: DeepCoder for the 256-sandbox run (data prep only)

The 256-sandbox run (the default M=256) is most informative with a task whose *check* is
expensive enough to load the sandbox tier, which DeepScaleR's single numeric assert is not.
[`agentcore_slime_demo/build_deepcoder.py`](agentcore_slime_demo/build_deepcoder.py) prepares one from
DeepCoder (`agentica-org/DeepCoder-Preview-Dataset`, `primeintellect` subset): it runs a gold Python
solution over every hidden stdin/stdout test and keeps only problems that pass, recording how long the
full suite takes.

```bash
docker exec agentcore_p4 bash -lc 'cd /work/code/demos/demo_cpu_scale_lanes && \
    python3 agentcore_slime_demo/build_deepcoder.py --probe 1500'          # -> data/deepcoder_probe.csv
docker exec agentcore_p4 bash -lc 'cd /work/code/demos/demo_cpu_scale_lanes && \
    python3 agentcore_slime_demo/build_deepcoder.py --train 512 --eval 64 --min-suite-s 8'
```

`data/deepcoder_probe.csv` is the committed 1500-problem probe: 74 % pass with the gold solution,
gold suite p50 4.2 s, median 101 tests per problem; the run keeps the ≥ 8 s tail (13 % of passing problems). **Not wired into
training yet:** the rows carry `family: "deepcoder"`, but no `DeepCoderTask` checker exists in the
vendored `task.py`, so `FAMILY=deepcoder` does not run until that lands.

## Folder layout (self-contained)

- `agentcore_slime_demo/` — vendored shared library; **this demo adds** `harness_pool.py` (the warm
  workers, the free-session pool and the engine picker) and `harness_pool_generate.py` (the slime `--custom-generate-function-path`
  hook). `mini_agent.py` is the shared `MiniClaudeHarness` with one added `http=` param so the workers
  share one warm `aiohttp` session.
- `harness_scaling_demo.py` — the **offline** scaling-law demo (no AWS/GPU)
- `deploy/Dockerfile.sandbox.cpu` — the CPU sandbox Runtime image
- `data/` — `deepscaler_train.jsonl` / `deepscaler_eval.jsonl`; `deepcoder_probe.csv` (DeepCoder probe, see above)
- `1_…6_*.sh` + `deploy/` (`demo.env`, `steps_common.sh`, `agentcore_deploy.py`) — the numbered steps
- `prepare_data.sh` — build the dataset (step 1) · `run.sh` — the RL run (step 5) · `metrics.py`

Shared by reference (NOT vendored): the base Megatron checkpoint
(`code/p4_slime_train/ckpt/Qwen3-4B_torch_dist`, override `CKPT_BASE`) and the slime install.
