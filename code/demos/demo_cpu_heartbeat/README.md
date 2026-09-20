# demo_cpu_heartbeat — keep the sandbox warm across the weight-sync barrier (REAL CPU trainer)

This is a **real slime CPU RL trainer** (async/decoupled GRPO, Qwen3-4B, 10 train steps + 1 held-out
eval + a single save-at-end — the same shape as `demo_cpu_mbpp`) whose one distinguishing feature
is a **keep-warm lever**: it holds its **remote AgentCore sandbox sessions warm across the live
weight-sync barrier** so none is idle-reclaimed mid-run.

**Why it matters.** In async/decoupled slime, generation **pauses** each step while the trainer syncs
updated weights into SGLang. During that window no `invoke_agent_runtime` calls flow, and AgentCore
reclaims a session purely on **absence of traffic** (15-min microVM idle timeout / ~600 s
GPU-instance idle timeout, reset on each invoke). A long enough sync barrier — or a slow rollout
tail — can therefore let an **in-flight** sandbox be idle-reclaimed mid-episode, losing its
workspace. The heartbeat closes that gap.

## The lever (a heartbeat that lives *outside* the harness)

The mechanism is `agentcore_slime_demo/session_keeper.py`, unchanged in spirit from the offline demo:

- **`SessionKeeper`** — one **process-wide** registry + a single background loop. It heartbeats any
  registered session that has gone idle past `ping_fraction × idle_timeout` with a cheap no-op
  (`exec "true"` — one `invoke_agent_runtime` round-trip), which resets AgentCore's idle clock. Real
  episode traffic refreshes the *same* clock, so outside a stall it fires **zero** heartbeats.
- **`KeptSandbox`** — the composition point. Wrapping any sandbox in it adds register / touch /
  heartbeat wiring **without the harness or the sandbox knowing weight sync exists**.

In this trainer the keeper runs in **continuous keep-warm mode** (`keep_warm_begin()`): it
heartbeats idle sessions without ever closing the gate, because a decoupled slime run gives the
harness no callback at its *internal* weight-sync barrier — but generation naturally goes silent
exactly during that barrier, so an always-on idle heartbeat is what actually holds sessions warm
across the stall. (The gate-based `sync_begin`/`sync_end` quiesce path stays available and is what
the offline `heartbeat_demo.py` drives explicitly.)

The wiring is a few lines in the vendored `agentcore_slime_demo/generate.py`: a lazily-started
process-wide `SessionKeeper` singleton, and — when `HEARTBEAT_KEEP_WARM` is truthy — every episode's
`make_sandbox(...)` result is wrapped in a `KeptSandbox`. With `HEARTBEAT_KEEP_WARM=0` the demo behaves
**exactly** like `demo_cpu_mbpp` (no wrap, no keeper), which is the natural A/B for measuring the
lever.

Heartbeat backstops **idle reclaim**, not an instance *recycle* (a GPU-instance recycle kills the
session regardless) — for that you need episode-level recovery (`recovery.resilient_episode`) and/or
application snapshot (see the sibling [`demo_gpu_snapshot`](../demo_gpu_snapshot/)).

## Remote-only (no local sandbox)

`SANDBOX=local` is rejected — a CPU training run must never silently fall back off the AgentCore
substrate it is meant to exercise. Use `agentcore-runtime` (default; the K=2 CPU Runtime pool) or
`agentcore` (Code-Interpreter microVM).

## Reproduce (inside the `agentcore_p4` container)

```bash
# 1. build data (deepscaler numeric-answer math; ships in data/, rebuild is idempotent)
docker exec agentcore_p4 bash /work/code/demos/demo_cpu_heartbeat/prepare_data.sh

# 2. REAL 10-step train + 1 eval through the REMOTE CPU sandbox, sessions held warm across sync
docker exec -e NR=11 -e EVAL_INTERVAL=10 -e EVAL_NS=2 agentcore_p4 \
    bash -lc 'cd /work && bash code/demos/demo_cpu_heartbeat/run.sh 2>&1 | tee code/demos/demo_cpu_heartbeat/smoke_run.log'

# 3. parse the log into CSVs + a curve PNG
docker exec agentcore_p4 python3 /work/code/demos/demo_cpu_heartbeat/metrics.py \
    /work/code/demos/demo_cpu_heartbeat/smoke_run.log --csv train.csv --eval-csv train.eval.csv --plot train.png
```

## Knobs (env vars; defaults in `run.sh`)

| Knob | Default | Meaning |
|---|---|---|
| `HEARTBEAT_KEEP_WARM` | `1` | `1` wrap every sandbox in `KeptSandbox`+`SessionKeeper`; `0` = plain (== `demo_cpu_mbpp`) |
| `HEARTBEAT_IDLE` | `600` | session idle timeout to defend (s); heartbeat fires at `ping_fraction × this` |
| `HEARTBEAT_PING_FRACTION` | `0.5` | heartbeat once a session is idle ≥ `fraction × idle_timeout` |
| `SANDBOX` | `agentcore-runtime` | remote sandbox: `agentcore-runtime` \| `agentcore`. **`local` is rejected.** |
| `FAMILY` | `deepscaler` | task family (pure-CPU RLVR checker) |
| `RB` / `NS` | 4 / 8 | GRPO prompts-per-step / group size (`GBS = RB×NS`) |
| `NR` | 11 | num-rollout ⇒ real steps `NR-1` (10 train + 1 eval) |
| `MODE` | `B` | `B` fully-async (deep buffer) \| `A` 1-step-async |
| `CONC` | 4 | sandbox/episode concurrency; episodes in flight `S = CONC × num_engines` |
| `EVAL_INTERVAL` / `EVAL_NS` | 10 / 2 | held-out eval cadence / samples |

## What to watch

- `rollout/raw_reward` and `eval/deepscaler` — that the run trains and evals like any CPU trainer.
- The keeper's log lines (`[keeper] …`) and the sandbox `[sbx-op] … op=exec command=true` heartbeat
  round-trips appearing **during the sync stalls** and nowhere else — evidence the lever fires only
  when a session is genuinely idle. Compare a `HEARTBEAT_KEEP_WARM=0` run (no heartbeats) as the A/B.
- `perf/sandbox_ms_*` vs `perf/sglang_gen_ms_*` — the remote round-trip split (`metrics.py`).

## The seconds-long offline illustration

`heartbeat_demo.py` (+ `agentcore_idle_sim.py`) plays the **naive → heartbeat → recover** arc in a few
seconds with **no AWS and no GPU** — the fastest way to *see* the mechanism before committing a real
run:

```bash
bash -c 'IDLE=4 PAUSE=10 python3 /work/code/demos/demo_cpu_heartbeat/heartbeat_demo.py'
```

ACT 1 = naive "stop invoking" (session idle-reclaimed, gone on resume); ACT 2 = heartbeat quiesce
(survives a pause far longer than the timeout); ACT 3 = recover-if-reclaimed (heartbeat is necessary
but not sufficient — a recycle still needs re-seeding). It drives the keeper's explicit
`sync_begin`/`sync_end` gate; the real trainer above uses the same keeper in continuous keep-warm
mode.

## Folder layout (self-contained)

- `agentcore_slime_demo/` — vendored library: the in-process harness (`generate.py` incl. the
  keep-warm wiring, `mini_agent.py`), the remote sandbox client (`agentcore_runtime_sandbox.py`),
  the keep-warm lever (`session_keeper.py`), the dataset builder (`build_real_datasets.py`), the
  task registry (`task_families.py`).
- `prepare_data.sh` (step 1) · `run.sh` (the real RL run, step 2) · `metrics.py` (step 3).
- `heartbeat_demo.py` / `agentcore_idle_sim.py` — the offline animation of the same mechanism.
- `data/` — `deepscaler_{train,eval}.jsonl`.
