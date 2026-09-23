# demo_cpu_terminalbench — RL over a LARGE image pool (one image per task), one isolated microVM per episode

Terminal-Bench ships each task as its **own container environment** (a task-specific `Dockerfile` +
a hidden pytest oracle). This demo turns that into a **large image pool**: one container image → one
AgentCore Runtime **per task** (~229 single-container tasks), and every rollout episode is routed to
**its task's own Runtime**. Where `demo_cpu_imagepool` had 3 images routed by *family*, this is the
same routing mechanism at **task scale** — hundreds of images, one per task.

The policy is Qwen3-4B trained by slime (async/decoupled GRPO). The agent is the faithful
**tool-calling** loop (`MiniClaudeHarness` → `AnthropicAdapter` → the policy under training; tools run
in the sandbox), so it exercises the adapter / tool-parser path (needs `--sglang-tool-call-parser
qwen25 --sglang-reasoning-parser qwen3`). The reward is the task's **own hidden pytest oracle** run by
`grade.sh` (RLVR — no LLM judge). **REMOTE-ONLY** (`SANDBOX=local` is rejected). Fully self-contained:
owns its `data/`, `ckpt/`, `run.sh`, a vendored `agentcore_slime_demo/`, its `fetch`/`deploy` scripts.

---

## The problem this demo solves: isolation vs. cold start

Terminal-Bench tasks act on the **whole OS** — they install packages, edit `/etc`, start services,
write outside the workdir. So episodes must be **isolated from each other at the machine level**: a
shared microVM with a reset working folder would leak processes, `/tmp`, packages and config from
one episode (or one GRPO sibling) into the next. On AgentCore the isolation boundary is the
**session**: one `runtimeSessionId` = one dedicated microVM, and when the session is stopped the
microVM is terminated and its memory sanitized. So **every episode gets its own brand-new session,
never reused**.

The cost is a **cold start per episode**: AgentCore allocates a microVM, **pulls that task's image
from ECR**, and boots it before any work happens. The demo hides that cost instead of avoiding it.

### Fresh session per episode, pre-opened (the design)

1. **Strict routing.** The episode's `task.image` (the task name) is looked up in the pool map
   `AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE`. An unmapped task is an **error** — it is never sent to a
   fallback Runtime (that would run it on another task's image).
2. **Cold start before the episode is even drawn.** slime only hands the pool `CONC × engines`
   episodes of the current batch and does not draw the next batch until this rollout ends, so the
   pool alone can't look far enough ahead. The look-ahead data source (`tb_lookahead_source.py`,
   wired with `--data-source-path`) replays slime's deterministic prompt order and, on every draw,
   tells the pool the episodes of the **next `TB_PREOPEN_BATCHES` rollouts** too. The pool keeps
   sessions booted for the first `TB_PREOPEN_BATCHES × RB × NS` upcoming episodes (default 2 batches =
   2 × `RB` × `NS` = 64), pooled per task: a worker takes any ready session on its episode's task
   Runtime, and the next upcoming episode starts booting (`TerminalBenchPool._refill`).
   `[demo] open_wait_ms` is the part of the cold start that was still on the critical path.
3. **Stop on exit.** When the episode ends, `StopRuntimeSession` tears the microVM down immediately
   (frees the quota slot; nothing carries over).
4. **Heartbeat across the weight-sync stall.** In async/decoupled slime, generation pauses while the
   trainer syncs weights into SGLang; an episode mid-turn then sends no invokes and AgentCore would
   idle-reclaim its microVM (~15-min timeout). Every open session — pre-opened or running — is
   registered with a `SessionKeeper`, which pings any session idle past
   `WARM_PING_FRACTION × WARM_IDLE_TIMEOUT`.

The agent works directly in the task's baked workdir (`/app`); the hidden oracle is baked at
`/opt/acs/oracle`, outside it, and only injected by `grade.sh` after the agent's turns.

---

## How an episode flows

```
slime → terminalbench_pool_generate.generate → TerminalBenchPool.submit
  worker pulls episode → route by task.image → _runtime_arn_for_image → task's Runtime ARN
  (at submit) open a BRAND-NEW session on that ARN while the episode waits   ← its own microVM
    SessionKeeper heartbeats it until it is stopped
      write PROBLEM.md (the instruction) into the task's workdir (/app)
      MiniClaudeHarness loop: LLM via adapter (policy under training), bash/write_file/read_file tools
    grade: bash /opt/acs/oracle/grade.sh  (injects hidden tests + runs run-tests.sh → exit 0 = solved)
  finish_session → loss-masked GRPO samples (list); adapter.drop_session; StopRuntimeSession
```

Only single-container tasks are poolable (one image ↔ one sandbox); `fetch_terminalbench.py`
auto-skips the multi-container tasks (a task whose `docker-compose.yaml` declares extra services).

## Reproduce (inside the `agentcore_p4` container unless noted)

```bash
D=code/demos/demo_cpu_terminalbench

# 0. Fetch + classify the Terminal-Bench corpus into tasks/ + tasks/manifest.json (no AWS/Docker).
python3 $D/fetch_terminalbench.py                 # all single-container tasks (~229); TB_LIMIT=5 to smoke

# 1. Build + push one image per task to ECR and register one managed Runtime per task (real AWS $).
#    Prove the pipeline first, then scale. Writes tasks/arn_by_image.json (the pool map).
TB_LIMIT=2 python3 $D/deploy/deploy_terminalbench.py     # validate on 2 tasks
python3 $D/deploy/deploy_terminalbench.py                # then the full pool

# 2. (optional) Manually prewarm + MEASURE per-start seconds for the whole pool.
AGENTCORE_REGION=us-west-2 python3 $D/prewarm_pool.py     # measures cold-start seconds per task image

# 3. Build the train/eval dataset from the manifest (offline, stdlib-only).
docker exec agentcore_p4 bash /work/$D/prepare_data.sh    # -> data/tb_train.jsonl + tb_eval.jsonl

# 4. REAL RL train + eval through the isolated pool (async/decoupled slime). Smoke = NR=3.
#    run.sh loads tasks/arn_by_image.json itself; just needs AWS creds in the container.
docker exec -e NR=3 -e EVAL_INTERVAL=2 agentcore_p4 \
    bash -lc 'cd /work && bash '$D'/run.sh 2>&1 | tee '$D'/smoke_run.log'

# 5. Parse the log into CSVs + a PNG.
docker exec agentcore_p4 python3 /work/$D/metrics.py /work/$D/smoke_run.log \
    --csv train.csv --eval-csv train.eval.csv --plot train.png
```

## Knobs (env vars; defaults in `run.sh`)

| Knob | Default | Meaning |
|---|---|---|
| `SANDBOX` | `agentcore-runtime` | fresh AgentCore session per episode. **`local` is rejected.** |
| `TB_PREOPEN_BATCHES` | `2` | **the one pre-open knob**: upcoming batches kept booted ahead, i.e. `TB_PREOPEN_BATCHES × RB × NS` microVMs (64) — idle microVMs you pay for, each counting against the account's concurrent-session quota. `0` = look-ahead off |
| `WARM_IDLE_TIMEOUT` / `WARM_PING_FRACTION` | `900` / `0.5` | heartbeat any session idle past `fraction × timeout` |
| `HARNESS_WORKERS` | `4` | warm in-process harness workers (the harness tier) |
| `EPISODE_MAX_TURNS` / `EPISODE_MAX_TOKENS` | `6` / `1024` | agentic turns / tokens per episode |
| `RB` / `NS` / `NR` | 4 / 8 / 11 | GRPO prompts-per-step / group size / num-rollout (real steps `NR-1`) |
| `CONC` | `⌈2W/E⌉` | episode-admission concurrency `S = CONC × num_engines` (W running + W queued) |
| `MODE` | `A` | stock async only — the pool returns a fan-out list; `B` is rejected |
| `EVAL_INTERVAL` / `EVAL_NS` | 10 / 2 | held-out eval cadence / samples |

Deploy-side: `TB_LIMIT` / `TB_TASKS` (subset), `SKIP_BUILD=1` (reuse pushed images), `TB_WORKDIR`
(the task workdir to snapshot, default `/app`), `CONTINUE_ON_ERROR=1` (skip a task that fails to build).

## What to watch

- `rollout/raw_reward` and `eval/terminalbench` — the learning signal / held-out solve rate.
- `[demo] fresh_session=1 preopened=… open_wait_ms=… preopen_idle_ms=…` — one line per episode: every
  episode is on its own new session; `open_wait_ms` near 0 means its cold start was fully hidden by the
  pre-open. `preopen_idle_ms` is the reverse gap — how long the booted microVM sat idle before a worker
  claimed it. It must stay under `WARM_IDLE_TIMEOUT` (the heartbeat bridges it past
  `WARM_PING_FRACTION × timeout`); if it's routinely large, lower `TB_PREOPEN_BATCHES`.
- `[demo] heartbeat_session=… total_pings=…` — beats fired to hold sessions across the sync stall.
- `perf/sandbox_ms_*` vs `perf/sglang_gen_ms_*` — the sandbox-vs-decode latency split; every episode
  pays one `op=ping` cold open (an image pull + boot), but it runs while the episode is queued.
- `prewarm_pool.py` → `tasks/prewarm_times.csv` — per-Runtime cold-start seconds (min/p50/p95/max).

## Folder layout (self-contained)

- `agentcore_slime_demo/` — vendored library. Key pieces for this demo: `terminalbench_pool.py`
  (fresh session per episode + pre-open + strict routing), `terminalbench_pool_generate.py` (the slime
  hook), `agentcore_runtime_sandbox.AgentCoreRuntimeSandbox` (client; stops the session on exit),
  `sandbox_server_agent.py` (the in-image `/invocations` server), `session_keeper.py` (heartbeat),
  and `task_families.TerminalBenchTask`.
- `fetch_terminalbench.py` — fetch + classify the corpus into `tasks/` + `manifest.json`.
- `deploy/deploy_terminalbench.py` — build+push one image per task, register one Runtime per task,
  bake the hidden oracle + `grade.sh`; writes `tasks/arn_by_image.json`.
- `prewarm_pool.py` — manual pool prewarm + per-start latency probe.
- `build_terminalbench_dataset.py` / `prepare_data.sh` — dataset (step 3) · `run.sh` — the RL run
  (step 4) · `metrics.py` — parse (step 5).
- `tasks/` (gitignored) — the fetched task dirs + `manifest.json` + `arn_by_image.json` + `prewarm_times.csv`.
- `data/` — `tb_train.jsonl` / `tb_eval.jsonl`.
```
