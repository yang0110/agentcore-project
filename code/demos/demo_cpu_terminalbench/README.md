# demo_cpu_terminalbench — RL over a LARGE image pool (one image per task), kept warm the reliable way

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

## The problem this demo solves: keeping a large pool warm

A pool of hundreds of task images makes **cold start** the dominant cost. Opening a *fresh* session
per episode means: on the first episode of every task, AgentCore allocates a microVM, **pulls that
task's image from ECR**, and boots it before any work happens. Do that per episode and the rollout
spends most of its wall-clock waiting on pulls.

AgentCore gives exactly **one guaranteed warm lever: session affinity** — the same `runtimeSessionId`
stays pinned to the same warm microVM; a new/absent one cold-starts. There is **no documented
cross-session image cache**, and a microVM is terminated + sanitized when its session ends. So
"prewarm and release" is best-effort only. **The reliable way is to keep a live session per task and
heartbeat it** — which is what this demo does.

### Warm-shared session reuse + heartbeat (the reliable path)

1. **One live `runtimeSessionId` per task.** Every episode routed to a task reuses that task's ONE
   shared session. Because AgentCore scopes a session per `(runtimeArn, runtimeSessionId)` and each
   task has its own ARN, one shared id yields **exactly one warm microVM per task**
   (`WarmSharedRuntimeSandbox`, `terminalbench_pool.py`).
2. **Per-episode workdir isolation via a `sub_session`.** Sharing a microVM means episodes would
   otherwise see each other's files. The server (`SANDBOX_SHARED_MODE=cpu`) resets a **pristine copy
   of the task's baked workdir template** per episode: `op=open` → `cp -a /opt/acs/task_template/.
   /sessions/<sub>/`; every `exec`/`write_file`/`read_file` resolves inside that dir; `op=close`
   removes it. The reset is cheap because the task's environment is already **in the image** — nothing
   is re-pulled, only a small directory is copied.
3. **A `SessionKeeper` heartbeat holds each session warm across the weight-sync stall.** In
   async/decoupled slime, generation pauses while the trainer syncs weights into SGLang; with no
   invokes flowing, AgentCore would idle-reclaim the microVMs (~15-min timeout). The pool registers
   ONE keepalive **anchor** per task ARN; the keeper pings any anchor that goes idle past
   `WARM_PING_FRACTION × WARM_IDLE_TIMEOUT`, so the whole pool survives the barrier.

**Honest caveat — isolation scope.** The per-episode reset isolates the **workdir**, where
Terminal-Bench tasks graded on file artifacts live (the large majority). A task that mutates **global**
OS state (`apt install`, `/etc`, `/root`) and grades on it would leak across episodes. For those, set
`TB_SESSION_RECYCLE_EVERY=K` to mint a fresh session for that task every K episodes (one amortized
cold start per K, trading a little warmth for a clean OS). Default `0` = never recycle (pure warm).

---

## How an episode flows

```
slime → terminalbench_pool_generate.generate → TerminalBenchPool.submit
  worker pulls episode → route by task.image → _runtime_arn_for_image → task's Runtime ARN
    ensure warm anchor for that ARN (SessionKeeper heartbeats it)   ← one live session per task
    open per-episode sub_session (server cp's a fresh workdir template)   ← isolation + no re-pull
      write PROBLEM.md (the instruction) into the fresh workdir
      MiniClaudeHarness loop: LLM via adapter (policy under training), bash/write_file/read_file tools
    grade: bash /opt/acs/oracle/grade.sh  (injects hidden tests + runs run-tests.sh → exit 0 = solved)
  finish_session → loss-masked GRPO samples (list); adapter.drop_session
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
AGENTCORE_REGION=us-west-2 python3 $D/prewarm_pool.py     # KEEP_WARM=1 to leave sessions alive

# 3. Build the train/eval dataset from the manifest (offline, stdlib-only).
docker exec agentcore_p4 bash /work/$D/prepare_data.sh    # -> data/tb_train.jsonl + tb_eval.jsonl

# 4. REAL RL train + eval through the WARM-SHARED pool (async/decoupled slime). Smoke = NR=3.
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
| `SANDBOX` | `agentcore-warm-shared` | warm-shared per-task sessions. **`local` is rejected.** |
| `TB_SHARED_RUNTIME_SESSION` | per-run | shared session base; the pool derives one warm session per task |
| `TB_SESSION_RECYCLE_EVERY` | `0` | mint a fresh session every K episodes/task (0 = never; use for global-state tasks) |
| `WARM_IDLE_TIMEOUT` / `WARM_PING_FRACTION` | `900` / `0.5` | heartbeat any session idle past `fraction × timeout` |
| `HARNESS_WORKERS` | `4` | warm in-process harness workers (the harness tier) |
| `EPISODE_MAX_TURNS` / `EPISODE_MAX_TOKENS` | `6` / `1024` | agentic turns / tokens per episode |
| `RB` / `NS` / `NR` | 4 / 8 / 11 | GRPO prompts-per-step / group size / num-rollout (real steps `NR-1`) |
| `CONC` | `⌈W/E⌉` | episode-admission concurrency `S = CONC × num_engines` |
| `MODE` | `A` | stock async only — the pool returns a fan-out list; `B` is rejected |
| `EVAL_INTERVAL` / `EVAL_NS` | 10 / 2 | held-out eval cadence / samples |

Deploy-side: `TB_LIMIT` / `TB_TASKS` (subset), `SKIP_BUILD=1` (reuse pushed images), `TB_WORKDIR`
(the task workdir to snapshot, default `/app`), `CONTINUE_ON_ERROR=1` (skip a task that fails to build).

## What to watch

- `rollout/raw_reward` and `eval/terminalbench` — the learning signal / held-out solve rate.
- `[demo] warm_anchor …` — one line per task session coming up (and `live_sessions` = tasks kept
  warm); `[demo] warm_reuse=1 … ep_on_session=N` — episodes reusing an already-warm session (N grows).
- `[demo] heartbeat_session=… total_pings=…` — beats fired to hold sessions across the sync stall.
- `perf/sandbox_ms_*` vs `perf/sglang_gen_ms_*` — the sandbox-vs-decode latency split; with warming
  working, the per-episode sandbox `op=open` stays small (no per-episode image pull).
- `prewarm_pool.py` → `tasks/prewarm_times.csv` — per-Runtime cold-start seconds (min/p50/p95/max).

## Folder layout (self-contained)

- `agentcore_slime_demo/` — vendored library. Key pieces for this demo: `terminalbench_pool.py`
  (warm-shared pool + SessionKeeper anchors + recycle), `terminalbench_pool_generate.py` (the slime
  hook), `agentcore_runtime_sandbox.WarmSharedRuntimeSandbox` (client), `sandbox_server_agent.py`
  (`SANDBOX_SHARED_MODE=cpu` workdir-reset handler), `session_keeper.py` (heartbeat),
  `task_families.TerminalBenchTask`, and `sandbox.make_sandbox("agentcore-warm-shared", …)`.
- `fetch_terminalbench.py` — fetch + classify the corpus into `tasks/` + `manifest.json`.
- `deploy/deploy_terminalbench.py` — build+push one image per task, register one Runtime per task,
  bake the workdir template + hidden oracle + `grade.sh`; writes `tasks/arn_by_image.json`.
- `prewarm_pool.py` — manual pool prewarm + per-start latency probe.
- `build_terminalbench_dataset.py` / `prepare_data.sh` — dataset (step 3) · `run.sh` — the RL run
  (step 4) · `metrics.py` — parse (step 5).
- `tasks/` (gitignored) — the fetched task dirs + `manifest.json` + `arn_by_image.json` + `prewarm_times.csv`.
- `data/` — `tb_train.jsonl` / `tb_eval.jsonl`.
```
