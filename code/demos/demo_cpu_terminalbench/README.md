# demo_cpu_terminalbench — RL over a LARGE image pool (one image per task), one isolated microVM per episode

**Sandbox tier 3: own Runtime, custom image.** Each task pins its own OS image, so each gets its own image and Runtime (microVM). The `warmstart/` probe runs tier 2 (stdlib image) and tier 1 (Code Interpreter) side by side. See the tier table in [`../README.md`](../README.md#which-sandbox-does-a-task-need-three-tiers).

Terminal-Bench ships each task as its **own container environment** (a task-specific `Dockerfile` +
a hidden pytest oracle). This demo turns that into a **large image pool**: one container image → one
AgentCore Runtime **per task** (~229 single-container tasks), and every rollout episode is routed to
**its task's own Runtime**. Where `demo_cpu_tasks VARIANT=imagepool` has 3 images routed by *family*, this is the
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

## Step by step (numbered scripts; run from this dir on the HOST — container steps `docker exec` themselves)

There is no single sandbox image here: every task gets its own image and Runtime, built from `tasks/`
by `deploy/deploy_terminalbench.py`. The step scripts wrap that script.

```bash
bash 1_prepare_data.sh     # 1. container: fetch_terminalbench.py (tasks/ + manifest.json, ~229 single-container tasks)
                           #    then prepare_data.sh -> data/tb_{train,eval}.jsonl
bash 2_build_image.sh      # 2. TB_NO_REGISTER=1 deploy_terminalbench.py: build + push one arm64 image per task (ECR acs-tb-*)
bash 3_push_ecr.sh         # 3. no-op: prints tasks/images.csv (step 2 already pushed)
bash 4_deploy_runtime.sh   # 4. deploy_terminalbench.py: one Runtime per task (acs_tb_*) -> tasks/arn_by_image.json
python3 validate_oracle.py # 4b. (container, AGENTCORE_REGION=us-west-2) null must FAIL, oracle must PASS -> tasks/oracle.csv
bash 1_prepare_data.sh     # 4c. rebuild the dataset from the usable tasks only (oracle=1, null=0)
bash 5_run.sh              # 5. container: 10 train steps + 1 eval through the pool; logs/<tag>.{log,csv,eval.csv,png}
bash 6_cleanup.sh          # 6. delete every Runtime in tasks/arn_by_image.json
```

- **Validate on 2 tasks first** (real AWS $): `TB_LIMIT=2 bash 2_build_image.sh && TB_LIMIT=2 bash 4_deploy_runtime.sh`,
  then drop `TB_LIMIT` for the full pool.
- `run.sh` loads `tasks/arn_by_image.json` itself, so `deploy/runtime.env` stays empty.
- Last full run: 218/229 Runtimes deployed, 166 usable after the oracle check (135 train + 31 eval).
- Optional: `python3 prewarm_pool.py` measures per-task cold-start seconds (`tasks/prewarm_times.csv`);
  `probe_coldstart.py` splits each cold open into pull+boot vs ready.
- Train to convergence (inside the container): `train.sh` (`STEPS`, `EVAL_EVERY`, `SANDBOX_CONC`).

## BENCH=swebench — a SWE-bench Verified subset on the same pool

**Use case covered.** RL on real GitHub issues: the agent edits a real repo (django, sympy, sphinx,
pytest, astropy, requests, xarray, pylint, flask) and the reward is the instance's hidden
FAIL_TO_PASS + PASS_TO_PASS tests. 20 instances, one Runtime each, the same `TerminalBenchPool`
(fresh session per episode, pre-open, heartbeat, strict routing by `task.image` = instance id).

**Why this demo.** It is the standard agentic-coding benchmark. It also shows that the image-pool
pattern works for images the microVM fleet cannot run: SWE-bench images are amd64-only and ~1-2 GB
compressed (3-6 GB on disk), so each Runtime sits on an **m5.xlarge capacity provider**
(`acs_swe_cp`, 60 GiB root). One session is one whole instance, so isolation is still per machine.

**Design considerations.**
- *Hidden tests.* The `test_patch` never enters the image or the dataset row; it lives only in
  the trainer-side `swebench/manifest.json`. After the agent's turns, `swebench.grade_swebench`:
  1. resets the files the patch touches to `base_commit`, discarding agent edits (added files are removed);
  2. `git apply`s the patch;
  3. runs the official per-repo/version `test_cmd` in conda env `testbed`;
  4. parses the log with vendored SWE-bench parsers.
- *Grading rule.* Solved iff every F2P and P2P test is PASSED/XFAIL, as in the official grader.
- *Deviation: no reinstall.* The official `install` step is not re-run, because repos are editable
  installs and the instance may lack PyPI egress. The oracle check catches any instance this breaks.
- *Selection.* `fetch_swebench.py` picks "<15 min fix" instances with 1-3 F2P, 1-150 P2P, at most
  2 test files and an image of at most 3 GB, from repos the vendored parsers support (quota: django 6,
  sympy 4, sphinx 3, pytest 2, one each for the rest). matplotlib, sklearn and seaborn are excluded
  (images of 4-8 GB, slow suites).
- *Cold start.* Opening a session boots an m5 instance and pulls the image (~100 s measured). That
  is much slower than a microVM, so keep `TB_PREOPEN_BATCHES` on. Idle instances are released
  600 s after their last session.
- *Server shim.* On Instances the data plane sends chunked `/invocations` bodies, which the shared
  `acs_http_server.py` misreads as a ping. The images therefore run `deploy/acs_swe_http_server.py`,
  which accepts both chunked and Content-Length bodies.

```bash
python3 fetch_swebench.py                   # swebench/manifest.json + data/swebench_{train,eval}.jsonl
python3 deploy/deploy_swebench.py           # wrap + push + one Runtime per instance -> swebench/arn_by_image.json
AGENTCORE_REGION=us-west-2 python3 swebench_oracle_check.py   # gold must solve, empty patch must not -> swebench/oracle.csv
python3 fetch_swebench.py --data-only       # keep only usable instances
BENCH=swebench bash run.sh                  # (container) same run, SWE data + pool map, eval/swebench
```

Knobs:
- Deploy: `SWE_TASKS`, `SWE_LIMIT`, `SWE_BUILD_CONC`, `SWE_REBUILD`, `SWE_NO_REGISTER`,
  `SWE_NO_PUSH` (local build; then `swebench_oracle_check.py --local`), `SWE_INSTANCE`, `SWE_ROOT_GIB`.
- Data: `SWE_QUOTA`, `SWE_INSTANCES`, `SWE_MAX_TURNS` (12), `SWE_TEST_TIMEOUT` (600 s, which keeps
  the grade exec under `INVOKE_READ_TIMEOUT` 900).
- Episodes: consider raising `EPISODE_MAX_TURNS` / `TB_EXEC_TIMEOUT`, since SWE episodes are longer
  than Terminal-Bench ones.

Cleanup: delete the `acs_swe_*` Runtimes listed in `swebench/arn_by_image.json` and then the
`acs_swe_cp` capacity provider. `6_cleanup.sh` covers only `tasks/`.

## Warm-start probe (`warmstart/`): why the pre-open works, and what heartbeat adds

A small host-only probe (no GPU, no model, no trainer; about 7 min, a few cents) that isolates the two
mechanisms above. The same M=24 MockPolicy `sql` episodes run in batches of 6 in three modes:

- **cold**: open at episode start.
- **preopen**: the next batch is booted while this one runs, the same look-ahead idea as
  `TB_PREOPEN_BATCHES`.
- **heartbeat**: sessions sit idle 150 s, past a 60 s idle timeout, with and without `SessionKeeper`
  pings.

The probe runs against a stdlib CPU Runtime, `acs_warmstart_cpu`, and the managed Code Interpreter. The
Runtime reuses the shared `acs-sandbox-cpu:latest` image with a 60 s `idleRuntimeSessionTimeout`. That value is
shortened so the probe finishes in minutes. Training Runtimes keep the 15-min default (`WARM_IDLE_TIMEOUT=900`,
pings at 450 s), and the mechanism is the same.

```bash
bash warmstart/run_warmstart.sh                                  # creates the Runtime if needed, then runs
BACKENDS=runtime,codeinterpreter bash warmstart/run_warmstart.sh # both backends (the recorded run)
BACKENDS=local IDLE_S=3 GAP_S=5 bash warmstart/run_warmstart.sh  # offline, seconds
bash warmstart/run_warmstart.sh cleanup                          # delete the Runtime
```

Recorded run (`warmstart/logs/warmstart_20260925_100103.*`, 2026-09-25, us-west-2; all rewards 1.0):

| backend | mode | wait p50 | episodes that waited for a boot | idle held | makespan |
|---|---|---:|---:|---:|---:|
| own Runtime | cold | 774 ms | 24/24 | 0 s | 31.9 s |
| own Runtime | preopen | **5 ms** | 8/24 (the first batch) | 75 s | 25.5 s (−20 %) |
| Code Interpreter | cold | 1715 ms | 24/24 | 0 s | 28.7 s |
| Code Interpreter | preopen | **0 ms** | 6/24 (the first batch) | 70 s | 24.3 s (−15 %) |

| backend | heartbeat | sessions surviving 150 s idle | resume |
|---|---|---:|---|
| own Runtime | on (14 pings) | **3/3** | 104 ms, files intact |
| own Runtime | off | 0/3 | **silently reclaimed**: the same session id returns a fresh, empty sandbox |
| Code Interpreter | on / off | 0/3 | expires at its `sessionTimeoutSeconds` regardless of pings |

What the probe shows:

- **Pre-open removes the boot from the critical path; only the first batch pays it.** The price is
  "idle held": booted sessions waiting for work. That is the same trade-off `preopen_idle_ms` shows at
  full scale. The smoke had 64 sessions ahead and about 95 % idle, so try `TB_PREOPEN_BATCHES=1`.
- **Heartbeat only helps your own Runtime.** Without it, a reclaimed session looks healthy but has lost
  its files, which is why this demo's keeper is always on. On the Code Interpreter the lever is a
  generous session timeout, not pings.

Knobs: `BACKENDS` (runtime / codeinterpreter / local), `MODES` (cold,preopen,heartbeat), `FAMILY`,
`M`, `BATCH`, `THINK_S`, `IDLE_S`, `GAP_S`, `HB_SESSIONS`. `warmstart/runtime.env` holds the Runtime ARN.

## Knobs (env vars; defaults in `run.sh`)

| Knob | Default | Meaning |
|---|---|---|
| `SANDBOX` | `agentcore-runtime` | fresh AgentCore session per episode. **`local` is rejected.** |
| `TB_PREOPEN_BATCHES` | `2` | **the one pre-open knob**: upcoming batches kept booted ahead, i.e. `TB_PREOPEN_BATCHES × RB × NS` microVMs (64) — idle microVMs you pay for, each counting against the account's concurrent-session quota. `0` = look-ahead off |
| `WARM_IDLE_TIMEOUT` / `WARM_PING_FRACTION` | `900` / `0.5` | heartbeat any session idle past `fraction × timeout`. This demo's `SessionKeeper` is **always on** (pre-opened sessions must survive), so the cross-demo `HEARTBEAT=1` knob is not needed here |
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
- `warmstart/` — the warm-start probe: `warmstart_demo.py`, `deploy_warmstart.py`, `run_warmstart.sh`,
  `runtime.env`, `logs/`.
- `fetch_terminalbench.py` — fetch + classify the corpus into `tasks/` + `manifest.json`.
- `deploy/deploy_terminalbench.py` — build+push one image per task, register one Runtime per task,
  bake the hidden oracle + `grade.sh`; writes `tasks/arn_by_image.json`.
- `prewarm_pool.py` — manual pool prewarm + per-start latency probe.
- `validate_oracle.py` — null/oracle check per deployed task → `tasks/oracle.csv` (the `usable` filter).
- `build_terminalbench_dataset.py` / `prepare_data.sh` — dataset (step 1) · `run.sh` — the RL run
  (step 5) · `metrics.py` — parse (run by step 5).
- `1_…6_*.sh` + `deploy/demo.env` — the numbered steps (BUILD/PUSH/DEPLOY/CLEANUP_CMD overrides).
- `tasks/` (gitignored) — the fetched task dirs + `manifest.json` + `arn_by_image.json` + `prewarm_times.csv`.
- `data/` — `tb_train.jsonl` / `tb_eval.jsonl`.
```
