# demo_cpu_tasks — agentic RLVR on our own AgentCore Runtimes: every row is a task + an image; `DATASET=` any dataset or mix (incl. Terminal-Bench)

**Sandbox tier 2: own Runtime, generic image** (`SANDBOX=agentcore-runtime`, the only backend). Two
arm64 microVM images: `cpu` (stdlib) and `libs` (+ pandas / scipy / scikit-learn). **Plus tier 3 for
`DATASET=terminalbench`**: one custom image + Runtime per Terminal-Bench task (~166), in the same run
and the same batch if you mix it. See the tier table in
[`../README.md`](../README.md#which-sandbox-does-a-task-need-three-tiers).

**The general pattern: ONE ROW = ONE TASK + ONE IMAGE, ONE IMAGE = ONE Runtime.** Every dataset row names
its task (`metadata.task`: family, seed files, hidden checker) and its sandbox image
(`metadata.task.image`). Rows can share an image: every mbpp row is `cpu`, every pandas row `libs`, while
each Terminal-Bench task has its own. `image_routing.py` maps each image to its Runtime ARN, strictly.
Each of the `RB × NS` samples of a step then gets **its own fresh session** on its row's Runtime, so
**sessions per step = RB × NS** (80 for the default 10-dataset mix, RB=10), spread over however many Runtimes the batch touches.
A mixed batch (`DATASET=mbpp,terminalbench`) is the same thing with rows that name different images.

**The point of this demo: most agentic-RL tasks from the literature need only a generic sandbox.** One
stdlib image serves every dataset below except the pandas/scipy/sklearn ones, which get one extra image.
Nothing is task-specific in the image, so it is built and deployed once and reused by every dataset.

An agentic-RL loop RL-trains Qwen3-4B on verifiable tasks. The agent is the tool-calling mini_claude
loop (`write_file` / `bash` / `read_file`). It runs **inside the episode's sandbox session**
(`in_session_harness.py` writes `session_runner.py` into the session and long-polls it) and calls the
policy through this box's per-engine `AnthropicAdapter`, which records the exact tokens. A **hidden
grader** scores the working directory after the loop stops. The trainer is real slime,
async/decoupled (`train_async.py`), 4 train + 4 inference GPUs, 0 GPUs for the env. Only the adapter
and the token record live in the trainer: **there is no in-trainer harness** (`HARNESS=direct` and
`HARNESS_WHERE=trainer` were removed on 2026-09-28).

**No Code Interpreter.** `SANDBOX=agentcore` (the managed Code Interpreter) was removed on 2026-09-28.
It has no route back to the adapter, so the agent loop would have had to run in the trainer. It
survives only as a probe (`compare_backends.py`); `run.sh` rejects it, and `SANDBOX=local`.

This folder also absorbed several former demos:

- `demo_cpu_datasets` and `demo_cpu_mixtask`: now `DATASET=<name>` / `DATASET=a,b,c` / `DATASET=mixed`.
- `demo_cpu_imagepool`: its pandas/scipy/sklearn tasks are now `DATASET=libs`, on the one `libs` image
  (the three per-library images are gone).
- `demo_cpu_heartbeat`: now the `HEARTBEAT` knob (on by default).
- `demo_cpu_customruntime`: now `compare_backends.py`.
- `VARIANT=bigimage`: moved out to `demo_cpu_instance_pool` (itself removed 2026-09-28).
- `demo_cpu_terminalbench` (merged 2026-09-28): now `DATASET=terminalbench` (section 3) and
  `VARIANT=terminalbench` for its per-task image pool; its tooling lives in `terminalbench/`.
  SWE-bench (one image per row, on m5 Instances) stays its own demo, `demo_cpu_swebench`.
- `demo_cpu_scale_lanes` (removed 2026-09-28): its decoupled harness tier is now every demo's default
  (`harness_pool.py`), its 4:256:256 run is the **`SCALE=1`** preset (section below), its offline
  sweep is `harness_scaling_demo.py` and its logs are in `logs/scale/`.

**Harness tier (the default rollout topology).** `harness_pool.py` joins three independently sized tiers
per episode: `HARNESS_WORKERS` dispatch workers (default RB×NS = 32) pull episodes off one queue; each
episode takes its sandbox first (a fresh session, or with `SANDBOX_SESSIONS=M>0` one of M pooled
sessions per image in a fresh `/workspace/ep-<hex>` workdir), then the least-loaded SGLang engine,
sticky for all its turns. `throughput = min(inference, sandbox, harness)`: the harness must scale along
with SGLang and the sandbox, or a few coroutines silently become the ceiling, and it scales with the
sandbox (one mini_claude loop per session), not inside the trainer, as in DSec. Episodes return one bare
Sample, the shape slime's fully-async driver (the only rollout driver) needs. See [`../README.md`](../README.md#harness-tier-default-in-every-demo).

## 1. The tasks: dataset, source, environment, image

Every row is self-describing (`metadata.task = {family, task_id, image, spec, …}`). `generate._task_from_sample`
rebuilds the right task (seed files, hidden checker and reward) for each row, so the same harness trains
all of them. Select a dataset with `DATASET=<file prefix>` (`FAMILY` is an alias). Each dataset writes
its own `eval/<name>` metric and its own `ckpt/Qwen3-4B_<name>_async`.

| `DATASET` | family | dataset (URL) | train/eval | reward | env deps (checker) | image |
|---|---|---|---:|---|---|---|
| `mbpp` | coding | [google-research-datasets/mbpp](https://huggingface.co/datasets/google-research-datasets/mbpp) | 500/8 | hidden asserts | python stdlib | `cpu` |
| `humaneval` | coding | [openai/openai_humaneval](https://huggingface.co/datasets/openai/openai_humaneval) | 140/24 | the row's own `test` fn | python stdlib | `cpu` |
| `gsm8k` | math | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) (`main`) | 200/40 | exact integer match | python stdlib | `cpu` |
| `math500` | math | [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) (numeric subset) | 160/40 | numeric match, tol 1e-4 | python stdlib | `cpu` |
| `deepscaler` | math | [agentica-org/DeepScaleR-Preview-Dataset](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset) | 400/40 | numeric match, tol 1e-4 | python stdlib | `cpu` |
| `gretelsql` | sql | [gretelai/synthetic_text_to_sql](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql) | 300/40 | SQL result set vs gold | `sqlite3` | `cpu` |
| `hotpotqa` | rag | [hotpotqa/hotpot_qa](https://huggingface.co/datasets/hotpotqa/hotpot_qa) (distractor) | 300/40 | SQuAD-normalized EM | files in `./docs` | `cpu` |
| `xlam` | tool call | [minpeter/xlam-function-calling-60k-parsed](https://huggingface.co/datasets/minpeter/xlam-function-calling-60k-parsed) | 300/40 | order-insensitive gold-call match | python stdlib | `cpu` |
| `mcp` | tool call | [Team-ACE/ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) (MCP-style schemas) | 300/40 | order-insensitive gold-call match | python stdlib | `cpu` |
| `libs` | datasci + scipy + sklearn | synthetic (`task_families.py`) | 72/6 (+2 per family) | numeric assert on `answer` | **pandas, scipy, scikit-learn** | **`libs`** |
| `mixed` | math + sql + ctf + datasci | synthetic mix | 174/9 | per family | stdlib + pandas | `cpu` + **`libs`** (datasci rows) |
| `terminalbench` | terminal tasks (whole-OS) | [laude-institute/terminal-bench](https://github.com/laude-institute/terminal-bench) | 135/31 | the task's hidden pytest oracle | the task's own Dockerfile | **one per task** (section 3) |
| `math` `swe` `tool_use` `sql` `ctf` `rag` `regex` `config` `data` `planning` | synthetic zoo (`build_family_datasets.py`) | — | tiny (4–9) | per family | stdlib | `cpu` |

**Image routing.** Each row carries its `image` (`cpu`, `datasci` / `scipy` / `sklearn`, or a
Terminal-Bench task name; rows written before 2026-09-28 fall back to their task class's tag).
`run.sh` calls `image_routing.py`, which builds `AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE`: `cpu` is the
K=2 `AGENTCORE_SANDBOX_RUNTIME_ARNS`, the three library tags alias the `libs` Runtime, and when the
data has Terminal-Bench rows it merges the pool map `terminalbench/tasks/arn_by_image.json`. If the
dataset needs an image that has no Runtime, `run.sh` stops before any GPU work. Without that check an
unmapped row would run on another image and train on a silent 0 reward.

**Where a task-specific image IS needed:** the task pins its own OS or toolchain.
- Terminal-Bench, one image per task: `DATASET=terminalbench` here (section 3).
- SWE-bench, one image per dataset row: `demo_cpu_swebench`.
- An image larger than 2 GB: an m5 capacity-provider Instance, as in `demo_cpu_swebench`.
- CUDA: the GPU demos.

**Mixed-task batches (the default).** With no `DATASET`, a run trains on
every literature dataset: the real HF RLVR sets plus Terminal-Bench,
`mbpp,humaneval,gsm8k,math500,deepscaler,gretelsql,hotpotqa,xlam,mcp,terminalbench` (the synthetic zoo
and SWE-bench, which is `demo_cpu_swebench`, are left out). So **every step's batch holds one prompt of
each of the 10 datasets**: 2 coding, 3 math, 1 SQL, 1 RAG, 2 tool-calling and 1 Terminal-Bench
(10 × NS=8 = 80 episodes, on the `cpu` Runtime and the per-task
Terminal-Bench Runtimes at once). Any comma list works the same way (`DATASET=gsm8k,terminalbench`),
and so do the prebuilt `mixed` (sql/math/ctf/datasci) and `libs` (datasci/scipy/sklearn).
How the mix is guaranteed:
- `mix_data.py` groups the rows into task strata (source dataset × family), balances them to the
  smallest stratum (135 rows each for the default, capped by Terminal-Bench), and deals them
  round-robin into `ckpt/_mix/<a+b+…>_train.jsonl`. Row *i* belongs to task *i* mod K.
- `run.sh` then drops `--rollout-shuffle`, so each run of RB consecutive prompts spans min(RB, K)
  tasks. If you did not set `RB` and K > RB, RB is raised to K (the default gets RB=10, GBS=80), so
  no step is missing a task. Instead of per-epoch reshuffles, the file holds as many reshuffled
  passes as `NR × RB` needs.
- Eval uses a balanced mix too (`eval/mix` plus `eval/mix_<family>` per task; mbpp's 8 eval rows cap
  it at 8 per task).
- **Evidence.** `batch_mix_log.py` (slime's custom rollout/eval log hooks) logs what was *actually*
  trained on each step: a `[batch] rollout=N n_tasks=5 coding=8 math=8 …` line plus
  `mix/n_tasks`, `mix/episodes_<family>`, `mix/reward_<family>` in the step metrics. `metrics.py`
  prints the mix summary (tasks/step, per-family reward first→last) and plots one reward panel per
  family. Under the fully-async driver the trained batch is the first RB groups to *finish*, so a slow
  Terminal-Bench group can land a step later than it was drawn; these logs show the true composition.

GRPO normalizes the advantage **within** a group of `NS` repeats of one prompt, so the families'
reward scales never mix. Terminal-Bench rows in the mix switch on the Terminal-Bench run settings
(section 3: look-ahead pre-open, 8 turns × 1024 tokens, `MAX_TOKENS_PER_GPU=16384`) for the whole
batch. `DATASET=<one name>` trains on a single task (shuffled as usual). `mixed`/`libs` are built by
`build_mixed_datasets.py --name <n> --families …`.

## 2. Step by step (run from this dir on the HOST — container steps `docker exec` themselves)

```bash
bash 1_prepare_data.sh     # container: prepare_data.sh (all datasets, or DATASET=<name>) + verify_data.py
bash 2_build_image.sh      # arm64 images from deploy/Dockerfile.sandbox.{cpu,libs}
bash 3_push_ecr.sh         # -> ECR acs-tasks-cpu, acs-tasks-libs
bash 4_deploy_runtime.sh   # VPC Runtimes acs_tasks_cpu + acs_tasks_libs -> deploy/runtime.env
                           #   (AGENTCORE_SANDBOX_RUNTIME_ARNS = cpu, ..._ARN_BY_IMAGE = {cpu, libs})
VARIANT=terminalbench bash 2_build_image.sh && VARIANT=terminalbench bash 4_deploy_runtime.sh
                           # the per-task Terminal-Bench Runtimes the DEFAULT mix needs (section 3)
bash 5_run.sh              # container: DEFAULT mixed batch (coding+math+sql+rag+terminalbench), 10 steps + 1 eval
                           #   logs/<tag>.{log,csv,eval.csv,png}; [batch] lines = per-step task mix
DATASET=gsm8k bash 5_run.sh     # single task, no Terminal-Bench Runtimes needed; good first run
DATASET=libs  bash 5_run.sh     # pandas / scipy / sklearn tasks on the libs Runtime
DATASET=gsm8k,gretelsql,hotpotqa bash 5_run.sh   # ad-hoc mix without Terminal-Bench, eval/mix + eval/mix_<family>
bash 6_cleanup.sh          # delete the Runtimes
```

- Stdlib datasets only? `IMAGES=cpu:Dockerfile.sandbox.cpu` on steps 2–4 skips the libs image.
  Without step 4, `run.sh` uses the baked shared K=2 `acs_sandbox_cpu` ARNs (stdlib datasets only).
- Both images are `python:3.12-slim` plus the sandbox server; `libs` adds numpy, pandas, scipy and
  scikit-learn. The in-session loop needs only `python3`, which both have.
- The Runtimes are `networkMode=VPC` so the session can dial the adapter back (SG→SG rule on
  `ADAPTER_PORT_BASE`…, see `docs/agentcore-networking.md`). If the session cannot reach the adapter,
  or the image has no `python3`, the episode **fails** (`HarnessPlacementError`); the loop never moves
  into the trainer.
- `AGENTCORE_SANDBOX_RUNTIME_ARNS` (comma list) fans episodes out round-robin for the Runtime-count
  ablation (`run_runtime_ablation.sh`).

**Same episode, same reward, different ops: `compare_backends.py`** (a probe, not a training path)

```bash
python3 compare_backends.py --backends local                                  # offline: solved=True reward=1.0
AGENTCORE_SANDBOX_RUNTIME_ARN=<cpu-arn> python3 compare_backends.py --backends local agentcore agentcore-runtime
```

The script asserts that every backend (including the managed Code Interpreter) gives the same reward,
then prints who builds the image, which deps you get, and the deploy cost. Old logs are in
`logs/customruntime/`.

## 3. `DATASET=terminalbench`: one image per task (was `demo_cpu_terminalbench`)

Terminal-Bench ships each task as its **own container environment**: a task-specific `Dockerfile` plus
a hidden pytest oracle. Each task becomes one image and one AgentCore Runtime (~229 single-container
tasks fetched, 218 deployed, **166 usable** after the oracle check: 135 train + 31 eval). The row's
`image` is the task name, so an episode always lands on **its task's own Runtime**. It is the same
row → image → Runtime routing as `cpu`/`libs`, at task scale. The reward is the task's hidden pytest
oracle (`/tests/run-tests.sh`, exit 0 = solved), injected only at grade time.

### Isolation vs. cold start

Terminal-Bench tasks act on the **whole OS**: they install packages, edit `/etc`, start services.
Episodes must be isolated from each other at the machine level. On AgentCore the boundary is the
session (one `runtimeSessionId` = one dedicated microVM, terminated and sanitized when stopped), so
**every episode gets a brand-new session, never reused**. `SANDBOX_SESSIONS` pooling does not apply to
these rows. The price is a cold start per episode (allocate a microVM, pull the task's image, boot).
The demo hides that cost instead of avoiding it:

1. **Strict routing.** `task.image` → the pool map. An unmapped task is an error, never a fallback.
2. **Pre-open.** slime hands the pool only the current batch. The look-ahead data source
   (`agentcore_slime_demo/tb_lookahead_source.py`, added by `run.sh` via `--data-source-path`) replays
   slime's prompt order and tells the pool about the next `TB_PREOPEN_BATCHES` rollouts too. The pool
   keeps `TB_PREOPEN_BATCHES × RB × NS` sessions (64) booted ahead, pooled per task
   (`TerminalBenchPool._refill`). `[demo] open_wait_ms` is the part of the cold start still on the
   critical path.
3. **Stop on exit.** `StopRuntimeSession` right after grading frees the quota slot.
4. **Heartbeat across the weight-sync stall.** Every open session (pre-opened or running) is registered
   with a `SessionKeeper`, which pings any session idle past `WARM_PING_FRACTION × WARM_IDLE_TIMEOUT`.
   It is always on for these rows. The `HEARTBEAT` knob covers the family rows.

The other rows get pre-open too (default on): `PREOPEN` (RB×NS, one step) fresh `cpu`/`libs` sessions
stay booted ahead per image (`agentcore_slime_demo/preopen.py`). No look-ahead is needed there, because
all those rows share one image. `PREOPEN=0` opens on demand.

**Mixed batches.** `generate.generate` sends Terminal-Bench rows to `TerminalBenchPool`, a
`HarnessPool` subclass with its own sandbox tier. Every other row goes through the default harness
tier. Both pools share the engines' adapters, so one batch can span both.

```
slime → generate.generate → family == terminalbench → TerminalBenchPool.submit
  harness worker → route by task.image → that task's Runtime ARN
  take a pre-opened session (or open one now)            ← its own microVM, heartbeated
    write PROBLEM.md into the task's workdir (/app)
    InSessionHarness: the mini_claude loop runs IN the session, LLM via the least-loaded engine's adapter
    grade: bash /opt/acs/oracle/grade.sh (injects the hidden tests, runs run-tests.sh)
  ONE loss-masked Sample; StopRuntimeSession
```

**Run settings `run.sh` applies when the data has Terminal-Bench rows.** The look-ahead data source
slides with the fully-async driver's per-group draws (drawn groups appended, the next
`TB_PREOPEN_BATCHES×RB` prompts announced). `HARNESS_WORKERS=RB×NS`,
`CONC=⌈2W/E⌉` (W running + W queued), `EPISODE_MAX_TURNS=8`, `EPISODE_MAX_TOKENS=1024`,
`MINI_AGENT_MAX_TOOL_CHARS=2000`, `TB_EXEC_TIMEOUT=180`, `MAX_TOKENS_PER_GPU=16384` (one full
trajectory). Each one can be overridden.

### Steps (the per-task pool is `VARIANT=terminalbench`)

```bash
VARIANT=terminalbench TB_LIMIT=2 bash 2_build_image.sh   # validate on 2 tasks first (real AWS $)
VARIANT=terminalbench bash 2_build_image.sh     # fetch (if needed) + build + push one arm64 image per task (ECR acs-tb-*)
VARIANT=terminalbench bash 3_push_ecr.sh        # no-op: prints terminalbench/tasks/images.csv
VARIANT=terminalbench bash 4_deploy_runtime.sh  # one Runtime per task (acs_tb_*) -> terminalbench/tasks/arn_by_image.json
docker exec -e AGENTCORE_REGION=us-west-2 agentcore_p4 \
  python3 /work/code/demos/demo_cpu_tasks/terminalbench/validate_oracle.py   # null FAILS, oracle PASSES -> tasks/oracle.csv
DATASET=terminalbench bash 1_prepare_data.sh    # rows for the usable tasks only -> data/terminalbench_{train,eval}.jsonl
DATASET=terminalbench bash 5_run.sh             # or a mix: DATASET=gsm8k,terminalbench bash 5_run.sh
VARIANT=terminalbench bash 6_cleanup.sh         # delete every Runtime in terminalbench/tasks/arn_by_image.json
```

- `VARIANT=terminalbench` keeps its own `deploy/runtime.terminalbench.env`, so the `cpu`/`libs` ARNs in
  `deploy/runtime.env` stay intact. `5_run.sh` needs no `VARIANT`: `image_routing.py` reads the pool
  map itself.
- `terminalbench/tasks/` (the fetched corpus + pool map) is gitignored. Deploy knobs: `TB_LIMIT` /
  `TB_TASKS` (subset), `SKIP_BUILD=1`, `TB_WORKDIR` (default `/app`), `CONTINUE_ON_ERROR=1`.
- **Rebuild caveat.** `deploy_terminalbench.py` tags each image with a hash of its server layer, which
  includes `agentcore_slime_demo/sandbox_server_agent.py`. The merge made that file this demo's
  version, so the next Terminal-Bench deploy rebuilds every image once (~1 h). The deployed pool keeps
  working until then.
- `terminalbench/probe_coldstart.py` splits each cold open into pull+boot vs ready.

### Warm-start probe (`terminalbench/warmstart/`)

A host-only probe (no GPU, no trainer; ~7 min) that isolates pre-open and heartbeat. It runs 24
MockPolicy episodes in batches of 6 against a stdlib Runtime with a 60 s idle timeout, plus the
Code Interpreter. Run it with `bash terminalbench/warmstart/run_warmstart.sh`; `BACKENDS=local` runs it
offline and `… cleanup` deletes the Runtime. From the recorded run (2026-09-25):

- **Pre-open removes the boot from the critical path.** Wait p50 on the own Runtime dropped from 774 ms
  to 5 ms, makespan was −20 %, and only the first batch waited.
- **Heartbeat only helps your own Runtime.** 3/3 sessions survived 150 s idle with pings. Without
  pings, 0/3 survived, and a reclaimed session silently returns a fresh, empty sandbox. The Code
  Interpreter expires at its session timeout regardless of pings.

The first AWS smoke was ~95 % idle with 64 sessions ahead, so try `TB_PREOPEN_BATCHES=1`.

## `SCALE=1`: the harness-scaling run (was `demo_cpu_scale_lanes`)

All three tiers turned up together: **E=4 SGLang engines (TP1) : W=256 harness workers : S=256
sessions in flight**, each session carrying its own mini_claude loop.

```bash
DATASET=deepscaler bash 1_prepare_data.sh   # the only dataset the preset needs
SCALE=1 bash 5_run.sh                       # DATASET=deepscaler RB=32 (x NS=8 = 256/step) GPUS_PER_ENGINE=1
                                            # HARNESS_WORKERS=256 CONC=ceil(256/4)=64 -> S=256
SCALE=1 SANDBOX_SESSIONS=64 bash 5_run.sh   # 64 pooled sessions: expect sandbox_wait_ms > 0
```

Any knob still overrides the preset. The ckpt gets the `_scale` suffix and the log tag `-scale`. Read
which tier binds from the `[demo]` records: `qdepth` > 0 = harness-bound, `sandbox_wait_ms` > 0 =
sandbox-bound, `engine_inflight` climbing = inference-bound. The earlier results of this run (the 2026-09-26 and
2026-09-28 256-worker smokes, e.g. the 592-episode one) are in `logs/scale/`; background in
[`docs/harness-scaling.md`](../../../docs/harness-scaling.md).

```bash
python3 harness_scaling_demo.py      # offline, seconds, no AWS/GPU: throughput vs W, the min(...) knee
WORKERS=1,2,4,8 EPISODES=24 CINF=4 CSBX=8 python3 harness_scaling_demo.py   # knobs: WORKERS EPISODES CINF CSBX TINF TSBX
```

## `HEARTBEAT`: keep idle Runtime sessions warm across the weight-sync barrier (was `demo_cpu_heartbeat`)

While slime syncs weights, generation stops and sandbox sessions sit idle. On our own Runtime an idle
session is reclaimed after its idle timeout, and a reclaimed session id silently comes back empty.

- **What it does.** `HEARTBEAT` (default **on**; `HEARTBEAT=0` is the A/B
  baseline) wraps every episode's sandbox in `KeptSandbox`. A process-wide `SessionKeeper`
  (`session_keeper.py`) then pings any session that has been idle for more than
  `HEARTBEAT_IDLE × HEARTBEAT_PING_FRACTION` s (600 × 0.5) with `exec("true")`. Each ping logs
  `[demo] heartbeat_session=…`.
- New Runtimes are also created with `RUNTIME_IDLE_TIMEOUT=3600` as a backstop.

```bash
python3 heartbeat_demo.py                      # offline ~30 s: naive loss -> heartbeat survives -> recover
HEARTBEAT=0 DATASET=deepscaler bash 5_run.sh   # the no-heartbeat baseline
```

## Knobs (env vars, defaults from `run.sh`)

| Knob | Default | Meaning |
|---|---|---|
| `DATASET` (alias `FAMILY`) | `mbpp,humaneval,gsm8k,math500,deepscaler,gretelsql,hotpotqa,xlam,mcp,terminalbench` | dataset, comma list to mix (task-interleaved batches), `mixed` or `libs` |
| `SANDBOX` | `agentcore-runtime` | the only backend. `local` and `agentcore` (Code Interpreter) are rejected |
| `IMAGES` | `cpu:… libs:…` | images built/deployed by steps 2–4 (`deploy/demo.env`) |
| `RB` / `NS` / `NR` / `TEMP` | 4 / 8 / 11 / 1.0 | prompts/step (raised to the task count K of a mix if unset: default 5), GRPO group, 10 train + 1 eval, temperature |
| `CONC` | `4` (`SCALE=1`: ⌈W/E⌉) | concurrent episodes per engine; `S = CONC × engines`. Also the fully-async buffer depth: staleness grows with `CONC` |
| `GPUS_PER_ENGINE` | `2` (`SCALE=1`: 1) | TP per SGLang engine; engines E = `ROLLOUT_GPUS / GPUS_PER_ENGINE` |
| `HARNESS_WORKERS` | empty → RB×NS | harness tier: dispatch workers pulling from one queue |
| `SANDBOX_SESSIONS` | `0` | 0 = fresh session per episode; M > 0 = M pooled sessions per image, fresh `/workspace/ep-<hex>` per episode (not for tasks that change the OS) |
| `PREOPEN` | RB×NS (one step) | fresh sessions kept booted ahead per image for the non-Terminal-Bench rows (still one episode each, heartbeated while waiting); `0` = open on demand |
| `SANDBOX_OPEN_RATE` | `20` | session opens per second (pacing) |
| `SCALE` | `0` | `1` = the harness-scaling preset (see above) |
| `EVAL_INTERVAL` / `EVAL_NS` / `SAVE_INTERVAL` | 10 / 2 / 1000000 | eval cadence / samples / save only at the end |
| `AGENTCORE_SANDBOX_RUNTIME_ARNS` | step 4, else baked K=2 | the `cpu` Runtime(s) |
| `AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE` | step 4 | `{cpu, libs}`; `run.sh` aliases datasci/scipy/sklearn → libs |
| `HEARTBEAT` / `HEARTBEAT_IDLE` / `HEARTBEAT_PING_FRACTION` | on / 600 / 0.5 | session keeper |
| `MINI_AGENT_MAX_TURNS` / `MINI_SESSION_POLL_SEC` | 2×task turns / 25 | in-session loop turns / long-poll length |
| `ADAPTER_PORT_BASE` / `ADAPTER_PUBLIC_HOST` | 18100 / auto | where the session dials the adapter back |
| `ACTOR_GPUS` / `ROLLOUT_GPUS` | 4 / 4 | GPU split (TP2 actor, so `ACTOR_GPUS` is even) |
| `TB_PREOPEN_BATCHES` | `2` | Terminal-Bench rows: upcoming batches kept booted ahead (`× RB × NS` microVMs); `0` = look-ahead off |
| `WARM_IDLE_TIMEOUT` / `WARM_PING_FRACTION` | 900 / 0.5 | Terminal-Bench rows: the always-on keeper for fresh sessions |
| `EPISODE_MAX_TURNS` / `EPISODE_MAX_TOKENS` / `TB_EXEC_TIMEOUT` | 8 / 1024 / 180 | Terminal-Bench rows: turns / tokens per turn / seconds per bash command |
| `MAX_TOKENS_PER_GPU` | 9216 (Terminal-Bench rows: 16384) | slime `--max-tokens-per-gpu` |
| `SLIME_CUDA_VISIBLE` / `SLIME_NUM_GPUS` | unset / 8 | pin ray to a GPU subset on a shared box (e.g. `2,3,4,5` / `4`) |

Only knobs that match `PASS_RE` in `deploy/steps_common.sh` are forwarded by `docker exec`.

To train to convergence, run inside the container:

```bash
DATASET=gsm8k STEPS=50 EVAL_EVERY=10 SANDBOX_CONC=4 ./train.sh
```

`STEPS` maps to `NR=STEPS+1`, `EVAL_EVERY` to `EVAL_INTERVAL` and `SANDBOX_CONC` to `CONC`.
`SANDBOX_RUNTIMES` caps K. Keep the train set at `>= NR×RB` prompts so the run stays in one epoch.

## Metrics

`5_run.sh` runs `metrics.py` on its log and writes `logs/<tag>.{csv,eval.csv,png}`, plus the raw
per-op files `<tag>.sandbox_ops.csv` and `<tag>.sglang_gen.csv`. Watch these:

- `rollout/raw_reward`, the learning signal.
- `eval/<dataset>`, the held-out solve-rate.
- `perf/sandbox_ms_{mean,p50,p95}` and `perf/sandbox_count`, the sandbox round-trips (from the trainer:
  runner start, long-polls, grading).
- `perf/sglang_gen_ms_*`, pure decode time.
- `harness_where=session` on every `[demo]` episode line, plus the tier fields `worker`, `qdepth`,
  `sandbox_wait_ms`, `engine`, `engine_inflight` (and `sandbox_slot` / `session_episode` when pooled).
- The stickiness check (`router.csv`, now with `sandbox_slot`): sessions and engines per worker, workers
  per session. A worker using several engines or sessions is expected, not a violation.

## Folder layout (self-contained)

- `agentcore_slime_demo/`: the vendored shared lib. `run.sh` puts `$DEMO_DIR` on `PYTHONPATH`. Key
  pieces: `generate.py` (the slime hook), `harness_pool.py` (the harness tier), `mini_claude_episode.py` (episode + hidden grader),
  `in_session_harness.py` + `session_runner.py` (the loop in the session), `task_families.py`,
  `build_real_datasets.py` and `real_datasets.py` (HF loaders).
- `1_…6_*.sh`: the numbered steps. Settings are in `deploy/demo.env` and shared logic in
  `deploy/steps_common.sh`.
- `deploy/`: `Dockerfile.sandbox.cpu`, `Dockerfile.sandbox.libs` and `agentcore_deploy.py`.
- `image_routing.py`: the per-image ARN map + the fail-fast check (called by `run.sh`; `--families`).
- `terminalbench/`: `fetch_terminalbench.py`, `deploy_terminalbench.py` (+ `acs_ctx/`, the per-task
  image layer), `validate_oracle.py`, `build_terminalbench_dataset.py`, `probe_coldstart.py`,
  `warmstart/`, and the gitignored `tasks/`. The pool itself is `agentcore_slime_demo/terminalbench_pool.py`
  + `tb_lookahead_source.py`.
- `prepare_data.sh`: download and prepare the data (step 1).
- `verify_data.py`: an offline checker proof (the reference PASSES, a wrong answer FAILS).
- `build_dataset.py` (MBPP), `build_family_datasets.py` (zoo), `build_mixed_datasets.py` (`mixed`, `libs`).
- `data/`: `*_train.jsonl` / `*_eval.jsonl` (incl. `libs_*`, `mixed_*`, `terminalbench_*`).
- `run.sh`: the RL run. `train.sh`: the convergence wrapper. `metrics.py`: parsing, CSVs, PNG, stickiness.
- Probes: `compare_backends.py`, `netcheck.py` (safe-networking preflight, `docs/agentcore-networking.md`), `heartbeat_demo.py` (+ `agentcore_idle_sim.py`), `harness_scaling_demo.py`,
  `smoke_remote_families.py`.
- `logs/{imagepool,heartbeat,customruntime,scale}/`: logs from the merged folders (`scale/` = the
  former `demo_cpu_scale_lanes`). The `bigimage` logs went with
  `demo_cpu_instance_pool` (removed 2026-09-28; in git history).

Prereqs: the `agentcore_p4` container (repo at `/work/code`), `AWS_REGION=us-west-2` and AWS creds,
the base checkpoint `code/p4_slime_train/ckpt/Qwen3-4B_torch_dist` (`CKPT_BASE`), and the Qwen3-4B HF
snapshot in the container cache.
