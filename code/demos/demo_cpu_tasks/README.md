# demo_cpu_tasks — one demo, any RLVR dataset or MIX of datasets (`DATASET=`), CPU AgentCore sandbox (in-process harness)

This single folder replaces two former demos that ran the same harness, sandbox and trainer and
differed only in their data:

- `demo_cpu_datasets` — one dataset per run (itself the merge of ten per-dataset folders
  `demo_cpu_mbpp`, `demo_cpu_math`, `demo_cpu_gsm8k` … `demo_cpu_mcp`);
- `demo_cpu_mixtask` — one batch spanning several task families.

A single dataset is just a one-element mix, so both are now one knob:

| `DATASET=` | trains | eval metric | ckpt |
|---|---|---|---|
| `mbpp` (default), `gsm8k`, … | one dataset (table below) | `eval/<name>` | `ckpt/Qwen3-4B_<name>_async` |
| `gsm8k,gretelsql,hotpotqa` | an ad-hoc mix: `run.sh` interleaves those datasets' files round-robin | `eval/mix` | `ckpt/Qwen3-4B_gsm8k+gretelsql+hotpotqa_async` |
| `mixed` | the prebuilt math + sql + ctf + datasci mix (the former `demo_cpu_mixtask`) | `eval/mixed` | `ckpt/Qwen3-4B_mixed_async` |

An agentic-RL loop that RL-trains Qwen3-4B to solve verifiable tasks (default: MBPP Python problems) by **writing `solution.py` →
running the hidden unit tests → reading the failure → fixing → repeat**. The AgentCore Runtime is
the **RL environment**: each turn's `write_file` / `exec` / `read_file` is executed inside a managed
serverless sandbox Runtime, reached over `invoke_agent_runtime`, while the multi-turn `run_episode`
harness and the slime trainer stay on the training box.

## What this demo shows
- **Task:** multi-turn "write `solution.py` → run hidden checker → fix" over a **zoo of real,
  verifiable-reward (RLVR) datasets** pulled from HuggingFace. The in-process harness is
  task-agnostic — the SAME `run_episode` loop trains any of them; only the seed dataset + hidden
  checker differ. Select one with `DATASET=<name>` (the file prefix; `FAMILY` is an accepted alias);
  no other change. Each dataset writes its own `eval/<name>` metric and `ckpt/Qwen3-4B_<name>_async`:

  | `DATASET`     | family    | dataset (HuggingFace)                          | train/eval | reward |
  |---------------|-----------|------------------------------------------------|-----------:|--------|
  | `mbpp` (default; alias `coding`) | coding | MBPP (`mbpp`, via `build_dataset.py`) | 500/8 | hidden asserts |
  | `gsm8k`       | math      | `openai/gsm8k` (config `main`)                 |    200/40  | exact integer match |
  | `math500`     | math      | `HuggingFaceH4/MATH-500` (numeric subset)      |    160/40  | numeric match (tol 1e-4) |
  | `deepscaler`  | math      | `agentica-org/DeepScaleR-Preview-Dataset`      |    400/40  | numeric match (tol 1e-4) |
  | `humaneval`   | coding    | `openai/openai_humaneval`                      |    140/24  | row's own `test` fn |
  | `gretelsql`   | sql       | `gretelai/synthetic_text_to_sql`               |    300/40  | SQL execution vs gold result set |
  | `hotpotqa`    | rag       | `hotpotqa/hotpot_qa` (distractor)              |    300/40  | SQuAD-normalized exact match |
  | `xlam`        | toolcall  | `minpeter/xlam-function-calling-60k-parsed`    |    300/40  | order-insensitive gold-call match |
  | `mcp`         | toolcall  | `Team-ACE/ToolACE` (MCP-style tool schemas)    |    300/40  | order-insensitive gold-call match |

  The synthetic task-family zoo from `build_family_datasets.py` (`math` `swe` `tool_use` `sql` `ctf`
  `rag` `regex` `config` `data` `planning`) is selectable the same way. `math` is the one the
  former `demo_cpu_math` defaulted to; the rest are tiny (4–9 rows) and meant for offline smoke tests.
  An unknown `DATASET` fails fast and lists what is in `data/`.
- **Mixing datasets:** each row is self-describing
  (`metadata.task = {family, task_id, spec, …}`), and `generate._task_from_sample` rebuilds the right
  task (seed files + hidden checker + reward) **per row**, so a batch that spans families needs no
  harness or sandbox change. GRPO normalizes advantage **within** a group of `NS` repeats of the same
  prompt (one family); mixing happens **across** groups, so the families' reward scales never need
  reconciling. `DATASET=a,b,c` truncates each dataset to the shortest one and interleaves them
  (`a0 b0 c0 a1 b1 c1 …`, train and eval alike) into `ckpt/_mix/`, so every step spans all of them.
  The sandbox session count (one per episode) is unchanged by mixing.
- **The one custom-image family:** `DATASET=mixed` includes `datasci`, whose checker needs pandas —
  absent from the generic CPU image. Each task carries an `image` tag (`cpu` by default, `datasci`
  here) and the sandbox client routes an episode to the Runtime for its image via
  `AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE='{"cpu": "<arn>", "datasci": "<arn>"}'`. Build that
  Runtime from `deploy/Dockerfile.sandbox.datasci`. Unset, datasci episodes fall back to the CPU
  pool, fail to import pandas and score 0. (For a pool with one image per family, see
  `demo_cpu_imagepool`.)
- **Reward:** the candidate's answer is run against the dataset's hidden checker (for MBPP: its asserts); the
  learning signal is `rollout/raw_reward`, and held-out **solve-rate** is the key eval metric. GRPO
  advantage is 0 when every sample in a prompt's group gets the same reward, so reward variance
  (higher `TEMP` / harder tasks) is what produces gradient.
- **Sandbox arrangement:** ONE CPU-backed AgentCore Runtime sandbox (`sandbox_server_agent.py`).
  Every checker is plain Python/IO, so the sandbox is CPU-only and all 8 L40S GPUs serve the
  policy (**4 train + 4 inference + 0 env**). The harness runs **in-process** on the training box
  via `--custom-generate-function-path agentcore_slime_demo.generate.generate`; there is NO separate
  harness Runtime. Trainer is real slime, async/decoupled (`train_async.py`, no `--colocate`).
  A fleet variant fans episodes round-robin across several sandbox Runtimes for the runtime-count
  throughput ablation.

## AgentCore basics: launch / terminate / latency
- **Launch:** the shared deploy orchestrator (`code/deploy/deploy_two_runtime.py`) builds/pushes the
  image, provisions IAM/roles, and creates the base `acs_sandbox_cpu` Runtime. The sandbox comes up
  serverless — a session is created lazily on the first `invoke_agent_runtime`. There is no fixed
  cold-start number for the CPU sandbox.
- **Terminate:** `python3 deploy_two_runtime.py cleanup` (from `code/deploy/`) deletes
  `acs_sandbox_cpu`. **Fleet caveat:** extra fleet runtimes `acs_sandbox_cpu_1..N` are NOT deleted by
  cleanup — delete them manually via the control-plane `delete_agent_runtime` API. The demo `run.sh`
  does NO runtime teardown; it only cleans up local processes (`pkill -9 sglang`, `ray stop`).
- **Latency:** each step's train CSV carries the sandbox round-trip time
  `perf/sandbox_ms_{mean,p50,p95}` (with `perf/sandbox_count` = number of `invoke_agent_runtime`
  round-trips) and, split out separately, pure SGLang decode time `perf/sglang_gen_ms_{mean,p50,p95}`.
  Because the harness is in-process, **both** latency splits are fully present in `train.log`.
  `metrics.py` also writes raw per-op CSVs `<log>.sandbox_ops.csv` and `<log>.sglang_gen.csv`.

## Folder layout (self-contained)
- `agentcore_slime_demo/` — vendored shared library (this demo imports only this copy;
  `run.sh` puts `$DEMO_DIR` on `PYTHONPATH`, not `/work/code`), incl. `build_real_datasets.py` +
  `real_datasets.py` (the HuggingFace loaders → task rows)
- `deploy/` — `Dockerfile.sandbox.cpu`, the Dockerfile the sandbox image is built from, and
  `Dockerfile.sandbox.datasci` (the same server + pandas, for `DATASET=mixed`)
- `prepare_data.sh` — **download + prepare the datasets** (step 1; all, or `DATASET=<name>`); idempotent
- `verify_data.py` — offline checker proof per real dataset (reference PASSES, wrong FAILS)
- `build_dataset.py` — the MBPP loader; `build_mixed_datasets.py` — builds `DATASET=mixed`
  (offline, stdlib); `data/` — every task's `*_train.jsonl` / `*_eval.jsonl`, plus
  `mixed_train.jsonl` / `mixed_eval.jsonl` and per-family `mixed_<family>_eval.jsonl`
- `run.sh` — the RL training run (async/decoupled slime)
- `metrics.py` — reward + latency parsing (CSVs + PNG)

## Step by step

### 0. Prereqs
- Run everything INSIDE the persistent container `agentcore_p4` (slimerl/slime); the repo is at
  `/work/code`, this demo at `/work/code/demos/demo_cpu_tasks`.
- Region `us-west-2` (`AWS_REGION`), plus AWS creds (`AWS_BEARER_TOKEN_BEDROCK`) for the
  `invoke_agent_runtime` calls.
- Shared, NOT vendored (reference by path): the deploy orchestrator under `code/deploy/`, and the
  base Megatron checkpoint at `code/p4_slime_train/ckpt/Qwen3-4B_torch_dist` (override with
  `CKPT_BASE`). The Qwen3-4B HF snapshot must be in the container HF cache (baked into `run.sh` as
  `$HF`).

### 1. Download + prepare the datasets
`data/` is already populated; step 1 re-pulls the whole zoo from scratch (one-time network for the
HF rows). One idempotent script builds every dataset's train+eval JSONL, or just one:
```bash
bash prepare_data.sh                  # all nine real datasets (row counts in the table above)
DATASET=gsm8k bash prepare_data.sh    # just one
DATASET=mixed bash prepare_data.sh    # the prebuilt mix (offline); change it with FAMILIES / BALANCED / REPEAT
FAMILIES=math,sql,swe,rag DATASET=mixed bash prepare_data.sh
```
Ad-hoc mixes (`DATASET=a,b,c`) need no build step — `run.sh` interleaves the per-dataset files.
It calls the two vendored builders — `build_dataset.py` (MBPP) and
`agentcore_slime_demo/build_real_datasets.py` (the HF zoo) — with the exact splits/limits/offsets
that produced the on-disk files (see the script header). Each file's prefix == the `DATASET` label,
so training needs no `DATA`/`EVAL_DATA` overrides. Keep the train set `>= NR*RB` prompts so one epoch
spans the whole run (e.g. 500 ≥ 11×4) — that keeps save-only-at-end holding.

Verify the checkers offline before any GPU run (reference solution PASSES, a wrong one FAILS):
```bash
python3 verify_data.py                # every real dataset
python3 verify_data.py gsm8k xlam     # just these
```

### 2. Build the sandbox image
Build from INSIDE this demo dir so `COPY agentcore_slime_demo/...` resolves against the vendored
copy. Platform `linux/arm64`, ECR repo `acs-sandbox-cpu`. No torch/prepare step (CPU):
```bash
docker build --platform linux/arm64 -f deploy/Dockerfile.sandbox.cpu .
```

### 3. Launch the Runtime
From `code/deploy/` (the shared orchestrator handles ECR/IAM/roles):
```bash
python3 deploy_two_runtime.py all        # creates the base acs_sandbox_cpu runtime
```
For the runtime-count fleet (ablation only):
```bash
FLEET_N=4 python3 deploy_cpu_sandbox_fleet.py    # prints the export line for ARNS below
```

### 4. Export the ARN(s)
Single Runtime:
```bash
export AGENTCORE_SANDBOX_RUNTIME_ARN=<arn>
```
Fleet (plural, comma-separated, round-robin across endpoints):
```bash
export AGENTCORE_SANDBOX_RUNTIME_ARNS=<arn0>,<arn1>,...
```
`run.sh` requires either the plural `AGENTCORE_SANDBOX_RUNTIME_ARNS` or the single
`AGENTCORE_SANDBOX_RUNTIME_ARN` when `SANDBOX=agentcore-runtime` (the default). It ships pinned to a
K=2 default ARNS list; override by exporting your own.

### 5. Run RL training
Pick a dataset with `DATASET` (defaults to `mbpp`); the file prefix == `DATASET`, so nothing else
changes between datasets:
```bash
DATASET=mbpp       ./run.sh 2>&1 | tee train.log   # MBPP coding (default)
DATASET=gsm8k      ./run.sh 2>&1 | tee train.log   # grade-school math (easiest — good first run)
DATASET=math500    ./run.sh 2>&1 | tee train.log   # competition math
DATASET=deepscaler ./run.sh 2>&1 | tee train.log   # olympiad math
DATASET=humaneval  ./run.sh 2>&1 | tee train.log
DATASET=gretelsql  ./run.sh 2>&1 | tee train.log   # text-to-SQL
DATASET=hotpotqa   ./run.sh 2>&1 | tee train.log   # multi-hop RAG
DATASET=xlam       ./run.sh 2>&1 | tee train.log   # function calling
DATASET=mcp        ./run.sh 2>&1 | tee train.log   # MCP-style tool calling (ToolACE)

# mixes — one batch spans several families
DATASET=gsm8k,gretelsql,hotpotqa ./run.sh 2>&1 | tee train.log   # ad-hoc mix, eval/mix
AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE='{"cpu":"<cpu arn>","datasci":"<pandas arn>"}' \
DATASET=mixed      ./run.sh 2>&1 | tee train.log   # prebuilt math+sql+ctf+datasci, eval/mixed
```
Key knobs (env vars, defaults from `run.sh`):
- `SANDBOX` (`agentcore-runtime`) — remote Runtime; `agentcore` = Code Interpreter microVM.
  `local` is rejected (CPU demos are remote-only).
- `MODE` (`B`) — fully-async (deep buffer, staleness grows with `CONC`); `A` = 1-step-async.
- `RB` (`4`) prompts/step, `NS` (`8`) GRPO group size (`GBS=RB×NS`), `NR` (`11`) num-rollout ⇒
  real steps = `NR-1` = 10 train + 1 eval (smoke).
- `TEMP` (`1.0`) rollout temperature.
- `CONC` (`4`) sglang-server-concurrency ⇒ concurrent episodes `S = CONC×2`; in Mode B it also sets
  buffer depth (the sandbox-concurrency ablation knob). `INFER_CONC` (`0`=off) caps decode
  concurrency to hold inference load constant while sweeping `CONC`.
- `EVAL_INTERVAL` (`10`), `EVAL_NS` (`2`), `SAVE_INTERVAL` (`1000000`, save-only-at-end).
- `DATASET` (`mbpp`; alias `FAMILY`) — selects the dataset (data-file prefix + eval label + ckpt
  dir), a comma list to mix, or `mixed`. See the tables up top.
- `AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE` (unset) — JSON image-tag → Runtime ARN map; only
  `DATASET=mixed` (its `datasci` family) needs it.

Runtime-count ablation (sweeps `RUNTIMES={1 2 4}`): `./run_runtime_ablation.sh`.

**One-command train-to-convergence.** `train.sh` wraps `run.sh` with three top-level knobs —
training **steps**, eval **cadence**, and the two **sandbox** counts — passing `DATASET` (and
everything else) through:
```bash
DATASET=mbpp STEPS=50 EVAL_EVERY=10 SANDBOX_CONC=4 SANDBOX_RUNTIMES=2 ./train.sh 2>&1 | tee train.log
```
`STEPS` → `NR=STEPS+1` (real GRPO updates; `STEPS=2` = smoke); `EVAL_EVERY` → `EVAL_INTERVAL`;
`SANDBOX_CONC` → `CONC` (concurrent episodes, `S=CONC×2`); `SANDBOX_RUNTIMES` → # distinct sandbox
Runtimes (K, truncates the ARN list). Convergence is dataset-dependent — watch `eval/<dataset>` and raise
`STEPS`. For a mix, the aggregate `eval/mix` can rise while one family lags; the `[demo]` per-family
signal in the log shows each family's reward.

### 6. Read metrics / latency
```bash
python3 metrics.py train.log \
    --csv train.csv --eval-csv train.eval.csv --plot train.png
```
Produces a per-step train CSV, a per-eval held-out solve-rate CSV, and a multi-panel PNG, plus the
raw per-op latency CSVs `train.log.sandbox_ops.csv` and `train.log.sglang_gen.csv` and a run-level
per-op latency summary. Watch `rollout/raw_reward` (learning signal), `train/grad_norm` (policy
moving), and the `perf/sandbox_ms_*` vs `perf/sglang_gen_ms_*` split (is the remote sandbox or the
trainer/SGLang the bottleneck).

### 7. Terminate
```bash
cd code/deploy && python3 deploy_two_runtime.py cleanup   # deletes acs_sandbox_cpu
```
**Fleet caveat:** `cleanup` does NOT delete the extra fleet runtimes `acs_sandbox_cpu_1..N` — delete
those manually via the control-plane `delete_agent_runtime` API. `run.sh` itself tears down nothing
remote (only local `pkill sglang` / `ray stop`).
