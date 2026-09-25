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
- **Launch:** `2_build_image.sh` → `3_push_ecr.sh` → `4_deploy_runtime.sh` (vendored
  `deploy/agentcore_deploy.py`) creates the demo's own Runtimes. The sandbox comes up serverless — a
  session is created lazily on the first `invoke_agent_runtime`. There is no fixed cold-start number
  for the CPU sandbox.
- **Terminate:** `6_cleanup.sh` deletes the Runtimes it created. `run.sh` itself does NO runtime
  teardown; it only cleans up local processes (`pkill -9 sglang`, `ray stop`).
- **Latency:** each step's train CSV carries the sandbox round-trip time
  `perf/sandbox_ms_{mean,p50,p95}` (with `perf/sandbox_count` = number of `invoke_agent_runtime`
  round-trips) and, split out separately, pure SGLang decode time `perf/sglang_gen_ms_{mean,p50,p95}`.
  Because the harness is in-process, **both** latency splits are fully present in `train.log`.
  `metrics.py` also writes raw per-op CSVs `<log>.sandbox_ops.csv` and `<log>.sglang_gen.csv`.

## Variant: `VARIANT=bigimage`: a large image on a CPU Instance (was `demo_cpu_bigimage`)

The Code Interpreter microVM caps the image at ~2 GB, so a heavy checker environment (torch, a full
scientific stack) won't fit. The escape hatch is **purely a deployment change**: build a large custom
Runtime image (`deploy/Dockerfile.sandbox.bigcpu`, ~4 GB) and run it on an AgentCore
**capacity-provider Instance** (`CPU_INSTANCE`, default `m5.xlarge`). The harness, reward path and slime
args are unchanged. Default task is `DATASET=deepscaler`; any dataset works.

| | Code Interpreter microVM | `VARIANT=bigimage`: Runtime on a CPU Instance |
|---|---|---|
| Image cap | ~2 GB | instance root disk (`ROOT_FREE_GIB`, default 40 GB) |
| Heavy deps (torch, scipy, sklearn) | won't fit | baked in |
| Compute | AWS-managed microVM | your EC2 family (`CPU_INSTANCE`) |
| Session length | ~8 h | 14 days |

The variant runs **warm-shared** (`SANDBOX=agentcore-ray-farm`). All episodes share one
`runtimeSessionId` (`B4_SHARED_RUNTIME_SESSION`, minted per run), so they land on the same warm m5
and the ~60–80 s cold-boot is paid **once per run**. The server-side `SANDBOX_FARM=cpu` farm
(`sandbox_cpu_farm.py`) gives every per-episode `sub_session` a fresh workdir plus its own
`HOME`/`TMPDIR`/caches, all wiped on close. `SANDBOX=agentcore-runtime` falls back to one fresh
instance per episode, which pays a cold-boot every time. See
[`docs/sandbox-fresh-env-on-persistent-instance.md`](../../../docs/sandbox-fresh-env-on-persistent-instance.md).

```bash
python3 probe_bigimage.py                  # offline: image size vs the microVM cap + LocalSandbox contract parity
python3 netcheck.py                        # offline: safe-networking checklist (see docs/agentcore-networking.md)
VARIANT=bigimage bash 2_build_image.sh     # host: ~4 GB amd64 image
VARIANT=bigimage bash 3_push_ecr.sh        # host: -> ECR acs-sandbox-bigcpu
VARIANT=bigimage bash 4_deploy_runtime.sh  # host: CPU capacity provider acs_cpu_cp_bigimage + Runtime -> deploy/runtime.bigimage.env
VARIANT=bigimage bash 5_run.sh             # container: 10 train steps + 1 eval on the warm m5
VARIANT=bigimage bash 6_cleanup.sh         # host: delete the Runtime + the CP
```

Networking stays pull-only (`networkMode=PUBLIC`, IAM `invoke_agent_runtime`, no inbound port).
Deploy knobs: `CPU_INSTANCE`, `ROOT_FREE_GIB`, `FORCE_RECREATE=1`. Smoke results from the old folder
are in `logs/bigimage/`.

## Folder layout (self-contained)
- `agentcore_slime_demo/` — vendored shared library (this demo imports only this copy;
  `run.sh` puts `$DEMO_DIR` on `PYTHONPATH`, not `/work/code`), incl. `build_real_datasets.py` +
  `real_datasets.py` (the HuggingFace loaders → task rows)
- `1_…6_*.sh` — the numbered step scripts (settings in `deploy/demo.env`, shared logic in `deploy/steps_common.sh`)
- `deploy/` — `Dockerfile.sandbox.cpu`, the Dockerfile the sandbox image is built from, and
  `Dockerfile.sandbox.datasci` (the same server + pandas, for `DATASET=mixed`), and for
  `VARIANT=bigimage` `Dockerfile.sandbox.bigcpu` + `deploy_bigimage.py` (CPU capacity provider + Runtime)
- `probe_bigimage.py`, `netcheck.py` (+ `netcheck_sample_{good,bad}.json`): offline probes for the bigimage variant
- `prepare_data.sh` — **download + prepare the datasets** (step 1; all, or `DATASET=<name>`); idempotent
- `verify_data.py` — offline checker proof per real dataset (reference PASSES, wrong FAILS)
- `build_dataset.py` — the MBPP loader; `build_mixed_datasets.py` — builds `DATASET=mixed`
  (offline, stdlib); `data/` — every task's `*_train.jsonl` / `*_eval.jsonl`, plus
  `mixed_train.jsonl` / `mixed_eval.jsonl` and per-family `mixed_<family>_eval.jsonl`
- `run.sh` — the RL training run (async/decoupled slime)
- `metrics.py` — reward + latency parsing (CSVs + PNG)

## Step by step (numbered scripts; run from this dir on the HOST — container steps `docker exec` themselves)

Prereqs: the `agentcore_p4` container (repo at `/work/code`), `AWS_REGION=us-west-2` + AWS creds,
the base checkpoint `code/p4_slime_train/ckpt/Qwen3-4B_torch_dist` (`CKPT_BASE`) and the Qwen3-4B HF
snapshot in the container cache.

```bash
bash 1_prepare_data.sh     # 1. container: prepare_data.sh (all datasets, or DATASET=<name>) + verify_data.py
bash 2_build_image.sh      # 2. arm64 images cpu (Dockerfile.sandbox.cpu) + datasci (Dockerfile.sandbox.datasci)
bash 3_push_ecr.sh         # 3. -> ECR acs-tasks-cpu, acs-tasks-datasci (deploy/images.env)
bash 4_deploy_runtime.sh   # 4. Runtimes acs_tasks_cpu, acs_tasks_datasci -> deploy/runtime.env
                           #    (ARNS = cpu; ARN_BY_IMAGE {cpu,datasci} for DATASET=mixed)
bash 5_run.sh              # 5. container: 10 train steps + 1 eval; logs/<tag>.{log,csv,eval.csv,png}
bash 6_cleanup.sh          # 6. delete the Runtimes (ECR repos, data, logs, ckpt are kept)
```

- **Single-dataset runs** only need the `cpu` image: prefix steps 2/3/4/6 with
  `IMAGES=cpu:Dockerfile.sandbox.cpu`. `datasci` is only routed to by `DATASET=mixed`.
- **Without step 4**, `5_run.sh` falls back to the baked shared K=2 `acs_sandbox_cpu` ARNs in `run.sh`.
- **`VARIANT=bigimage`** runs the same six steps on a CPU Instance (see the section above).
- `FORCE_RECREATE=1 bash 4_deploy_runtime.sh` recreates the Runtimes.

Pick the dataset at run time; the file prefix == `DATASET`, so nothing else changes:
```bash
DATASET=gsm8k bash 5_run.sh                       # easiest — good first run
DATASET=gsm8k,gretelsql,hotpotqa bash 5_run.sh    # ad-hoc mix, eval/mix (no build step)
DATASET=mixed bash 5_run.sh                       # prebuilt math+sql+ctf+datasci, eval/mixed
FAMILIES=math,sql,swe,rag DATASET=mixed bash 1_prepare_data.sh   # rebuild the prebuilt mix
```
Only knobs matching `PASS_RE` in `deploy/steps_common.sh` are forwarded by `docker exec`; add new ones there
(all demo copies are identical).
Keep the train set `>= NR*RB` prompts so one epoch spans the whole run (save-only-at-end holds).
`verify_data.py [names…]` re-checks the checkers offline (reference PASSES, a wrong one FAILS).

### Knobs (env vars, defaults from `run.sh`)
| Knob | Default | Meaning |
|---|---|---|
| `DATASET` (alias `FAMILY`) | `mbpp` | dataset, comma list to mix, or `mixed` |
| `SANDBOX` | `agentcore-runtime` | remote Runtime; `agentcore` = Code Interpreter. `local` is rejected |
| `MODE` | `B` | fully-async (staleness grows with `CONC`); `A` = 1-step-async |
| `RB` / `NS` / `NR` / `TEMP` | 4 / 8 / 11 / 1.0 | prompts/step, GRPO group, 10 train + 1 eval, temperature |
| `CONC` | `4` | concurrent episodes `S = CONC×2`; Mode B buffer depth |
| `INFER_CONC` | `0` (off) | caps decode concurrency while sweeping `CONC` |
| `EVAL_INTERVAL` / `EVAL_NS` / `SAVE_INTERVAL` | 10 / 2 / 1000000 | eval cadence / samples / save-only-at-end |
| `AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE` | from `deploy/runtime.env` | image → ARN map; only `DATASET=mixed` needs it |

Runtime-count ablation (`RUNTIMES={1 2 4}`): `./run_runtime_ablation.sh`. Train to convergence
(inside the container): `DATASET=mbpp STEPS=50 EVAL_EVERY=10 SANDBOX_CONC=4 SANDBOX_RUNTIMES=2 ./train.sh`
(`STEPS` → `NR=STEPS+1`, `EVAL_EVERY` → `EVAL_INTERVAL`, `SANDBOX_CONC` → `CONC`,
`SANDBOX_RUNTIMES` → K). For a mix, `eval/mix` can rise while one family lags; the `[demo]` per-family
signal in the log shows each family.

### Key metrics
`rollout/raw_reward` (learning signal), `train/grad_norm`, `eval/<dataset>`, and the
`perf/sandbox_ms_*` vs `perf/sglang_gen_ms_*` split (is the sandbox or SGLang the bottleneck).
`metrics.py` also writes the raw per-op CSVs `<log>.sandbox_ops.csv` and `<log>.sglang_gen.csv`.
