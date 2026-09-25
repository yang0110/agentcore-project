# demo_cpu_tasks — literature RLVR tasks on the DEFAULT AgentCore sandbox (managed Code Interpreter, no image); `DATASET=` any task or mix, `HEARTBEAT=1`

**The point of this demo: most agentic-RL tasks from the literature train on the default AgentCore
runtime as-is.** `SANDBOX=agentcore` (the default) runs every episode in a managed **AgentCore Code
Interpreter** session: no Dockerfile, no ECR push, no Runtime to deploy or delete. You prepare the data
and run the trainer; steps 2/3/4/6 of the step scripts are no-ops.

An agentic-RL loop RL-trains Qwen3-4B to solve verifiable tasks by **writing `solution.py` → running
the hidden checker → reading the failure → fixing → repeat**. Each turn's `write_file` / `exec` /
`read_file` runs inside a remote AgentCore sandbox session. The multi-turn `run_episode` harness and
the slime trainer stay on the training box (in-process, via
`--custom-generate-function-path agentcore_slime_demo.generate.generate`). The trainer is real slime,
async/decoupled (`train_async.py`), 4 train + 4 inference GPUs, 0 GPUs for the env.

This folder also absorbed several former demos:

- `demo_cpu_datasets` and `demo_cpu_mixtask`: now `DATASET=<name>` / `DATASET=a,b,c` / `DATASET=mixed`.
- `demo_cpu_imagepool`: its pandas/scipy/sklearn tasks are now `DATASET=libs`, **on the default Code
  Interpreter**. That service already has those libraries, so the three custom images are gone.
- `demo_cpu_heartbeat`: now `HEARTBEAT=1`.
- `demo_cpu_customruntime`: now `compare_backends.py`.
- `VARIANT=bigimage`: moved out to `demo_cpu_instance_pool`. Custom images at scale are in
  `demo_cpu_terminalbench`.

To scale the rollout as N 1:1:1 lanes (harness worker : SGLang engine : persistent sandbox session),
see the standalone `demo_cpu_scale_lanes`.

## 1. The tasks: dataset, source, environment, image

Every row is self-describing (`metadata.task = {family, task_id, spec, …}`). `generate._task_from_sample`
rebuilds the right task (seed files, hidden checker and reward) for each row, so the same harness trains
all of them. Select a dataset with `DATASET=<file prefix>` (`FAMILY` is an alias). Each dataset writes
its own `eval/<name>` metric and its own `ckpt/Qwen3-4B_<name>_async`.

| `DATASET` | family | dataset (URL) | train/eval | reward | env deps (checker) | Docker image? |
|---|---|---|---:|---|---|---|
| `mbpp` (default) | coding | [google-research-datasets/mbpp](https://huggingface.co/datasets/google-research-datasets/mbpp) | 500/8 | hidden asserts | python stdlib | **no**: Code Interpreter |
| `humaneval` | coding | [openai/openai_humaneval](https://huggingface.co/datasets/openai/openai_humaneval) | 140/24 | the row's own `test` fn | python stdlib | **no** |
| `gsm8k` | math | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) (`main`) | 200/40 | exact integer match | python stdlib | **no** |
| `math500` | math | [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) (numeric subset) | 160/40 | numeric match, tol 1e-4 | python stdlib | **no** |
| `deepscaler` | math | [agentica-org/DeepScaleR-Preview-Dataset](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset) | 400/40 | numeric match, tol 1e-4 | python stdlib | **no** |
| `gretelsql` | sql | [gretelai/synthetic_text_to_sql](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql) | 300/40 | SQL result set vs gold | `sqlite3` | **no** |
| `hotpotqa` | rag | [hotpotqa/hotpot_qa](https://huggingface.co/datasets/hotpotqa/hotpot_qa) (distractor) | 300/40 | SQuAD-normalized EM | files in `./docs` | **no** |
| `xlam` | tool call | [minpeter/xlam-function-calling-60k-parsed](https://huggingface.co/datasets/minpeter/xlam-function-calling-60k-parsed) | 300/40 | order-insensitive gold-call match | python stdlib | **no** |
| `mcp` | tool call | [Team-ACE/ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) (MCP-style schemas) | 300/40 | order-insensitive gold-call match | python stdlib | **no** |
| `libs` | datasci + scipy + sklearn | synthetic (`task_families.py`) | 72/6 (+2 per family) | numeric assert on `answer` | **pandas, scipy, scikit-learn** | **no**: preinstalled in the Code Interpreter |
| `mixed` | math + sql + ctf + datasci | synthetic mix | 174/9 | per family | stdlib + pandas | **no** |
| `math` `swe` `tool_use` `sql` `ctf` `rag` `regex` `config` `data` `planning` | synthetic zoo (`build_family_datasets.py`) | — | tiny (4–9) | per family | stdlib | **no** |

**Where a custom image IS needed:** the task needs an OS or toolchain that the Code Interpreter lacks.
It has no git and no internet (so no `pip install` at run time). Examples:

- Terminal-Bench, where each task pins its own OS image: `demo_cpu_terminalbench`.
- An image larger than 2 GB: `demo_cpu_instance_pool`.
- CUDA: the GPU demos.

**The Code Interpreter environment** was probed live on 2026-09-25 in us-west-2:

- Python 3.12.13 on aarch64.
- Libraries: numpy 1.26.4, pandas 2.3.1, scipy 1.16.0, scikit-learn 1.5.0, sympy, torch 2.3.0 (CPU),
  statsmodels, networkx, matplotlib, requests, sqlite3.
- Toolchains: gcc 11.5, node v24.
- **No** git, **no** internet.

A MockPolicy episode of every one of the 13 task families scored reward 1.0 on it. Rerun that check with
`SANDBOX=agentcore python3 smoke_remote_families.py`.

**Mixing:** `DATASET=a,b,c` truncates each dataset to the shortest one and interleaves them
(`a0 b0 c0 a1 …`) into `ckpt/_mix/`, with the eval metric `eval/mix`. GRPO normalizes the advantage
**within** a group of `NS` repeats of one prompt, so the families' reward scales never mix. `mixed` and
`libs` are prebuilt mixes (`build_mixed_datasets.py --name <n> --families …`) with per-family eval
files `<n>_<family>_eval.jsonl`.

## 2. Train on the default runtime (no image)

```bash
bash 1_prepare_data.sh          # container: prepare_data.sh (all datasets, or DATASET=<name>) + verify_data.py
bash 5_run.sh                   # container: 10 train steps + 1 eval on the Code Interpreter; logs/<tag>.{log,csv,eval.csv,png}
DATASET=gsm8k bash 5_run.sh     # easiest; good first run
DATASET=libs  bash 5_run.sh     # pandas / scipy / sklearn tasks, still no image
DATASET=gsm8k,gretelsql,hotpotqa bash 5_run.sh   # ad-hoc mix, eval/mix
```

`2_build_image.sh`, `3_push_ecr.sh`, `4_deploy_runtime.sh` and `6_cleanup.sh` print
`default Code Interpreter: no image/Runtime needed` and exit 0 when `SANDBOX=agentcore`.

**How it works:**

- Each episode opens its own Code Interpreter session (`StartCodeInterpreterSession`), runs its
  write/exec/read ops (`InvokeCodeInterpreter`), and stops the session.
- `AgentCoreSandbox` in `agentcore_slime_demo/sandbox.py` works in the session's native writable CWD.
  `writeFiles` accepts only **relative** paths.
- Sessions are isolated microVMs. No ARN and no IAM execution role are involved.

**Quotas** (us-west-2 defaults, from `aws service-quotas list-service-quotas --service-code bedrock-agentcore`):

| Quota | Code | Default | This demo |
|---|---|---:|---|
| Concurrent active Code Interpreter sessions | L-CAF6F552 | 1000 | `S = CONC × engines` = 4 × 2 = **8** episodes in flight |
| StartCodeInterpreterSession rate | L-6B09F374 | 30/s | about one per episode |
| InvokeCodeInterpreter rate (every write_file/exec/read_file) | L-04A719E9 | 30/s | about 6–10 per episode, spread over turns |

The default `CONC=4` sits far below all three quotas. The invoke rate is the binding one: at about
8 ops/episode and 20 s/episode, 30/s allows roughly 75 episodes in flight. Raise the quota before
running with `CONC ≳ 32`.

**Opt-in: your own Runtime** (`SANDBOX=agentcore-runtime`). Every step then does real work:

```bash
SANDBOX=agentcore-runtime bash 2_build_image.sh     # arm64 image from deploy/Dockerfile.sandbox.cpu (stdlib only)
SANDBOX=agentcore-runtime bash 3_push_ecr.sh        # -> ECR acs-tasks-cpu
SANDBOX=agentcore-runtime bash 4_deploy_runtime.sh  # Runtime acs_tasks_cpu -> deploy/runtime.env (ARNS)
SANDBOX=agentcore-runtime bash 5_run.sh             # without step 4: the baked shared K=2 acs_sandbox_cpu ARNs
SANDBOX=agentcore-runtime bash 6_cleanup.sh
```

- `Dockerfile.sandbox.cpu` is `python:3.12-slim` plus the sandbox server. It has **no pandas, scipy or
  sklearn**, so `DATASET=libs` and the datasci share of `mixed` score 0 on it unless you add those
  libraries to the image.
- The vendored lib still routes `task.image` to `AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE` when you give
  it a per-image ARN map.
- `AGENTCORE_SANDBOX_RUNTIME_ARNS` (comma list) fans episodes out round-robin for the Runtime-count
  ablation (`run_runtime_ablation.sh`).

**Same episode, same reward, different ops: `compare_backends.py`**

```bash
python3 compare_backends.py --backends local                                  # offline: solved=True reward=1.0
AGENTCORE_SANDBOX_RUNTIME_ARN=<cpu-arn> python3 compare_backends.py --backends local agentcore agentcore-runtime
```

The script asserts that every backend gives the same reward, then prints who builds the image, which
deps you get, and the deploy cost. Old logs are in `logs/customruntime/`.

## `HEARTBEAT=1`: keep idle Runtime sessions warm across the weight-sync barrier (was `demo_cpu_heartbeat`)

While slime syncs weights, generation stops and sandbox sessions sit idle. On **your own Runtime**
(`SANDBOX=agentcore-runtime`), an idle session is reclaimed after its idle timeout.

- **What it does.** `HEARTBEAT=1` (default 0, which leaves the run unchanged) wraps every episode's
  sandbox in `KeptSandbox`. A process-wide `SessionKeeper`
  (`session_keeper.py`) then pings any session that has been idle for more than
  `HEARTBEAT_IDLE × HEARTBEAT_PING_FRACTION` s (600 × 0.5) with `exec("true")`. Each ping logs
  `[demo] heartbeat_session=…`.
- **Code Interpreter: no effect.** On the default Code Interpreter a session ends at the
  `session_timeout_seconds` set when it starts, however much traffic it gets. Pings do not extend it
  (verified live). `run.sh` therefore warns and turns `HEARTBEAT` off for `SANDBOX=agentcore`, and the
  lib skips it with one `[demo] heartbeat_skipped=code_interpreter` line.
- **The Code Interpreter lever.** Set `AGENTCORE_CI_SESSION_TIMEOUT` (default **3600 s** per episode,
  max 8 h). Every session is stopped when its episode ends, so a generous
  timeout costs nothing.

```bash
python3 heartbeat_demo.py                      # offline ~30 s: naive loss -> heartbeat survives -> recover
HEARTBEAT=1 SANDBOX=agentcore-runtime DATASET=deepscaler bash 5_run.sh
```

## Knobs (env vars, defaults from `run.sh`)

| Knob | Default | Meaning |
|---|---|---|
| `DATASET` (alias `FAMILY`) | `mbpp` | dataset, comma list to mix, `mixed` or `libs` |
| `SANDBOX` | `agentcore` | managed Code Interpreter; `agentcore-runtime` = your own Runtime (needs ARNs). `local` is rejected |
| `MODE` | `B` | fully-async (staleness grows with `CONC`); `A` = 1-step-async |
| `RB` / `NS` / `NR` / `TEMP` | 4 / 8 / 11 / 1.0 | prompts/step, GRPO group, 10 train + 1 eval, temperature |
| `CONC` | `4` | concurrent episodes per engine; `S = CONC × engines` |
| `INFER_CONC` | `0` (off) | caps decode concurrency while sweeping `CONC` |
| `EVAL_INTERVAL` / `EVAL_NS` / `SAVE_INTERVAL` | 10 / 2 / 1000000 | eval cadence / samples / save only at the end |
| `AGENTCORE_SANDBOX_RUNTIME_ARNS` | baked K=2 | only for `SANDBOX=agentcore-runtime` |
| `HEARTBEAT` / `HEARTBEAT_IDLE` / `HEARTBEAT_PING_FRACTION` | 0 / 600 / 0.5 | session keeper (own Runtime only; no-op on the Code Interpreter) |
| `AGENTCORE_CI_SESSION_TIMEOUT` | 3600 | Code Interpreter session cap in seconds, set at session start (max 28800) |
| `ACTOR_GPUS` / `ROLLOUT_GPUS` | 4 / 4 | GPU split (TP2 actor, so `ACTOR_GPUS` is even) |
| `SLIME_CUDA_VISIBLE` / `SLIME_NUM_GPUS` | unset / 8 | pin ray to a GPU subset on a shared box (e.g. `2,3,4,5` / `4`) |

Only knobs that match `PASS_RE` in `deploy/steps_common.sh` are forwarded by `docker exec`.

To train to convergence, run inside the container:

```bash
DATASET=gsm8k STEPS=50 EVAL_EVERY=10 SANDBOX_CONC=4 ./train.sh
```

`STEPS` maps to `NR=STEPS+1`, `EVAL_EVERY` to `EVAL_INTERVAL` and `SANDBOX_CONC` to `CONC`.
`SANDBOX_RUNTIMES` caps K and applies only with `SANDBOX=agentcore-runtime`. Keep the train set at
`>= NR×RB` prompts so the run stays in one epoch.

## Metrics

`5_run.sh` runs `metrics.py` on its log and writes `logs/<tag>.{csv,eval.csv,png}`, plus the raw
per-op files `<tag>.sandbox_ops.csv` and `<tag>.sglang_gen.csv`. Watch these:

- `rollout/raw_reward`, the learning signal.
- `eval/<dataset>`, the held-out solve-rate.
- `perf/sandbox_ms_{mean,p50,p95}` and `perf/sandbox_count`, the sandbox round-trips.
- `perf/sglang_gen_ms_*`, pure decode time.
- The open/warm split in the `[demo]` lines.

The harness is in-process, so both latency splits are fully present in the log.

## Folder layout (self-contained)

- `agentcore_slime_demo/`: the vendored shared lib. `run.sh` puts `$DEMO_DIR` on `PYTHONPATH`. It adds:
  - `build_real_datasets.py` and `real_datasets.py` (HF loaders).
- `1_…6_*.sh`: the numbered steps. Settings are in `deploy/demo.env` (the `SANDBOX` switch lives there)
  and shared logic in `deploy/steps_common.sh`.
- `deploy/`: `Dockerfile.sandbox.cpu` (opt-in own Runtime) and `agentcore_deploy.py`.
- `prepare_data.sh`: download and prepare the data (step 1).
- `verify_data.py`: an offline checker proof (the reference PASSES, a wrong answer FAILS).
- `build_dataset.py` (MBPP), `build_family_datasets.py` (zoo), `build_mixed_datasets.py` (`mixed`, `libs`).
- `data/`: `*_train.jsonl` / `*_eval.jsonl` (incl. `libs_*`, `mixed_*`).
- `run.sh`: the RL run. `train.sh`: the convergence wrapper. `metrics.py`: parsing, CSVs, PNG, stickiness.
- Probes: `compare_backends.py`, `heartbeat_demo.py` (+ `agentcore_idle_sim.py`),
  `smoke_remote_families.py`.
- `logs/{imagepool,heartbeat,customruntime}/`: logs from the merged folders.
  `logs/bigimage/` is moving to `demo_cpu_instance_pool`.

Prereqs: the `agentcore_p4` container (repo at `/work/code`), `AWS_REGION=us-west-2` and AWS creds,
the base checkpoint `code/p4_slime_train/ckpt/Qwen3-4B_torch_dist` (`CKPT_BASE`), and the Qwen3-4B HF
snapshot in the container cache.
