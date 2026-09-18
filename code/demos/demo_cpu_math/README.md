# demo_cpu_math — math word problems, one CPU AgentCore sandbox (in-process harness)

An agentic-RL loop that RL-trains Qwen3-4B to solve **math word problems** by **writing
`solution.py` → running the hidden checker → reading the failure → fixing → repeat**. The AgentCore
Runtime is the **RL sandbox**: each turn's `write_file` / `exec` / `read_file` runs inside a managed
serverless (CPU) Code-Interpreter microVM, reached over `invoke_agent_runtime`, while the multi-turn
`run_episode` harness and the slime trainer stay in-process on the training box.

This is a **Tier-1, per-task CPU-microVM demo** (see [`docs/demos-program.md`](../../../docs/demos-program.md)).
It is the same loop as every CPU demo — only the **task** (and its dataset + checker) differs.

## What this demo shows
- **Task:** multi-turn math problem solving with a hidden numeric/assert checker (`MathTask` in
  `agentcore_slime_demo/task_families.py`). Trains on **real** competition/grade-school math
  datasets pulled from HuggingFace (all RLVR — exact numeric reward, no LLM judge):

  | `FAMILY`     | dataset (HuggingFace)                    | train/eval | reward |
  |--------------|------------------------------------------|-----------:|--------|
  | `gsm8k`      | `openai/gsm8k` (main)                    |    200/40  | exact integer match |
  | `math500`    | `HuggingFaceH4/MATH-500`                 |    160/40  | numeric match, tol 1e-4 |
  | `deepscaler` | `agentica-org/DeepScaleR-Preview-Dataset`|    400/40  | numeric match, tol 1e-4 |

  (The old hand-authored `data/math_{train,eval}.jsonl` seeds — 9/3 rows — remain for offline
  smoke tests but are too small for a real run.)
- **Reward:** the candidate's `solution.py` is run against the task's hidden checks; the learning
  signal is `rollout/raw_reward`, and held-out **solve-rate** (`eval/math`) is the key eval metric.
- **Sandbox arrangement:** ONE CPU-backed AgentCore Code-Interpreter sandbox. The checker is plain
  Python, so the sandbox is CPU-only and all 8 L40S GPUs serve the policy (**4 train + 4 inference +
  0 env**). Harness runs **in-process** (`--custom-generate-function-path
  agentcore_slime_demo.generate.generate`); there is NO separate harness Runtime.
- **Expected bottleneck:** **generation-bound** — the checker is cheap, so per-episode time is
  dominated by LLM decode. Tier-3 confirms this from `perf/sglang_gen_ms_*` ≫ `perf/sandbox_ms_*`
  and scales the **inference engine** accordingly. (Measure; don't assume.)

## Reproduce it (run these one by one)
Everything runs INSIDE the persistent container `agentcore_p4` (slimerl/slime); the repo is at
`/work/code`. `data/` is already populated — step 1 only re-pulls it from scratch.

```bash
# 1. Download + prepare ALL real datasets into data/ (one-time network to pull the HF rows).
#    Builds gsm8k (200/40), math500 (160/40), deepscaler (400/40); idempotent.
docker exec agentcore_p4 bash /work/code/demos/demo_cpu_math/prepare_data.sh

# 2. Train + eval on one task. FAMILY == the data-file prefix, so no DATA/EVAL_DATA overrides:
docker exec -e FAMILY=gsm8k      agentcore_p4 bash -lc 'cd /work && bash code/demos/demo_cpu_math/run.sh'
docker exec -e FAMILY=math500    agentcore_p4 bash -lc 'cd /work && bash code/demos/demo_cpu_math/run.sh'
docker exec -e FAMILY=deepscaler agentcore_p4 bash -lc 'cd /work && bash code/demos/demo_cpu_math/run.sh'
```
`run.sh` maps `FAMILY=<x>` → `data/<x>_train.jsonl` + `data/<x>_eval.jsonl` and eval label `<x>`.
Knobs (env vars): `NR` num-rollout ⇒ real steps `NR-1` (smoke uses `NR=3`, converge with a larger
budget), `EVAL_INTERVAL`, `RB`/`NS` (GRPO batch/group), `TEMP`, `SANDBOX` (`agentcore-runtime`
= real remote CPU sandbox, default; `local` = offline in-container; `agentcore` = Code-Interpreter
microVM). Real remote runs need `AGENTCORE_SANDBOX_RUNTIME_ARN(S)` + AWS creds in the container.

Offline dataset check (reference solution PASSES the hidden checker, a wrong one FAILS — proves the
reward is real before any GPU run):
```bash
docker exec agentcore_p4 python3 /work/code/verify_real_datasets.py
```

## Train to convergence
This demo's bar (per `docs/demos-program.md`) is **convergence**, not a smoke run: use a step budget
large enough that the held-out `eval/<task>` solve-rate **rises then plateaus**, with `train/grad_norm`
and `train/kl_loss` staying sane. Plot with `metrics.py` (below); `train.png` should show reward and
`eval/<task>` trending up then flat.

**`train.sh`** wraps `run.sh` with the top-level knobs — training **steps**, eval **cadence**, and
the two **sandbox** counts — passing `FAMILY` (and everything else) straight through:
```bash
docker exec -e FAMILY=gsm8k -e STEPS=50 -e EVAL_EVERY=10 -e SANDBOX_CONC=4 -e SANDBOX_RUNTIMES=2 \
    agentcore_p4 bash -lc 'cd /work && bash code/demos/demo_cpu_math/train.sh'
```

| Knob | Default | Maps to | Meaning |
|---|---|---|---|
| `STEPS` | 50 | `NR = STEPS+1` | real training steps; `STEPS=2` for a smoke |
| `EVAL_EVERY` | 10 | `EVAL_INTERVAL` | held-out eval cadence in steps (0 = off) |
| `SANDBOX_CONC` | 4 | `CONC` | concurrent episodes (`S = CONC × num_engines`) |
| `SANDBOX_RUNTIMES` | K=2 | truncates the ARN list | # distinct sandbox Runtimes to spread across |
| `FAMILY` | `math` | passthrough | which task: `gsm8k` / `math500` / `deepscaler` |

## Metrics → CSV + PNG
```bash
python3 code/demos/demo_cpu_math/metrics.py <train.log> \
    --csv train.csv --eval-csv train.eval.csv --plot train.png
```
Both latency splits (`perf/sandbox_ms_*` and `perf/sglang_gen_ms_*`) are fully present because the
harness is in-process — that ratio is the Tier-3 bottleneck signal.

## Folder layout (self-contained)
- `agentcore_slime_demo/` — vendored shared library (imports only this copy), incl.
  `build_real_datasets.py` + `real_datasets.py` (the HuggingFace loaders → task rows)
- `deploy/Dockerfile.sandbox.cpu` — the Dockerfile the CPU sandbox image is built from
- `data/` — this demo's datasets (`gsm8k_*`, `math500_*`, `deepscaler_*`; plus tiny legacy `math_*`)
- `prepare_data.sh` — **download + prepare all real datasets** (step 1 above)
- `build_family_datasets.py` — regenerates the tiny hand-authored `math_*` seeds (legacy/offline only)
- `run.sh` — the RL training run (async/decoupled slime); select the task with `FAMILY=<name>`
- `metrics.py` — reward + latency parsing (CSVs + PNG)

References (not copied): the base Megatron checkpoint, the slime install, and the shared deploy
orchestrator `code/deploy/deploy_two_runtime.py`.
