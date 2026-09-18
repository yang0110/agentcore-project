# demo_cpu_math500 — competition math, one CPU AgentCore sandbox (in-process harness)

Self-contained agentic-RL demo: RL-trains Qwen3-4B on **HuggingFaceH4/MATH-500** via the multi-turn
"write `solution.py` → run the hidden checker → read the failure → fix → repeat" loop. The AgentCore
Runtime is the **RL sandbox** — each turn's `write_file` / `exec` / `read_file` runs inside a managed
serverless CPU Code-Interpreter microVM (`invoke_agent_runtime`), while the `run_episode` harness and
the slime trainer stay **in-process**. Verifiable reward (RLVR): numeric match (tol 1e-4), numeric-answer subset — no LLM judge.

- **Task family:** `math`   ·   **Dataset:** `HuggingFaceH4/MATH-500`   ·   **Reward:** numeric match (tol 1e-4), numeric-answer subset
- **Data:** `data/math500_train.jsonl` (160 rows) + `data/math500_eval.jsonl` (40 rows)

## Reproduce (run one by one)
Everything runs INSIDE the persistent container `agentcore_p4` (slimerl/slime); the repo is at
`/work/code`. `data/` is already populated — step 1 only re-pulls it from scratch.
```bash
# 1. Download + prepare this task's dataset (one-time network to pull the HF rows)
docker exec agentcore_p4 bash /work/code/demos/demo_cpu_math500/prepare_data.sh

# 2. Prove the reward is real, offline (reference solution PASSES, a wrong one FAILS)
docker exec agentcore_p4 python3 /work/code/demos/demo_cpu_math500/verify_data.py

# 3. Train + eval (async/decoupled slime). Smoke uses NR=3; converge with a larger NR.
docker exec -e NR=3 -e EVAL_INTERVAL=2 agentcore_p4 bash -lc 'cd /work && bash code/demos/demo_cpu_math500/run.sh'
```
`run.sh` bakes in `FAMILY=math500` (⇒ `data/math500_{train,eval}.jsonl`, eval label `math500`).
Real remote runs need `AGENTCORE_SANDBOX_RUNTIME_ARN(S)` + AWS creds in the container (`run.sh` ships
a pinned CPU-sandbox ARNS default). `SANDBOX=local` runs fully offline (CI/unit tests).

## Train to convergence (one command)

For a real (non-smoke) run, **`train.sh`** wraps `run.sh` with the three top-level knobs a client
tunes — training **steps**, eval **cadence**, and the two **sandbox** counts — and passes everything
else through:

```bash
docker exec -e STEPS=50 -e EVAL_EVERY=10 -e SANDBOX_CONC=4 -e SANDBOX_RUNTIMES=2 agentcore_p4 \
    bash -lc 'cd /work && bash code/demos/demo_cpu_math500/train.sh'
```

| Knob | Default | Maps to | Meaning |
|---|---|---|---|
| `STEPS` | 50 | `NR = STEPS+1` | real training steps (GRPO updates); `STEPS=2` for a smoke |
| `EVAL_EVERY` | 10 | `EVAL_INTERVAL` | held-out eval cadence in steps (0 = off) |
| `SANDBOX_CONC` | 4 | `CONC` | concurrent episodes (`S = CONC × num_engines`) |
| `SANDBOX_RUNTIMES` | run.sh default (K=2) | truncates the ARN list | # **distinct** sandbox Runtimes to spread episodes across |

The two sandbox knobs are separate on purpose: `SANDBOX_CONC` is how many episodes run at once,
`SANDBOX_RUNTIMES` is across how many physical Runtimes they're spread. Convergence is
task-dependent — watch `eval/<task>` (see Metrics) and raise `STEPS`; staying at
`STEPS ≤ train_rows/RB − 1` keeps the run inside one epoch (single final checkpoint).

## Knobs (env vars; defaults in run.sh)
`NR` num-rollout ⇒ real steps `NR-1`; `EVAL_INTERVAL`; `RB`/`NS` GRPO batch/group (`GBS=RB×NS`);
`TEMP` rollout temperature; `CONC` episode concurrency (`S=CONC×2`); `SANDBOX`
(`agentcore-runtime` default | `agentcore` Code-Interpreter microVM | `local` offline).

## Metrics → CSV + PNG
```bash
docker exec agentcore_p4 python3 /work/code/demos/demo_cpu_math500/metrics.py <train.log> \
    --csv train.csv --eval-csv train.eval.csv --plot train.png
```
Watch `rollout/raw_reward` (learning signal), `eval/math500` (held-out solve-rate), and the
`perf/sandbox_ms_*` vs `perf/sglang_gen_ms_*` split (remote sandbox vs trainer/SGLang bottleneck).

## Folder layout (self-contained)
- `agentcore_slime_demo/` — vendored shared library (real_datasets.py loaders, in-process `generate`
  hook, the assert checker, mode_b_eval, megatron_init); this demo imports only this copy
- `deploy/Dockerfile.sandbox.cpu` — the Dockerfile the CPU sandbox image is built from
- `data/` — this task's `math500_train.jsonl` / `math500_eval.jsonl`
- `prepare_data.sh` — download + build the dataset (step 1)
- `verify_data.py` — offline checker proof (step 2)
- `run.sh` — the RL training run (async/decoupled slime; `FAMILY=math500` baked in)
- `metrics.py` — reward + latency parsing (CSVs + PNG)

Shared by reference (NOT vendored): the base Megatron checkpoint
(`code/p4_slime_train/ckpt/Qwen3-4B_torch_dist`, override `CKPT_BASE`), the slime install, and the
deploy orchestrator (`code/deploy/deploy_two_runtime.py`).
