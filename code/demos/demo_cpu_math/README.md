# demo_cpu_math — math word problems, one CPU AgentCore sandbox (in-process harness)

An agentic-RL loop that RL-trains Qwen3-4B to solve **math word problems** by **writing
`solution.py` → running the hidden checker → reading the failure → fixing → repeat**. The AgentCore
Runtime is the **RL sandbox**: each turn's `write_file` / `exec` / `read_file` runs inside a managed
serverless (CPU) Code-Interpreter microVM, reached over `invoke_agent_runtime`, while the multi-turn
`run_episode` harness and the slime trainer stay in-process on the training box.

This is a **Tier-1, per-task CPU-microVM demo** (see [`docs/demos-program.md`](../../../docs/demos-program.md)).
It is the same loop as every CPU demo — only the **task** (and its dataset + checker) differs.

## What this demo shows
- **Task:** the `math` family (`agentcore_slime_demo/task_families.py`, `MathTask`) — multi-turn
  math problem solving with a hidden numeric/assert checker. Seed set in `data/math_{train,eval}.jsonl`.
- **Reward:** the candidate's `solution.py` is run against the task's hidden checks; the learning
  signal is `rollout/raw_reward`, and held-out **solve-rate** (`eval/math`) is the key eval metric.
- **Sandbox arrangement:** ONE CPU-backed AgentCore Code-Interpreter sandbox. The checker is plain
  Python, so the sandbox is CPU-only and all 8 L40S GPUs serve the policy (**4 train + 4 inference +
  0 env**). Harness runs **in-process** (`--custom-generate-function-path
  agentcore_slime_demo.generate.generate`); there is NO separate harness Runtime.
- **Expected bottleneck:** **generation-bound** — the checker is cheap, so per-episode time is
  dominated by LLM decode. Tier-3 confirms this from `perf/sglang_gen_ms_*` ≫ `perf/sandbox_ms_*`
  and scales the **inference engine** accordingly. (Measure; don't assume.)

## Run it
```bash
# Build this task's dataset (writes math_train.jsonl + math_eval.jsonl into data/)
docker exec agentcore_p4 python /work/code/demos/demo_cpu_math/build_family_datasets.py \
    --out-dir /work/code/demos/demo_cpu_math/data

# Train (async/decoupled slime). FAMILY defaults to `math` in this folder's run.sh.
docker exec agentcore_p4 bash -lc \
    'cd /work && SANDBOX=local bash code/demos/demo_cpu_math/run.sh'          # offline sandbox
# SANDBOX=agentcore swaps in real AWS Code-Interpreter microVMs (needs AWS creds in the container).
```

## Train to convergence
This demo's bar (per `docs/demos-program.md`) is **convergence**, not a smoke run: use a step budget
large enough that the held-out `eval/math` solve-rate **rises then plateaus**, with `train/grad_norm`
and `train/kl_loss` staying sane. Plot with `metrics.py` (below); `train.png` should show reward and
`eval/math` trending up then flat.

## Metrics → CSV + PNG
```bash
python3 code/demos/demo_cpu_math/metrics.py <train.log> \
    --csv train.csv --eval-csv train.eval.csv --plot train.png
```
Both latency splits (`perf/sandbox_ms_*` and `perf/sglang_gen_ms_*`) are fully present because the
harness is in-process — that ratio is the Tier-3 bottleneck signal.

## Folder layout (self-contained)
- `agentcore_slime_demo/` — vendored shared library (imports only this copy)
- `deploy/Dockerfile.sandbox.cpu` — the Dockerfile the CPU sandbox image is built from
- `data/` — this task's dataset (`math_train.jsonl`, `math_eval.jsonl`)
- `build_family_datasets.py` — regenerates the family datasets (math is extracted from it)
- `run.sh` — the RL training run (async/decoupled slime); `FAMILY=math` by default
- `metrics.py` — reward + latency parsing (CSVs + PNG)

References (not copied): the base Megatron checkpoint, the slime install, and the shared deploy
orchestrator `code/deploy/deploy_two_runtime.py`.
