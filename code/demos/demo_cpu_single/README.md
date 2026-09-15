# demo_cpu_single — MBPP coding task, one CPU AgentCore sandbox (in-process harness)

An agentic-RL loop that RL-trains Qwen3-4B to solve Python problems by **writing `solution.py` →
running the hidden unit tests → reading the failure → fixing → repeat**. The AgentCore Runtime is
the **RL environment**: each turn's `write_file` / `exec` / `read_file` is executed inside a managed
serverless sandbox Runtime, reached over `invoke_agent_runtime`, while the multi-turn `run_episode`
harness and the slime trainer stay on the training box.

## What this demo shows
- **Task:** MBPP (`agentcore_slime_demo/mbpp.py`) — multi-turn Python code repair. 500-prompt train
  split + 8-prompt held-out eval split.
- **Reward:** the candidate's `solution.py` is run against the task's hidden MBPP asserts; the
  learning signal is `rollout/raw_reward`, and held-out **solve-rate** is the key eval metric. GRPO
  advantage is 0 when every sample in a prompt's group gets the same reward, so reward variance
  (higher `TEMP` / harder tasks) is what produces gradient.
- **Sandbox arrangement:** ONE CPU-backed AgentCore Runtime sandbox (`sandbox_server_agent.py`).
  The MBPP checker is plain Python/IO, so the sandbox is CPU-only and all 8 L40S GPUs serve the
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
  `run.sh` puts `$DEMO_DIR` on `PYTHONPATH`, not `/work/code`)
- `deploy/` — `Dockerfile.sandbox.cpu`, the Dockerfile the sandbox image is built from
- `build_dataset.py`, `data/` — the MBPP dataset (`mbpp_train.jsonl`, `mbpp_eval.jsonl`)
- `run.sh` — the RL training run (async/decoupled slime)
- `metrics.py` — reward + latency parsing (CSVs + PNG)

## Step by step

### 0. Prereqs
- Run everything INSIDE the persistent container `agentcore_p4` (slimerl/slime); the repo is at
  `/work/code`, this demo at `/work/code/demos/demo_cpu_single`.
- Region `us-west-2` (`AWS_REGION`), plus AWS creds (`AWS_BEARER_TOKEN_BEDROCK`) for the
  `invoke_agent_runtime` calls.
- Shared, NOT vendored (reference by path): the deploy orchestrator under `code/deploy/`, and the
  base Megatron checkpoint at `code/p4_slime_train/ckpt/Qwen3-4B_torch_dist` (override with
  `CKPT_BASE`). The Qwen3-4B HF snapshot must be in the container HF cache (baked into `run.sh` as
  `$HF`).

### 1. Build the dataset
`data/` is already populated; rebuild only to change size/splits:
```bash
python build_dataset.py --dataset mbpp --split "train+test" --limit 500 \
    --out data/mbpp_train.jsonl
python build_dataset.py --dataset mbpp --split validation --limit 8 \
    --out data/mbpp_eval.jsonl
```
Keep the train set `>= NR*RB` prompts so one epoch spans the whole run (500 ≥ 11×4) — that keeps
save-only-at-end holding.

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
```bash
SANDBOX=agentcore-runtime ./run.sh 2>&1 | tee train.log
```
Key knobs (env vars, defaults from `run.sh`):
- `SANDBOX` (`agentcore-runtime`) — remote Runtime; `agentcore` = Code Interpreter microVM;
  `local` = in-container offline sandbox (CI/unit tests).
- `MODE` (`B`) — fully-async (deep buffer, staleness grows with `CONC`); `A` = 1-step-async.
- `RB` (`4`) prompts/step, `NS` (`8`) GRPO group size (`GBS=RB×NS`), `NR` (`11`) num-rollout ⇒
  real steps = `NR-1` = 10 train + 1 eval (smoke).
- `TEMP` (`1.0`) rollout temperature.
- `CONC` (`4`) sglang-server-concurrency ⇒ concurrent episodes `S = CONC×2`; in Mode B it also sets
  buffer depth (the sandbox-concurrency ablation knob). `INFER_CONC` (`0`=off) caps decode
  concurrency to hold inference load constant while sweeping `CONC`.
- `EVAL_INTERVAL` (`10`), `EVAL_NS` (`2`), `SAVE_INTERVAL` (`1000000`, save-only-at-end).
- `FAMILY` (`coding`=MBPP) — the in-process harness is task-agnostic and can also train
  swe/tool_use/sql/math/ctf/rag families (see `build_family_datasets.py`).

Runtime-count ablation (sweeps `RUNTIMES={1 2 4}`): `./run_runtime_ablation.sh`.

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
