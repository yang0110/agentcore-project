# demo_cpu_customruntime — REAL RL training on YOUR custom AgentCore Runtime (CPU)

This demo trains Qwen3-4B with slime (async/decoupled GRPO) where **every rollout sandbox is your
own custom AgentCore Runtime** (`SANDBOX=agentcore-runtime`) — an OCI image you build and deploy
yourself (`sandbox_server_agent.py` inside a `BedrockAgentCoreApp`, answering `/invocations` +
`/ping`), not the managed Code Interpreter. The point: **a custom Runtime serves an RL training
substrate unchanged.** The in-process `run_episode` harness, the flattened-trajectory `Sample`
contract, the reward path and every slime arg are byte-for-byte identical to the other CPU demos;
the *only* thing that differs is which backend the sandbox factory opens.

The managed Code Interpreter (`SANDBOX=agentcore`) exposes the **identical** task-agnostic contract
(`write_file` / `exec` / `read_file`), so it is a drop-in alternative backend — the offline
`compare_backends.py` probe in this dir proves that parity: the same episode scores the same reward
on `local` / `agentcore` / `agentcore-runtime`; **only the operational trade-offs differ**.

| `SANDBOX=` value    | backend class             | what it is                                  | you build/deploy      | you control                                | GPU |
|---------------------|---------------------------|---------------------------------------------|-----------------------|--------------------------------------------|-----|
| `agentcore-runtime` | `AgentCoreRuntimeSandbox` | **your own custom Runtime** (your OCI image)| an image + a Runtime  | deps, tools, user, filesystem, VPC/IAM     | CPU here¹ |
| `agentcore`         | `AgentCoreSandbox`        | managed **Code Interpreter** (AWS microVM)  | nothing               | nothing — fixed stdlib-Python env, AWS CWD | no  |
| `local`             | `LocalSandbox`            | offline subprocess (dev/CI only)            | nothing               | the host env; **not** isolated             | n/a |

¹ This demo is CPU. The **exact same custom-Runtime path** puts a GPU under the sandbox on a
capacity-provider Instance — see the sibling **`demo_gpu_customruntime`**.

Why choose a custom Runtime as the training substrate: you own every dependency, tool, user,
filesystem layout, and the VPC/IAM posture (a hidden checker can need packages, a private endpoint,
a specific user) — at the cost of building an image, pushing it to ECR, and managing a Runtime. The
managed Code Interpreter is zero-ops but you take the stock environment AWS gives you (no custom
system packages, no GPU). Because the contract is task-agnostic you can move a whole RL workload
between them without touching training code.

This demo is **REMOTE-ONLY** (`SANDBOX=local` is rejected in `run.sh`): a CPU training run must never
silently fall back off the AgentCore substrate it is meant to exercise. It is **fully self-contained**
(owns its `data/`, `ckpt/`, `run.sh`, vendored `agentcore_slime_demo/`, `deploy/Dockerfile.sandbox.cpu`).

---

## Reproduce (inside the `agentcore_p4` container)

```bash
# 0. Deploy the custom CPU Runtime once (build -> ECR -> CreateAgentRuntime). This repo has tooling:
#    code/deploy/deploy_two_runtime.py builds+pushes acs-sandbox-cpu and creates acs_sandbox_cpu.
#    Export its ARN (or a comma-separated pool) — run.sh ships a K=2 baked default you can override:
export AGENTCORE_SANDBOX_RUNTIME_ARNS='arn:aws:bedrock-agentcore:us-west-2:<acct>:runtime/<cpu-rt>'

# 1. Build the dataset (offline; agentica-org/DeepScaleR numeric-answer subset, RLVR numeric match).
docker exec agentcore_p4 bash /work/code/demos/demo_cpu_customruntime/prepare_data.sh
#    -> data/deepscaler_{train,eval}.jsonl (400/40)

# 2. REAL RL train + eval on the custom Runtime: 10 train steps + 1 eval, single final ckpt.
docker exec -e NR=11 -e EVAL_INTERVAL=10 -e EVAL_NS=2 agentcore_p4 \
    bash -lc 'cd /work && bash code/demos/demo_cpu_customruntime/run.sh 2>&1 | tee code/demos/demo_cpu_customruntime/smoke_run.log'

# 3. Parse the log into CSVs + a PNG of the curves.
docker exec agentcore_p4 python3 /work/code/demos/demo_cpu_customruntime/metrics.py \
    /work/code/demos/demo_cpu_customruntime/smoke_run.log \
    --csv train.csv --eval-csv train.eval.csv --plot train.png
```

## Offline contract-parity probe (no training, no AWS needed for `local`)

`compare_backends.py` runs the **same trivial episode** against one or more backends and prints
reward + an operational-diff table — the proof that the custom Runtime and the Code Interpreter are
interchangeable substrates:

```bash
# offline (LocalSandbox, no creds) — same episode, proves the contract locally
python3 compare_backends.py --backends local --family ctf

# live parity: same episode on Code Interpreter AND your custom Runtime, asserts equal reward
AWS_REGION=us-west-2 AGENTCORE_SANDBOX_RUNTIME_ARN=<cpu-arn> \
    python3 compare_backends.py --backends agentcore agentcore-runtime local --family ctf
```

> Note: `run.sh` is now the **trainer** (it rejects `SANDBOX=local`). The backend-contrast driver is
> `compare_backends.py` — invoke it directly, as above; it stays in this dir as the offline parity check.

## Knobs (env vars; defaults in `run.sh`)

| Knob | Default | Meaning |
|---|---|---|
| `SANDBOX` | `agentcore-runtime` | sandbox backend: **your custom Runtime** \| `agentcore` (Code Interpreter). **`local` is rejected.** |
| `AGENTCORE_SANDBOX_RUNTIME_ARNS` | baked K=2 CPU pool | comma-separated custom-Runtime ARNs (round-robin); override with your own |
| `FAMILY` | `deepscaler` | task family (any in `agentcore_slime_demo/task_families.py`) |
| `RB` / `NS` | 4 / 8 | GRPO prompts-per-step / group size (`GBS = RB×NS`) |
| `NR` | 11 | num-rollout ⇒ real steps `NR-1` (10 train + 1 eval); keep the train set ≥ `NR×RB` |
| `CONC` | 4 | sandbox/episode concurrency; episodes in flight `S = CONC × num_engines` |
| `MODE` | `B` | `B` fully-async (deep buffer) \| `A` 1-step-async |
| `EVAL_INTERVAL` / `EVAL_NS` | 10 / 2 | held-out eval cadence / samples |

## What to watch

`rollout/raw_reward` (learning signal) and `eval/deepscaler` (held-out solve-rate) — these prove the
custom Runtime actually carries training. Also watch the `perf/sandbox_ms_*` vs `perf/sglang_gen_ms_*`
split (`metrics.py` writes per-op latency CSVs) — it confirms the remote custom-Runtime round-trips
and shows the custom Runtime is not the bottleneck.

## Deploying the custom Runtime

`deploy/Dockerfile.sandbox.cpu` is tiny — `python:3.12-slim` + `bedrock-agentcore` +
`sandbox_server_agent.py`, serving `/invocations` on 8080. Build (native amd64) with the demo dir as
context, push to ECR, `CreateAgentRuntime`. Tooling: `code/deploy/deploy_two_runtime.py` (base CPU
Runtime `acs_sandbox_cpu`), `code/deploy/deploy_cpu_sandbox_fleet.py` (`FLEET_N=4 …` mints identical
Runtimes and prints the `AGENTCORE_SANDBOX_RUNTIME_ARNS` list). The Code Interpreter backend
(`SANDBOX=agentcore`) needs none of this — just AWS credentials.

## Folder layout (self-contained)

- `run.sh` — the REAL RL trainer (custom Runtime substrate) · `prepare_data.sh` — build data ·
  `metrics.py` — parse the log to CSV/PNG.
- `compare_backends.py` — the offline contract-parity probe (same episode across backends).
- `agentcore_slime_demo/` — vendored harness/sandbox/task lib (`sandbox.py::make_sandbox` is the
  backend factory). Imported only from here.
- `deploy/Dockerfile.sandbox.cpu` — the custom-Runtime image.
- `data/` — `deepscaler_{train,eval}.jsonl`.
