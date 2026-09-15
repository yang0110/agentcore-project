# demo_cpu_2runtime — MBPP coding task, TWO AgentCore Runtimes (harness Runtime + CPU sandbox Runtime)

This is the **target decoupled layout** ("Demo A3"): the mini Claude-Code-like harness gets its **own
AgentCore Runtime #1**, and the CPU code-execution sandbox is a **separate AgentCore Runtime #2**.
The RL trainer (slime, on this box) never runs the agent loop or the sandbox itself — it only serves
the policy's completions and collects the trajectory.

## What this demo shows

- **Task:** MBPP Python code-repair. Each episode gives the policy a buggy/blank task + hidden
  asserts; the agent must `write_file` → `bash` (run the tests) → `read_file` → repeat until the
  asserts pass. Reward = tests pass.
- **The two-Runtime split:** the agent loop is `agentcore_slime_demo.mini_agent.MiniClaudeHarness`,
  wired in via `--custom-generate-function-path agentcore_slime_demo.agentcore_runtime_generate.generate`.
  That `generate` invokes **Runtime #1** (the harness, `harness_kind=mini`), whose three tools
  (`write_file` / `bash` / `read_file`) invoke **Runtime #2** (the CPU sandbox). The trainer process
  runs neither.
- **Why the harness needs VPC but the sandbox is public:** Runtime #1 must **dial back** to slime's
  `AnthropicAdapter` proxy on this box (a private IP, `ADAPTER_PUBLIC_HOST:ADAPTER_PORT`) to serve
  every model completion from the policy under training — so it runs `networkMode=VPC` and needs
  private VPC endpoints + one narrow security-group ingress rule (no public port). Runtime #2 is
  `networkMode=PUBLIC`: it's only ever invoked over the AWS control plane and never dials back, so it
  needs no VPC wiring. `deploy_two_runtime.py` sets all of this up automatically.

## AgentCore basics: launch / terminate / latency

- **Launch:** one command from `code/deploy/` — `python3 deploy_two_runtime.py all` — builds+pushes
  both arm64 images to ECR, creates the exec role, opens the pinned adapter port from the runtime SG,
  creates the private VPC endpoints (ECR/Logs/STS/bedrock-agentcore) the VPC-mode harness needs, then
  creates Runtime #1 (`acs_harness`, VPC) and Runtime #2 (`acs_sandbox_cpu`, PUBLIC). You must
  `export ADAPTER_PUBLIC_HOST=<this box PRIVATE IP>` first — that is the address Runtime #1 dials back
  to. It prints the export block with `AGENTCORE_HARNESS_RUNTIME_ARN` and
  `AGENTCORE_SANDBOX_CPU_RUNTIME_ARN`. Iterate on just the harness image with
  `python3 redeploy_harness.py` (reuses the same role / endpoints / ingress).
- **Terminate:** from `code/deploy/`, `python3 deploy_two_runtime.py cleanup` — deletes `acs_harness`
  and `acs_sandbox_cpu` (and the GPU capacity provider if one exists). `run.sh` does **no** Runtime
  teardown; it only `pkill`s local sglang and stops ray.
- **Latency (two-Runtime caveat):** per-op sandbox timings (`[sbx-op] op=… ms=…`, emitted by
  `agentcore_runtime_sandbox._timed_invoke`) land in **Runtime #1's CloudWatch log**, NOT in this
  trainer's `train.log`, because the harness runs on Runtime #1. To get per-op latency, download the
  harness Runtime's CloudWatch log and run `metrics.py` on it. The `perf/sglang_gen_ms_*` decode-time
  split is **not** produced here — completions are served by slime's in-tree `AnthropicAdapter`, not
  by our `generate`.

## Folder layout (self-contained)

- `agentcore_slime_demo/` — vendored copy of the shared library (this demo imports only this; it does
  not put `/work/code` on `PYTHONPATH`).
- `deploy/` — the two Dockerfiles: `Dockerfile.harness` (Runtime #1) and `Dockerfile.sandbox.cpu`
  (Runtime #2).
- `build_dataset.py`, `data/` — MBPP dataset builder + the built `mbpp_train.jsonl` / `mbpp_eval.jsonl`.
- `run.sh` — the slime async/decoupled training launcher.
- `metrics.py` — parses `train.log` (or a downloaded Runtime #1 log) into CSVs + a plot.

## Step by step

**0. Prereqs.** Run inside the persistent container `agentcore_p4` (repo mounted at `/work/code`;
in-container path `/work/code/demos/demo_cpu_2runtime`). Region is `us-west-2`. Have this box's
**private IP** (e.g. `PRIVATE-IP-f99864bc`) and the pinned adapter port `18021`. No GPU/torch prepare and no
smoke step are needed for the CPU sandbox.

**1. Build the dataset** (writes into this demo's own `data/`):

```bash
python3 build_dataset.py --dataset mbpp --split "train+test" --limit 500 --out data/mbpp_train.jsonl
python3 build_dataset.py --dataset mbpp --split validation   --limit 8   --out data/mbpp_eval.jsonl
```

500 train prompts keeps the whole run inside one epoch (`dataset_size ≥ NR × RB`), so the only
checkpoint saved is the final one.

**2. Build the two images** (self-contained: build with **this demo dir as the context** so
`COPY agentcore_slime_demo …` picks up the vendored copy). Both are `linux/arm64` (AgentCore's managed
fleet is ARM):

```bash
docker buildx build --platform linux/arm64 -f deploy/Dockerfile.harness \
  -t <ecr>/acs-harness:latest --push .
docker buildx build --platform linux/arm64 -f deploy/Dockerfile.sandbox.cpu \
  -t <ecr>/acs-sandbox-cpu:latest --push .
```

The harness image is a slim Python container (aiohttp + boto3 + bedrock-agentcore, no slime; runs
`python -m agentcore_slime_demo.harness_runtime_agent` after blanking `__init__.py`); the CPU sandbox
image is stdlib Python + bedrock-agentcore (runs `python -m sandbox_server_agent`).

**3. Launch both Runtimes.** From `code/deploy/` (shared infra — IAM/VPC/ECR helpers live here, not in
the demo). `deploy_two_runtime.py all` also builds+pushes both images, so step 2 is optional if you
just want the one-shot path:

```bash
cd /work/code/deploy
export AWS_REGION=us-west-2
export ADAPTER_PUBLIC_HOST=PRIVATE-IP-f99864bc        # <-- this box's PRIVATE IP
SKIP_PREPARE=1 python3 deploy_two_runtime.py all
```

`SKIP_PREPARE=1` skips the GPU-volume cold-launch (A3 uses only the CPU sandbox). Iterate on just the
harness image afterward with `ADAPTER_PUBLIC_HOST=PRIVATE-IP-f99864bc python3 redeploy_harness.py`.

**4. Export the ARNs + adapter vars.** Copy them from the printed export block (map the printed
`AGENTCORE_SANDBOX_CPU_RUNTIME_ARN` onto the sandbox var `run.sh` reads):

```bash
export SANDBOX=agentcore-runtime AWS_REGION=us-west-2
export AGENTCORE_HARNESS_RUNTIME_ARN=arn:aws:bedrock-agentcore:us-west-2:...:runtime/acs_harness-...
export AGENTCORE_SANDBOX_RUNTIME_ARNS=arn:aws:bedrock-agentcore:us-west-2:...:runtime/acs_sandbox_cpu-...
export ADAPTER_PUBLIC_HOST=PRIVATE-IP-f99864bc ADAPTER_PORT=18021
export AWS_BEARER_TOKEN_BEDROCK=<token>
```

`run.sh` requires `AGENTCORE_HARNESS_RUNTIME_ARN`, `ADAPTER_PUBLIC_HOST`, and a sandbox ARN. Note it
ships a pinned **K=2** default for `AGENTCORE_SANDBOX_RUNTIME_ARNS` (round-robin across two CPU sandbox
Runtimes); that plural var **takes precedence** over the singular `AGENTCORE_SANDBOX_RUNTIME_ARN`, so
set `AGENTCORE_SANDBOX_RUNTIME_ARNS` to the ARN(s) you actually deployed (one is fine).
`ADAPTER_PORT` defaults to `18021`.

**5. Run RL training:**

```bash
bash /work/code/demos/demo_cpu_2runtime/run.sh
```

Key knobs (env-var overrides): `RB` (prompts/step, default 4), `NS` (GRPO group size, 8),
`NR` (num-rollout; real steps = NR−1, default 11 → 10 train + 1 eval), `TEMP` (1.0),
`CONC` (sglang concurrency, 4), `EVAL_INTERVAL` (10; `0` disables), `EVAL_NS` (2),
`EPISODE_MAX_TURNS` / `MINI_AGENT_MAX_TURNS` (max model↔tool turns per episode, 3). GPU split on this
box is 4 train + 4 infer + 0 env (the sandbox is a separate CPU Runtime). `--sglang-tool-call-parser
qwen25` is required (without it Qwen3's `<tool_call>` blocks never parse, the tool loop dies after one
turn, and reward is all-zeros). `--async-save` + the `megatron_init.py` persistent-async-worker init
avoid the end-of-training checkpoint-save deadlock; both are baked into `run.sh`.

**6. Read metrics / latency:**

```bash
python3 metrics.py train.log --csv train.csv --eval-csv train.eval.csv --plot train.png
```

For sandbox per-op latency, download **Runtime #1's** CloudWatch log (the `[sbx-op]` records are there,
not in `train.log`) and run `metrics.py` on that file. `perf/sglang_gen_ms_*` is not emitted for this
demo.

**7. Terminate:**

```bash
cd /work/code/deploy && python3 deploy_two_runtime.py cleanup
```

Deletes `acs_harness` and `acs_sandbox_cpu`.
