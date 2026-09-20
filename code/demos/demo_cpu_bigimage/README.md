# demo_cpu_bigimage — a LARGE sandbox image (past the 2 GB microVM cap) on a CPU Instance

The AgentCore **Code Interpreter microVM** has an image-size cap (~2 GB), so you can't bake a heavy
checker environment (torch, a full scientific stack) into it. This demo shows the escape hatch: build
a **large custom Runtime image** (`deploy/Dockerfile.sandbox.bigcpu`, ~4 GB) and run it on an
AgentCore **capacity-provider Instance** (a CPU EC2 family you pick — `m5.xlarge` by default) instead
of a microVM. It then trains Qwen3-4B against that sandbox exactly like the other CPU demos.

The teaching point: **escaping the image cap is purely a deployment change.** The in-process
`run_episode` harness, the flattened-trajectory `Sample` contract, the reward path, and every slime
arg are byte-for-byte identical to `demo_cpu_customruntime` — only *where* the sandbox runs (a
capacity-provider Instance, not a microVM) differs. It is the CPU analogue of the GPU custom-Runtime
path in `demo_gpu_customruntime`.

| | Code Interpreter microVM | **This demo — Runtime on a CPU Instance** |
|---|---|---|
| Image cap | ~2 GB | instance root disk (`ROOT_FREE_GIB`, default 40 GB) |
| Heavy deps (torch, scipy, sklearn) | ✗ won't fit | ✓ baked in |
| Compute | AWS-managed microVM | **your EC2 family** (`CPU_INSTANCE`, default `m5.xlarge`) |
| Session length | ~8 h | 14 days |
| Reached via | `invoke_agent_runtime` (IAM pull) | `invoke_agent_runtime` (IAM pull) |

This demo is **REMOTE-ONLY** (`SANDBOX=local` is rejected in `run.sh`) and **fully self-contained**
(owns its `data/`, `run.sh`, vendored `agentcore_slime_demo/`, `deploy/`, and offline probes).

---

## Offline probes (no AWS, no heavy deps — run these first)

```bash
# 1. Prove the image exceeds the microVM cap AND the harness contract is unchanged (LocalSandbox):
python3 probe_bigimage.py
#    -> "~3.88 GB EXCEEDS the microVM cap -> deploy on a capacity-provider INSTANCE" + a solved local episode
#    (if you've built the image: python3 probe_bigimage.py --built-tag acs-sandbox-bigcpu for the REAL size)

# 2. Validate the (optional) safe-networking config BEFORE deploying — the checklist as code:
python3 netcheck.py --write-samples          # writes netcheck_sample_{good,bad}.json
python3 netcheck.py                            # checks the GOOD sample -> exits 0
python3 netcheck.py --config netcheck_sample_bad.json   # shows every unsafe pattern -> exits 1
python3 netcheck.py --config my_deploy.json    # validate YOUR config
```

## Networking (safe by default)

Default is the **pull** shape: `networkMode=PUBLIC`, reached only by IAM `invoke_agent_runtime`
(SigV4/TLS), **no inbound port and no security-group ingress rule opened**. That is all this demo
needs — the harness is in-process and the sandbox never dials back. Only if you add an in-sandbox
agent that must reach a service you host would you switch to VPC + an SG→SG rule; validate that with
`netcheck.py`. Full guidance: [`docs/agentcore-networking.md`](../../../docs/agentcore-networking.md).

## Reproduce (inside the `agentcore_p4` container)

```bash
# 0. Deploy the big-image Runtime on a CPU capacity-provider Instance (build -> ECR -> CP -> Runtime).
#    Prints `export AGENTCORE_SANDBOX_RUNTIME_ARN=...`. Optional: CPU_INSTANCE=m5.2xlarge / ROOT_FREE_GIB=60.
python3 deploy/deploy_bigimage.py
export AGENTCORE_SANDBOX_RUNTIME_ARN='arn:aws:bedrock-agentcore:us-west-2:<acct>:runtime/<bigcpu-rt>'

# 1. Build the dataset (offline; agentica-org/DeepScaleR numeric subset, RLVR numeric match).
docker exec agentcore_p4 bash /work/code/demos/demo_cpu_bigimage/prepare_data.sh
#    -> data/deepscaler_{train,eval}.jsonl (400/40)

# 2. REAL RL train + eval on the big-image Instance Runtime: 10 train steps + 1 eval, single final ckpt.
docker exec -e NR=11 -e EVAL_INTERVAL=10 -e EVAL_NS=2 \
    -e AGENTCORE_SANDBOX_RUNTIME_ARN="$AGENTCORE_SANDBOX_RUNTIME_ARN" agentcore_p4 \
    bash -lc 'cd /work && bash code/demos/demo_cpu_bigimage/run.sh 2>&1 | tee code/demos/demo_cpu_bigimage/smoke_run.log'

# 3. Parse the log into CSVs + a PNG of the curves.
docker exec agentcore_p4 python3 /work/code/demos/demo_cpu_bigimage/metrics.py \
    /work/code/demos/demo_cpu_bigimage/smoke_run.log \
    --csv train.csv --eval-csv train.eval.csv --plot train.png
```

## Knobs (env vars; defaults in `run.sh` / `deploy_bigimage.py`)

| Knob | Default | Meaning |
|---|---|---|
| `SANDBOX` | `agentcore-runtime` | the big-image Instance Runtime. **`local` is rejected.** |
| `AGENTCORE_SANDBOX_RUNTIME_ARN` | — (**required**) | the ARN from `deploy_bigimage.py` (no baked default — big image is not a shared microVM) |
| `CPU_INSTANCE` | `m5.xlarge` | EC2 family for the capacity provider (deploy) |
| `ROOT_FREE_GIB` | `40` | root-volume free space (must exceed the unpacked image) (deploy) |
| `FAMILY` | `deepscaler` | task family (any in `agentcore_slime_demo/task_families.py`) |
| `RB` / `NS` | 4 / 8 | GRPO prompts-per-step / group size |
| `NR` | 11 | num-rollout ⇒ real steps `NR-1` (10 train + 1 eval) |
| `MODE` | `B` | `B` fully-async \| `A` 1-step-async |
| `EVAL_INTERVAL` / `EVAL_NS` | 10 / 2 | held-out eval cadence / samples |

## Folder layout (self-contained)

- `run.sh` — the REAL RL trainer (big-image Instance Runtime) · `prepare_data.sh` — build data ·
  `metrics.py` — parse the log to CSV/PNG.
- `probe_bigimage.py` — offline: image-size-vs-cap verdict + LocalSandbox contract parity.
- `netcheck.py` (+ `netcheck_sample_{good,bad}.json`) — the safe-networking checklist as executable checks.
- `agentcore_slime_demo/` — vendored harness/sandbox/task lib (`sandbox.py::make_sandbox` is the backend factory).
- `deploy/Dockerfile.sandbox.bigcpu` — the ~4 GB image · `deploy/deploy_bigimage.py` — CP + Runtime deploy.
- `data/` — `deepscaler_{train,eval}.jsonl`.
