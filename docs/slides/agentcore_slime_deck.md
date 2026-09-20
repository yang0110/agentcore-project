
**Kaige Yang** — Senior Applied Scientist, 亚马逊云科技 (AWS)

September 10, 2026

---


## Purpose — a practitioner's guide to AgentCore as an RL sandbox

This deck is a **hands-on guide for AWS customers doing agentic RL**. It is grounded in a working slime + AgentCore project and in the **official AgentCore docs** [D1].

**After reading it, you will know:**

- **What AgentCore can and cannot do** as an RL sandbox (and, optionally, harness).
- **How to run a real task on it** — with datasets, Dockerfiles, and deploy commands.
- **How to build and customize** your own Runtime — CPU (Code Interpreter) and GPU (Instances).

**Scope.** AgentCore = the execution **sandbox**; **slime** = the RL trainer; the **harness stays in-process**. This deck focuses on the sandbox seam: build image → deploy Runtime → seed a task → invoke → read the reward.

## The RL loop in one slide — the two seams AgentCore fills

In agentic RL the policy acts over many turns against an executable environment; a **programmatic checker** turns the result into a scalar reward — no LLM judge [1, 2].

*Loop:* **Trainer** (slime/Megatron, GPU) ↔ **Inference** (SGLang, GPU) via weight sync → **Proxy** serves each turn → **Harness** (agent loop, CPU) drives turns → **Sandbox** runs each tool call + the checker → reward.

**Two seams AgentCore fills:**

- **Sandbox** — the isolated place untrusted model code executes (write→run→read) and the reward is scored. *This is what we use AgentCore for.* A sandbox wants to be ad-hoc + isolated per episode — a perfect fit for AgentCore's per-session microVM.
- **Harness** — the agent loop itself. We keep it **in-process** (warm, next to the trainer); episodes are independent so it needs warmth, not isolation. AgentCore *can* host a harness, but that is not the recommended path for RL.

Only Trainer, Inference, and a GPU sandbox spend GPUs; harness and proxy run on cheap CPU.

## What tasks do frontier labs run in a sandbox?

Frontier models train with **RL against executable, verifiable environments** — the pattern this deck operationalizes. **Common denominator:** untrusted output runs in an isolated sandbox; a checker produces the reward — exactly the sandbox seam (MAI / Qwen3.5 publish no RL-sandbox details — omitted).

| Model | Sandbox / verifier | Tasks | Reward |
|---|---|---|---|
| DeepSeek-R1 [2] | code compiler + rule verifier | math, competitive code | tests pass / answer match |
| Qwen3 [3] | RLVR verifiers + live tool/agent env | math, code, tool-use, RAG | rule + model-judge (w/ ref) |
| Llama-Nemotron [4] | **LLM-judge, no code exec** | math, science, instruction | answer-match + format |
| Kimi K2 [5] | simulated tool-use envs + verifiers | agentic tool use, math, code | verifiable rewards |

---


## AgentCore basics — services, and can / cannot

Amazon Bedrock AgentCore [D1] runs and secures agents & tools — any framework, any model, serverless. Services this project uses:

- **Runtime** — serverless, **per-session microVM** (isolated CPU/mem/FS); invoke via `InvokeAgentRuntime` (SigV4) [D2, D8].
- **Code Interpreter** — isolated CPU sandbox; **Python / JS / TS** [D6].
- **Browser · Gateway (APIs→MCP) · Memory · Identity · Observability (OTEL)** [D13].

**Can:** strong per-session isolation for untrusted code; in-account (IAM/SigV4, VPC); CPU by default, **GPU on Instances**; long sessions; bring-your-own container.

**Cannot (per docs):** microVM has **no GPU** — GPU only on the **Instances** compute type [D5]; **no cross-session GPU time-slicing**; session caps (microVM 8 h) [D3]; VPC-mode needs endpoints wired up [D11].

## Lifecycle & latency — launch → invoke → terminate

- **Launch.** First `InvokeAgentRuntime` with a `runtimeSessionId` **cold-starts** a microVM; the same id re-routes to the warm VM [D2]. Session id length is **33–256** [D8] (we pad: `token_hex(16)`=32 + prefix ≥ 33).
- **Terminate.** Auto on **idle (15 min, resets on each invoke)**, **max lifetime** (microVM **8 h**; **Instances up to 14 d**), or unhealthy → microVM terminated + memory sanitized [D3, D4]. Explicit `StopRuntimeSession` [D12].
- **Latency.** Docs promise "fast cold starts" but publish **no number** [D2]; a GPU (Instances/g6) first invoke can cold-launch the EC2 in **minutes**. So **measure it**: our sandbox emits `[sbx-op]` per round-trip and `[sglang-gen]` per generation → `perf/sandbox_ms_{p50,p95}` & `perf/sglang_gen_ms_*` columns.

---


## The task zoo — 11 families, one harness

One **task-agnostic** harness; each family changes only **prompt + seed files + checker** (`task_families.py`). All **RLVR** — reward is binary **1.0/0.0** from a checker, no LLM judge. **87 seeds** across ten families + MBPP/KernelBench; select with `FAMILY=sql`, dispatch by `metadata.task.family`.

| Family | Seed files | Checker |
|---|---|---|
| coding (MBPP/KernelBench) | — | hidden asserts / allclose on cuda |
| swe | — | assert suite green |
| tool_use | `tools.py` | result==gold ∧ tools called |
| sql | → `query.sql` | result-set == gold (SQLite) |
| math · ctf | — | `|ans−gold|≤1e-6` · `check(flag)` |
| rag | `docs/*.txt` | answer == grounded fact |
| regex · planning · config · data | (varies) | POS/¬NEG · `verify()` · constraints · `output==gold` |

## The task→sandbox pattern — seed, run, check, reward

`run_episode` (`harness.py`) is family-agnostic — the same loop for all 11 families:

```python
for name, content in task.setup_files():
    await sandbox.write_file(name, content)
await sandbox.write_file("run_checks.py", task.runner_source())
for _ in range(task.max_turns):
    text = await policy.complete(messages)                 # model turn (via proxy)
    code = task.extract_submission(text)                   # fenced block
    await sandbox.write_file(task.artifact_filename, code) # solution.py / query.sql
    exit_code, out, err = await sandbox.exec("python3 run_checks.py", timeout=60)
    if exit_code == 0: solved = True; break
    messages.append(task.observation(exit_code, out, err)) # feed failure back
reward = task.reward(0 if solved else 1)                   # 1.0 / 0.0
```

Most families reuse `assert_runner` (a stdlib checker); only SQL uses a custom SQLite runner. **The checker runs *inside* the sandbox — the microVM is the reward source.**

## Datasets — build once, seed per episode

Builders (in `demos/demo_cpu_mbpp/`): `build_dataset.py` (MBPP/KernelBench) and `build_family_datasets.py` (ten families, `--eval-frac 0.25`). Uniform row schema:

```json
{"prompt": "…task text…", "label": "mbpp_601",
 "metadata": {"task": {"family": "sql", "task_id": "…",
                       "spec": {"schema": ["…"], "expected": [["…"]]}}}}
```

- **coding** rows carry `CodingTask` kwargs (no `family` key ⇒ defaults to `coding`); other families carry `family` + `spec`.
- **Real splits:** MBPP 500/8, KernelBench 100/20, families ≈ 4–12 train each (87 seeds).
- `generate.py` rebuilds the Task from metadata, seeds the workspace, runs the checker, emits a **loss-masked slime `Sample`** (mask 0 on prompt/observation/tool tokens, 1 on the policy's own tokens); reward attached from the episode.
- Reference answers (`mock_attempts`) are **dropped** from rows so they can't leak into prompts.

---


## CPU sandbox — Code Interpreter (the default)

Serverless CPU microVM; **Python/JS/TS**; session default **900 s**, max 8 h; 30-day data TTL [D6, D7]. Our `AgentCoreSandbox` wraps it (ops: `writeFiles` / `executeCommand` / `readFiles`).

**Use for:** MBPP/SWE (tests pass), Data/SQL (result-set), tool & shell tasks.

**Inside the microVM (sandbox only — the agent loop is elsewhere):** the workspace (`PROBLEM.md`, agent-written `solution.py`, `run_checks.py`), tool exec (`write_file`/`bash`/`read_file`), and the checker (the reward).

**BYO image — `deploy/Dockerfile.sandbox.cpu`:**

```dockerfile
FROM public.ecr.aws/docker/library/python:3.12-slim
RUN pip install --no-cache-dir bedrock-agentcore
COPY agentcore_slime_demo/sandbox_server_agent.py /app/
ENV SANDBOX_WORKDIR=/workspace
CMD ["python", "-m", "sandbox_server_agent"]
```

`SANDBOX=local` runs the *same* checker with no microVM — offline CI.

## GPU sandbox — Runtime on Instances (g6 / L4)

GPU is **not a microVM feature** — it lives on the **Instances** compute type (EC2 via a capacity provider); microVM GPU is "Not supported" [D5]. Supported families: `g4dn, g5, g6, g6e, g7e, …`. Sessions up to **14 days**.

**Use for:** KernelBench — write `ModelNew`, build+run on cuda, reward = allclose + speedup [7].

**`deploy/Dockerfile.sandbox.gpu` — two GPU-specific tweaks:**

```dockerfile
FROM public.ecr.aws/docker/library/python:3.12-slim
RUN pip install --no-cache-dir bedrock-agentcore
COPY agentcore_slime_demo/sandbox_server_agent.py /app/
ENV LD_LIBRARY_PATH=/usr/lib64          # AgentCore injects libcuda.so.1 here
ENV SANDBOX_WORKDIR=/workspace
CMD ["sh","-c","pip install torch==2.5.1 --index-url .../cu124 \
                && exec python -m sandbox_server_agent"]
```

Torch is installed **at `CMD`, not baked** — the CUDA wheel exceeds AgentCore's 2048 MB image cap. **Why remote GPU:** untrusted CUDA never shares VRAM/host with training state; the checker's g6 is disposable.

## Build & deploy a custom Runtime — the how-to

One driver: `deploy/deploy_two_runtime.py`. Managed runtimes must be **arm64**; the GPU capacity provider is **amd64**.

```bash
export AWS_REGION=us-west-2
export ADAPTER_PUBLIC_HOST=<this-box-private-ip>   # harness dials it back
python3 deploy_two_runtime.py all                  # ECR + buildx + create runtimes
```

**Steps:** (1) ECR create + `docker buildx --platform` cross-build + push; (2) `CreateAgentRuntime` with `containerConfiguration` [D9]; (3) for GPU, `CreateCapacityProvider` (`ec2Configuration`: g6.xlarge, VPC subnets/SGs, EBS, `idleInstanceTimeout=600`, `maxLifetime=86400`) then attach via `capacityProviderConfiguration` [D5, D9].

**IAM:** `acs-exec-role` (ECR pull, logs, `InvokeAgentRuntime`) + `acs-operator-role` (`BedrockAgentCoreRuntimeInstancesOperatorRolePolicy`).

**Customize** = edit `sandbox_server_agent.py` — a `BedrockAgentCoreApp` with `@app.entrypoint`. Redeploy one sandbox image with `redeploy_gpu_sandbox.py`.

## Can AgentCore host every frontier sandbox? — tick by tick

**Bottom line:** the only hard limit is GPU **sharing / fractioning** — not any missing capability. *Legend:* ✓ native · ◑ needs a BYO container / egress.

| Task / sandbox | ✓? | How — and the limit |
|---|---|---|
| Code exec + unit tests (DeepSeek, Qwen) [2,3] | ✓ | Code Interpreter, or your image on a Runtime |
| Math / logic verifier | ✓ | stdlib checker in the CPU microVM |
| Tool-use env (Kimi K2) [5] | ✓ | Python tools; MCP via Gateway |
| Per-repo Docker image (SWE) [6] | ✓ | Runtime runs your OCI image; isolated per session |
| Browser / computer use | ✓ | managed Browser tool |
| Networked RAG / web read | ◑ | egress — public, or VPC + endpoints [D11] |
| GPU kernel / CUDA (KernelBench) [7] | ◑ | Instances only; 1 session = 1 GPU; **no MIG on L4** |

---


## slime integration — one hook, no fork

slime (THUDM) [9] = Megatron trainer + SGLang inference, run **async/decoupled**. AgentCore drops into two of slime's seams via **hooks — no fork**:

- `--custom-generate-function-path` → `generate.generate` (drives the harness, records turns).
- `--custom-rm-path` → the reward.

Each tool call is `invoke_agent_runtime(ARN, session_id)` (SigV4); boto3 sync is offloaded via `asyncio.to_thread`; **one session = one episode**. A proxy (`AnthropicAdapter`) fronts SGLang so every model call is the policy under training. Bound episode cost with `EPISODE_MAX_TURNS` (default 3).

**Caveat.** The two-Runtime sandbox client (`AgentCoreRuntimeSandbox`) is labeled scaffolding in-repo — validate before production.

## slime ↔ AgentCore Runtime — the call path

slime calls the `generate` hook once per prompt; the hook opens **one AgentCore session per episode** and each tool call becomes one SigV4 `InvokeAgentRuntime` [D8]:

```python
async def generate(args, sample, sampling_params):        # slime hook
    task = task_from_dict(sample.metadata["task"])
    async with make_sandbox("agentcore-runtime") as sb:   # -> AgentCoreRuntimeSandbox
        traj = await run_episode(task, policy, sb)        # seed/turn/check loop
    return build_token_stream(traj, sample)               # loss-masked Sample

def _invoke(self, op, **kw):                               # sandbox client: 1 op = 1 invoke
    body = json.dumps({"op": op, "session_id": self.session_id, **kw})
    r = self._rt.invoke_agent_runtime(                    # boto3 bedrock-agentcore
        agentRuntimeArn=self.arn, runtimeSessionId=self.session_id, payload=body)
    return json.loads(r["response"].read())               # {"exit_code","stdout",...}
```

`session_id` (≥33 chars) pins every op of an episode to **one warm microVM** [D2]; boto3's sync call is offloaded with `asyncio.to_thread`, so N episodes run concurrently. The `agentcore_slime_demo.sandbox_server_agent` `@app.entrypoint` on the Runtime dispatches `op` → `ping/write_file/exec/read_file`.

## Step-by-step — one MBPP episode end-to-end

1. `build_dataset.py --dataset mbpp` → `mbpp_train.jsonl` (rows carry `metadata.task`).
2. `run.sh` sets `SANDBOX=agentcore`, `FAMILY=coding`, `CONC`, `EPISODE_MAX_TURNS`; then `ray job submit … train_async.py --custom-generate-function-path …generate.generate`.
3. Per sample: `generate()` rebuilds the Task, `make_sandbox("agentcore")`, `async with sandbox:` → **cold-start** microVM.
4. **Seed:** write `PROBLEM.md` + `run_checks.py`.
5. **Turn loop:** `policy.complete` (via proxy — `[sglang-gen]`) → extract `solution.py` → `sandbox.exec("python3 run_checks.py")` (`[sbx-op]`) → observation.
6. `exit_code == 0` → solved; **reward = 1.0 / 0.0**.
7. `build_token_stream` → loss-masked `Sample` → buffer → Megatron step → weight sync.

**Swap to GPU:** `SANDBOX=agentcore-runtime` + a GPU ARN — **zero loop change** (KernelBench, allclose on cuda).

## Demos — one in-process harness, the sandbox is what varies

Policy is always **Qwen3-4B**, trained by slime. The harness is **in-process** — a warm agent loop next to the trainer (a harness wants warmth, not a cold session per episode). **AgentCore is the sandbox**, and that is the only thing that varies across demos: the sandbox substrate (CPU microVM vs GPU Runtime instance) and the task. **Identical code CPU vs GPU — only the sandbox ARN differs.**

| | CPU sandbox — MBPP | GPU sandbox — KernelBench |
|---|---|---|
| in-process harness | `demo_cpu_mbpp` | `demo_gpu_single` |

Same substrates, more tasks/patterns: `demo_cpu_math`, `demo_cpu_heartbeat` (heartbeat keepalive), `demo_gpu_shared` (Ray pack-many farm), `demo_gpu_snapshot` (app-level snapshot/resume).

## The GPU limit — and how you'd share one L4

AgentCore hands **one whole GPU per session** (1 `runtimeSessionId` = 1 EC2 instance). No documented cross-session time-slicing [D5].

- **Isolated scaling** = more instances (1 GPU each) — clean, N× cost.
- **Free concurrency** = many agents in one session (share the instance GPU; "must be mutually trusted" — no isolation).

Our B4 farm: a `RayGpuFarm` of fractional actors (`num_gpus=0.5`) in one session, with a per-process memory cap + an nvcc `flock` (a non-MIG L4 wedges under compile storms). **Takeaway:** isolated scaling → more instances; fractioning → many-agents-per-session or your own Ray/EKS.

| Share one physical GPU by | Isolation | AgentCore |
|---|---|---|
| Agents / session · MPS · time-slice · Ray `num_gpus<1` | none | native / self-managed |
| NVIDIA MIG | hard | **no on L4** |

## Networking & isolation — private by default

The loop runs untrusted model code, so the network is the primary control. Principle: **pull, don't expose.**

- Every sandbox/harness is **reached** by a SigV4 `invoke_agent_runtime` call — **no inbound port** on any Runtime.
- **One inbound rule ever:** tcp/`ADAPTER_PORT` on the trainer box, source = the *runtime SG* (never a CIDR).
- Harness dials back over `networkMode=VPC` [D10]; VPC-mode has no internet route, so it needs **interface endpoints**: `ecr.api`, `ecr.dkr`, `s3` (gateway), `logs`, `sts`, `bedrock-agentcore` [D11]. Miss one — or land the ENI in an AZ that doesn't serve `bedrock-agentcore` — and create fails with a **null reason**. Pin subnets to served AZs.
- **IAM/SigV4 is the auth**; the `session_id` bearer is *routing*, not auth — so the adapter must stay private.
- Everything keyed by `session_id`: one warm microVM per episode; distinct sid ⇒ distinct sandbox + trajectory bucket.

## Summary — can, cannot, how to run

- **Can:** isolated per-session sandbox for untrusted code (CPU default), in-account (IAM/VPC), long sessions, BYO container, **GPU on Instances**, and the harness on its own Runtime.
- **Cannot:** microVM GPU; cross-session GPU fractioning (no MIG on L4); exceed the microVM 8 h cap without Instances.
- **How to run:** build a sandbox image → deploy a Runtime / GPU capacity provider → seed a task → `invoke_agent_runtime` per tool call → the checker gives the reward → slime trains via one generate hook.
- **Repo:** `agentcore_slime_demo/` (harness, sandbox, tasks), `deploy/` (Dockerfiles + `deploy_two_runtime.py`), `demos/` (the 2×2). All grounded in the official AgentCore docs.

---

## References — papers & models

```{=latex}
\footnotesize
```

**Agentic RL & models.** [1] Agent Lightning — Harnessed Agentic RL. <https://arxiv.org/abs/2508.03680> · [2] DeepSeek-R1. <https://arxiv.org/abs/2501.12948> · [3] Qwen3 Technical Report. <https://arxiv.org/abs/2505.09388> · [4] Llama-Nemotron. <https://arxiv.org/abs/2505.00949> · [5] Kimi K2. <https://arxiv.org/abs/2507.20534> · [6] SWE-bench. <https://arxiv.org/abs/2310.06770> · [7] KernelBench. <https://huggingface.co/datasets/ScalingIntelligence/KernelBench> · [8] MBPP. <https://arxiv.org/abs/2108.07732> · [9] slime. <https://github.com/THUDM/slime>

## References — AgentCore docs (1/2)

```{=latex}
\scriptsize
```

**Official AgentCore docs.** [D1] What is AgentCore. <https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/what-is-bedrock-agentcore.html> · [D2] Host with Runtime (isolation, compute types). <https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agents-tools-runtime.html> · [D3] Runtime how it works (idle 15 m / max 8 h). <https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-how-it-works.html> · [D4] Lifecycle settings. <https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-lifecycle-settings.html> · [D5] Instances (GPU). <https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-instances-how-it-works.html> · [D6] Code Interpreter. <https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/code-interpreter-tool.html> · [D7] Code Interpreter session. <https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/code-interpreter-session-characteristics.html>

## References — AgentCore docs (2/2)

```{=latex}
\scriptsize
```

[D8] InvokeAgentRuntime. <https://docs.aws.amazon.com/bedrock-agentcore/latest/APIReference/API_InvokeAgentRuntime.html> · [D9] CreateAgentRuntime. <https://docs.aws.amazon.com/bedrock-agentcore-control/latest/APIReference/API_CreateAgentRuntime.html> · [D10] NetworkConfiguration. <https://docs.aws.amazon.com/bedrock-agentcore-control/latest/APIReference/API_NetworkConfiguration.html> · [D11] VPC interface endpoints. <https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/vpc-interface-endpoints.html> · [D12] StopRuntimeSession. <https://docs.aws.amazon.com/bedrock-agentcore/latest/APIReference/API_StopRuntimeSession.html> · [D13] Observability. <https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/observability.html>

---

*Thank you — Questions?*
