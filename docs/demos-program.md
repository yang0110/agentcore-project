# Demos program — the demo taxonomy

This is the anchor for the demo set. It fixes **what demos exist, how they're organized, and the
bar each must clear.** Every demo folder and its README conforms to this doc.

## Premise (from the architecture decision)

**AgentCore is the sandbox.** The harness stays **in-process** (a warm agent loop next to the
trainer); episodes are independent, so the harness needs *warmth*, not per-episode *isolation* —
whereas the sandbox wants exactly the ad-hoc, isolated, per-session microVM AgentCore provides. See
[harness-sandbox-decoupling.md](harness-sandbox-decoupling.md). The demos are therefore organized
around **the sandbox and the task**, not around where the harness runs.

The policy is always **Qwen3-4B**, trained by **slime** (async/decoupled GRPO). The reward is a
**programmatic checker** run in the sandbox — no LLM judge.

---

## The tiers

### Tier 0 — Foundations: build the whole workflow

Show the end-to-end pipeline once, with no AWS and no GPU, so the loop and the reward are legible
before any cost is incurred:

`offline multi-turn loop (LocalSandbox + MockPolicy)` → `deploy a sandbox Runtime` →
`seed a task (PROBLEM.md + run_checks.py)` → `invoke` → `read the reward` → `the slime train hook`.

Covered today by `code/scripts/smoke.py` and `code/scripts/e2e_rl.py`. Tier 0 presents these *as the
first demo*, framed as "how the workflow is built," not as loose scripts.

### Tier 1 — Per-task, CPU serverless microVM (Code Interpreter)

**One self-contained demo folder per task** (layout B). Each folder is a complete, readable recipe:
task → dataset → Dockerfile → launch → **train to convergence** → terminate. The sandbox is an
AgentCore **Code Interpreter** microVM (CPU); the harness is in-process.

| Demo folder | Task | Dataset | Likely bottleneck* |
|---|---|---|---|
| `demo_cpu_coding` | MBPP (real) | `mbpp` | generation (cheap checker) |
| `demo_cpu_swe`     | synthetic SWE unit-tests | `swe` | balanced |
| `demo_cpu_tool_use`| tool-call sequences | `tool_use` | generation |
| `demo_cpu_sql`     | SQL over sqlite | `sql` | generation (fast exec) |
| `demo_cpu_math`    | math word problems | `math` | **generation** (long reasoning, trivial check) |
| `demo_cpu_ctf`     | capture-the-flag | `ctf` | mixed |
| `demo_cpu_rag`     | retrieval QA | `rag` | generation |
| `demo_cpu_regex`   | regex synthesis | `regex` | generation |
| `demo_cpu_planning`| plan verification | `planning` | generation |
| `demo_cpu_config`  | config edits | `config` | generation |
| `demo_cpu_data`    | data transforms | `data` | generation |

\* *Hypothesis only — Tier 3 replaces it with a measurement.* Note most cheap-checker CPU tasks are
expected to be **generation-bound**; the clear **sandbox-bound** case is CUDA (Tier 2). That
asymmetry is the point of measuring rather than assuming.

### Tier 2 — GPU instance (capacity-provider EC2 GPU)

For tasks whose *checker itself needs CUDA*. The sandbox is a real AgentCore **GPU Runtime
instance** (a `g6`, x86_64/amd64); the trainer box keeps all its GPUs for policy + inference.

| Demo folder | Task | Notes |
|---|---|---|
| `demo_gpu_kernelbench` | KernelBench (level_1 CUDA) | checker = allclose vs torch ref, on `cuda` |
| `demo_gpu_shared` | KernelBench + Ray pack-many farm | many envs share one GPU (bursty, memory-bound) |

### Tier 3 — Advanced: scale the *measured* bottleneck

**For each task, measure first, then scale.** Per-episode wall-clock splits into **LLM generation**
(SGLang) and **sandbox exec** (checker). The repo already logs both per step:

- **`perf/sglang_gen_ms_*`** — time spent generating turns, and
- **`perf/sandbox_ms_*`** — time spent in the sandbox (tool exec + checker).

Decision procedure:

1. Run the task's Tier-1/2 demo, read `perf/sglang_gen_ms_*` vs `perf/sandbox_ms_*` from the CSV.
2. **`sglang_gen_ms ≫ sandbox_ms`** → **generation-bound** → scale the **inference engine** (more
   SGLang GPUs / replicas). Typical for cheap-checker CPU tasks (math, sql, rag, …).
3. **`sandbox_ms ≫ sglang_gen_ms`** → **sandbox-bound** → scale the **sandbox** (more microVM
   sessions; on GPU, the Ray pack-many farm / more capacity). Typical for KernelBench.
4. Show the throughput / step-time improvement after scaling the correct side — and that scaling the
   *other* side would not have helped.

This is why disaggregation matters: each component is sized to its own bottleneck, per task.

### Advanced: quiesce / resume the sandbox across the weight-sync barrier

> Built: [`code/demos/demo_cpu_quiesce/`](../code/demos/demo_cpu_quiesce/) — a standalone,
> offline (no AWS/GPU) runnable of the full arc below, in seconds. The reusable mechanism lives in
> `agentcore_slime_demo/session_keeper.py` (`SessionKeeper` + `KeptSandbox`).

During slime's trainer→inference **weight sync**, generation stalls, so open sandbox sessions sit
idle. This demo **quiesces** the sandbox for that window and **resumes** after. Chosen semantics:
**lightweight quiesce (a gate)** — hold new sandbox invokes during sync, keep sessions warm, resume
after. Targets the **GPU-instance** tier (where idle time is real cost), with a **CPU-microVM
quiesce variant** shown purely as the coordination pattern (no cost payoff on serverless).

The heartbeat lives **outside the harness** by design: `SessionKeeper` is a process-wide registry +
one background heartbeat loop driven by the trainer's `sync_begin()`/`sync_end()` barrier, and
`KeptSandbox` composes it with any `Sandbox` — so no harness grows its own keepalive.

**The trap this demo must handle: idle-timeout termination.** Nothing about weight sync terminates
the sandbox — but a sandbox is auto-reclaimed by *absence of invokes*:

| Clock | CPU (Code Interpreter) | GPU (capacity-provider instance) |
|---|---|---|
| **Idle timeout** (no invoke) | ~15 min (`recovery.py:18`) | `idleRuntimeSessionTimeout`/`idleInstanceTimeout` = 600 s (`deploy_two_runtime.py:402,430`) |
| **Max lifetime** | 8 h | `maxLifetime` = 24 h (up to 14 d) |
| **Recycle** | container crash | instance recycle — **GPU state NOT reattached** (`Dockerfile.sandbox.gpu:10`) |

So a quiesce that goes silent **longer than the idle timeout** loses the session (and on GPU, a
recycle loses the paused state). Sync is usually seconds, so a naive quiesce *usually* survives —
which is exactly why the demo shows the failure first. The robust pattern:

1. **Heartbeat keepalive** — during quiesce, send a cheap no-op invoke (`exec: "true"` / status
   ping) at an interval well under the idle timeout, to reset the idle clock. Keeps the pause
   arbitrarily long. Implemented as `SessionKeeper` + `KeptSandbox`
   (`agentcore_slime_demo/session_keeper.py`). *(Alternative: raise the idle timeout so the sync
   window fits — simpler, worse under other stalls. A microVM/instance **snapshot** is NOT an
   option — see [gpu-snapshot.md](gpu-snapshot.md).)*
2. **Recovery fallback** — if the session dies anyway (idle trip / instance recycle), resume via
   `recovery.py` self-heal: restart the session and re-seed/restore the workspace, don't crash the
   episode.

Demo arc: naive "stop invoking" (anti-pattern, silently loses the GPU instance) → heartbeat quiesce
that survives the idle timeout → resume → recover-if-reclaimed. This deliberately re-couples the
sandbox to the trainer's sync barrier — a **cost** optimization, not a correctness fix (async RL
episodes are otherwise designed to span weight versions).

---

## The convergence bar (Tier 1 & 2)

A per-task demo is "done" when it shows **learning to convergence**, not a smoke run:

- Enough training steps that the **held-out eval solve rate rises and plateaus** (not the default
  ~10-step run). Eval every `EVAL_INTERVAL` steps.
- The `train.png` for the folder shows `rollout/raw_reward` and `eval/<task>` trending up then flat.
- `train/grad_norm` and `train/kl_loss` stay sane (policy moving, not diverging).

Each Tier-1/2 README states the step budget and the observed convergence point for that task.

---

## Appendix — variant (not recommended): harness on its own AgentCore Runtime

`demo_cpu_2runtime` / `demo_gpu_2runtime` also host the *harness* on an AgentCore Runtime (Runtime #1
driving sandbox Runtime #2). Kept only as a reference that AgentCore *can* host a harness. Not part
of the recommended program: the current hook mints a fresh harness session per episode (a cold
start), when the harness wants a warm reused-session pool. See
[harness-sandbox-decoupling.md](harness-sandbox-decoupling.md) §3.

---

## Shared vs copied (layout B)

Each demo folder is self-contained: its own `data/`, `run.sh`, `README.md`, a vendored copy of the
`agentcore_slime_demo/` library, and its `deploy/` Dockerfile(s). It only *references* (does not
copy) the shared infra that would drift if duplicated: the base Megatron checkpoint, the slime
install, and the deploy orchestrator `code/deploy/deploy_two_runtime.py`. Trade-off accepted: a
reader gets one folder end-to-end, at the cost of the library being vendored per task — keep library
edits in `code/agentcore_slime_demo/` and re-vendor.
