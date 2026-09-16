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

There is a **third** tier hiding behind those two: the **harness** that drives the episodes. Rollout
throughput is `min(inference, sandbox, harness)`, so even after inference + sandbox are scaled, a
single-process harness can cap everything. [`demo_cpu_harness_scale`](../code/demos/demo_cpu_harness_scale/)
isolates that tier — one task, a pool of **W warm mini-Claude harness workers** — and shows the knee
(throughput rises ~linearly with `W`, then plateaus at `min(inference, sandbox)`) plus how the three
tiers route/load-balance differently (per-request router vs per-episode sticky sandbox vs per-episode
pull queue). Full write-up: [harness-scaling.md](harness-scaling.md).

This is why disaggregation matters: each component is sized to its own bottleneck, per task.

### Advanced: keeping a paused sandbox alive across the weight-sync barrier — two demos

During slime's trainer→inference **weight sync**, generation stalls, so open sandbox sessions sit
idle. Surviving that pause is **two patterns on two substrates** — one demo each:

| Demo | Substrate | Lever | Survives |
|---|---|---|---|
| [`demo_cpu_quiesce`](../code/demos/demo_cpu_quiesce/) | serverless CPU **Code-Interpreter microVM** | **heartbeat** (+ quiesce gate) | the *idle timeout* — keep the warm session from being reclaimed during the pause |
| [`demo_gpu_snapshot`](../code/demos/demo_gpu_snapshot/) | **GPU Runtime instance** | **application-level snapshot & resume** | a genuine *recycle* — restore workspace state onto a fresh session (there is no VM snapshot; [gpu-snapshot.md](gpu-snapshot.md)) |

**Heartbeat keeps a *live* session warm; snapshot restores a *lost* one.** Both are standalone,
offline (no AWS/GPU), runnable in seconds. Reusable mechanisms:
`agentcore_slime_demo/session_keeper.py` (`SessionKeeper` + `KeptSandbox`) and
`agentcore_slime_demo/snapshot_store.py` (`snapshot_workspace`/`restore_workspace`).

#### Demo 1 — microVM heartbeat (`demo_cpu_quiesce`)

Chosen semantics: **lightweight quiesce (a gate)** — hold new sandbox invokes during sync, keep
sessions warm via heartbeat, resume after. The heartbeat lives **outside the harness** by design:
`SessionKeeper` is a process-wide registry + one background heartbeat loop driven by the trainer's
`sync_begin()`/`sync_end()` barrier, and `KeptSandbox` composes it with any `Sandbox` — so no harness
grows its own keepalive.

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
   episode. The *restore-with-progress* form (application-level snapshot to S3, since there is no VM
   snapshot — [gpu-snapshot.md](gpu-snapshot.md)) is built + demonstrated in
   [`code/demos/demo_gpu_snapshot/`](../code/demos/demo_gpu_snapshot/) (`snapshot_store.py`).

Demo arc: naive "stop invoking" (anti-pattern, silently loses the GPU instance) → heartbeat quiesce
that survives the idle timeout → resume → recover-if-reclaimed. This deliberately re-couples the
sandbox to the trainer's sync barrier — a **cost** optimization, not a correctness fix (async RL
episodes are otherwise designed to span weight versions).

#### Demo 2 — runtime-instance application-level snapshot & resume (`demo_gpu_snapshot`)

For the **GPU Runtime instance**, the failure mode is not idle reclamation but a **recycle** that
comes back blank — and AgentCore offers **no VM/EC2 snapshot** ([gpu-snapshot.md](gpu-snapshot.md)).
So the only feasible snapshot is **application-level**: serialize the sandbox workspace to a store
you own (**S3** in production — no mount, AZ-agnostic; FSx/EFS need a mount AgentCore won't give you)
and **restore** it onto a fresh session after the recycle. `snapshot_workspace` / `restore_workspace`
(`agentcore_slime_demo/snapshot_store.py`) do this backend-agnostically.

Demo arc: RUN A (no snapshot — recycle loses progress, redo from scratch) vs RUN B (snapshot before
recycle → restore on fresh session → resume), printing the rework saved. Use this to survive a real
recycle; use the heartbeat (Demo 1) for the seconds-long weight-sync pause where no state is lost.

---

## The convergence bar (Tier 1 & 2)

A per-task demo is "done" when it shows **learning to convergence**, not a smoke run:

- Enough training steps that the **held-out eval solve rate rises and plateaus** (not the default
  ~10-step run). Eval every `EVAL_INTERVAL` steps.
- The `train.png` for the folder shows `rollout/raw_reward` and `eval/<task>` trending up then flat.
- `train/grad_norm` and `train/kl_loss` stay sane (policy moving, not diverging).

Each Tier-1/2 README states the step budget and the observed convergence point for that task.

---

## Shared vs copied (layout B)

Each demo folder is self-contained: its own `data/`, `run.sh`, `README.md`, a vendored copy of the
`agentcore_slime_demo/` library, and its `deploy/` Dockerfile(s). It only *references* (does not
copy) the shared infra that would drift if duplicated: the base Megatron checkpoint, the slime
install, and the deploy orchestrator `code/deploy/deploy_two_runtime.py`. Trade-off accepted: a
reader gets one folder end-to-end, at the cost of the library being vendored per task — keep library
edits in `code/agentcore_slime_demo/` and re-vendor.
