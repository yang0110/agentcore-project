# Demos program — the demo taxonomy

> **Historical (2026-09-30).** This doc was written for demos and code dirs that were removed on 2026-09-30; the only live demo is [`code/demos/demo_cpu_terminalbench`](../code/demos/demo_cpu_terminalbench/README.md).

This is the anchor for the demo set. It fixes **what demos exist, how they're organized, and the
bar each must clear.** Every demo folder and its README conforms to this doc.

## Premise (from the architecture decision)

**AgentCore is the sandbox, and the harness runs in it.** Since 2026-09-28 every demo runs the
mini_claude agent loop **inside the episode's sandbox session** (`in_session_harness.py`), dialing the
trainer's per-engine `AnthropicAdapter` back; the trainer keeps only a warm dispatch tier
(`harness_pool.py`: `HARNESS_WORKERS` workers, least-loaded engine per episode) and the adapters. The
harness scales with the sandbox, not the trainer (DSec's "move the agent loop out of the trainer"),
and it must scale along with inference and the sandbox: `throughput = min(inference, sandbox, harness)`
([harness-scaling.md](harness-scaling.md)). The one exception is the experimental `demo_cpu_branching`,
which runs the real Claude Code CLI in its sandbox microVM. History of the earlier in-process design:
[harness-sandbox-decoupling.md](harness-sandbox-decoupling.md). The demos are organized around **the
sandbox and the task**.

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
AgentCore **Code Interpreter** microVM (CPU); the harness was in-process. *(Historical: since
2026-09-28 the CPU tasks run on our own Runtime with the loop in the session; see the current set below.)*

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

> **As built (2026-09-25):** the per-task folders above are NOT separate demos. They are all ONE
> demo, `code/demos/demo_cpu_tasks`, selected with `DATASET=<name>` (real HF `mbpp` `gsm8k`
> `math500` `deepscaler` `humaneval` `gretelsql` `hotpotqa` `xlam` `mcp`, plus the synthetic `math`
> `swe` `tool_use` `sql` `ctf` `rag` `regex` `planning` `config` `data`). It replaced the ten
> per-dataset folders (`demo_cpu_mbpp`, `demo_cpu_math`, …) that differed only in data. Mixed-task
> batches are the same demo and, since 2026-09-28, its DEFAULT (`mbpp,humaneval,gsm8k,math500,deepscaler,gretelsql,hotpotqa,xlam,mcp`: the 9 real HF literature datasets,
> one prompt per dataset per step, RB=9; Terminal-Bench was in this mix 2026-09-28→09-30 and moved back to its own demo): `DATASET=a,b,c` (task-interleaved) or `DATASET=mixed` (math+sql+ctf+datasci,
> formerly `demo_cpu_mixtask`). All of them run on the DEFAULT managed Code Interpreter
> (`SANDBOX=agentcore`): no image, no Runtime. The pandas/scipy/sklearn tasks (formerly `demo_cpu_imagepool`,
> 3 custom images) are `DATASET=libs` there, since the Code Interpreter already has those libraries. The
> heartbeat (formerly `demo_cpu_heartbeat`) is the `HEARTBEAT` knob (default on for remote own Runtimes; a no-op on the
> Code Interpreter), and the custom-vs-managed Runtime check (formerly `demo_cpu_customruntime`) is
> `demo_cpu_tasks/compare_backends.py`. The ~4 GB image on a CPU capacity-provider Instance (formerly
> `demo_cpu_bigimage`, then `demo_cpu_tasks VARIANT=bigimage`, then the side note `demo_cpu_instance_pool`) was
> **removed 2026-09-28** with its AWS resources, and came back 2026-09-30 as `demo_cpu_instance` (strictly one
> m5 per session, no warm pool, no Landlock).
>
> **Current set (2026-09-30): 5 main demos + 1 experimental**, in storyline order:
> `demo_cpu_tasks` (tier 2, own `cpu`/`libs` microVM Runtimes, the 9-dataset literature mix by default, with the
> `SCALE=1` harness-scaling preset that replaced the `demo_cpu_scale_lanes` folder);
> `demo_cpu_terminalbench` (tier 3 image pool: one image + VPC Runtime per Terminal-Bench task, 218 deployed of
> 229, a fresh arm64 microVM per episode, look-ahead pre-open `TB_PREOPEN_BATCHES=2` + heartbeat; its
> `terminalbench/warmstart/` probe; split back out of `demo_cpu_tasks` 2026-09-30);
> `demo_cpu_swebench` (tier 3 image pool: 20 SWE-bench Verified instances, Epoch AI arm64 images ≤ 2 GB, a fresh
> microVM per episode); `demo_gpu_single` (KernelBench, strictly one g6.xlarge capacity-provider Instance per
> session; split back out of `demo_gpu_farm` 2026-09-30); `demo_cpu_instance` (strictly one m5.xlarge
> capacity-provider Instance per session); experimental `demo_cpu_branching`. **No demo shares a VM between
> episodes.** Every demo runs the same six step scripts (`1_prepare_data.sh` … `6_cleanup.sh`); see
> `code/demos/README.md`.
>
> *History (2026-09-28 → 09-30):* the set was 3 main demos (`demo_cpu_tasks` incl. `DATASET=terminalbench`,
> `demo_cpu_swebench`, `demo_gpu_farm FARM=fractional|whole`, local or `LOCAL_GPU=0` on a remote g6) + the
> twins `demo_cpu_swebench_eks` / `demo_cpu_swebench_ecs` + branching. `demo_gpu_farm` and both twins were
> **removed 2026-09-30** (code in git history).
>
> **Done (2026-09-28):** `demo_cpu_tasks` runs on our own Runtimes (`cpu` + `libs`, VPC) with the harness
> *inside* the sandbox session; the Code Interpreter survives only as the `compare_backends.py` probe. The
> decoupled harness tier (`harness_pool.py`) is the default rollout topology in every demo.

\* *Hypothesis only — Tier 3 replaces it with a measurement.* Note most cheap-checker CPU tasks are
expected to be **generation-bound**; the clear **sandbox-bound** case is CUDA (Tier 2). That
asymmetry is the point of measuring rather than assuming.

### Tier 2 — GPU instance (capacity-provider EC2 GPU)

For tasks whose *checker itself needs CUDA*. The sandbox is a real AgentCore **GPU Runtime
instance** (a `g6`, x86_64/amd64); the trainer box keeps all its GPUs for policy + inference.

| Demo folder | Task | Notes |
|---|---|---|
| `demo_gpu_kernelbench` → built as **`demo_gpu_single`** | KernelBench (level_1 CUDA) | checker = allclose vs torch ref, on `cuda`; **one g6.xlarge Instance per session, one episode per session** (`SANDBOX=agentcore-runtime`, capacity provider `acs_gpu_cp`). Also carries the custom-Runtime angle (`gpu_torch_smoke.py`) and `SNAPSHOT_EVERY=1` (merged 2026-09-25 from `demo_gpu_customruntime` / `demo_gpu_snapshot`) |

`demo_gpu_single` gives **full isolation**: an episode owns the whole Instance, kernel, disk and L4.
*History:* 2026-09-26 → 09-30 the GPU demo was `demo_gpu_farm` (`FARM=fractional`: 0.5+0.5 of one GPU;
`FARM=whole`: one whole GPU per session; local by default, `LOCAL_GPU=0` on one warm remote g6), a
trusted-only farm (shared kernel + shared GPU, no MIG, cooperative memory cap, one shared uid). It was
**removed 2026-09-30**; `demo_gpu_single` was split back out of it as strictly one session per Instance.

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
   sessions; on GPU, more g6 Instances within quota). Typical for KernelBench.
4. Show the throughput / step-time improvement after scaling the correct side — and that scaling the
   *other* side would not have helped.

There is a **third** tier hiding behind those two: the **harness** that drives the episodes. Rollout
throughput is `min(inference, sandbox, harness)`, so even after inference + sandbox are scaled, a
few coroutines can cap everything. The **decoupled harness tier** (`harness_pool.py`) is therefore the
default in every demo: W dispatch workers (default RB×NS) pull episodes off one queue, each episode takes
its sandbox first and then the least-loaded engine, and the loop itself runs in the session, so the
harness scales with the sandbox rather than the trainer. **`demo_cpu_tasks SCALE=1`** turns all three
tiers up together (4 SGLang engines : 256 harness workers : 256 sessions in flight, DeepScaleR; formerly
the `demo_cpu_scale_lanes` folder, renamed from `demo_cpu_harness_scale` 2026-09-23, 4 × 4 × 4 pinned
lanes until 2026-09-25). The offline `demo_cpu_tasks/harness_scaling_demo.py` shows the knee
(throughput rises ~linearly with `W`, then plateaus at `min(inference, sandbox)`); the per-episode
`qdepth` / `sandbox_wait_ms` / `engine_inflight` fields show which tier binds in a real run. Full
write-up: [harness-scaling.md](harness-scaling.md).

This is why disaggregation matters: each component is sized to its own bottleneck, per task.

### Advanced: keeping a paused sandbox alive across the weight-sync barrier — two mechanisms

During slime's trainer→inference **weight sync**, generation stalls, so open sandbox sessions sit
idle. Surviving that pause is **two patterns on two substrates**:

| Demo | Substrate | Lever | Survives |
|---|---|---|---|
| `HEARTBEAT=1` on any demo (formerly `demo_cpu_heartbeat`; probe [`demo_cpu_tasks/heartbeat_demo.py`](../code/demos/demo_cpu_tasks/heartbeat_demo.py)) | serverless CPU **microVM** (or any warm session) | **heartbeat** (+ quiesce gate) | the *idle timeout* — keep the warm session from being reclaimed during the pause |
| [`demo_gpu_single`](../code/demos/demo_gpu_single/) `SNAPSHOT_EVERY=1` (formerly `demo_gpu_snapshot`; in `demo_gpu_farm` 2026-09-28 → 09-30) | **GPU Runtime instance** | **application-level snapshot & resume** | a genuine *recycle* — restore workspace state onto a fresh session (there is no VM snapshot; [gpu-snapshot.md](gpu-snapshot.md)) |

**Heartbeat keeps a *live* session warm; snapshot restores a *lost* one.** Both have an offline
probe (no AWS/GPU) that runs in seconds (`heartbeat_demo.py`, `snapshot_demo.py`). Reusable mechanisms:
`agentcore_slime_demo/session_keeper.py` (`SessionKeeper` + `KeptSandbox`) and
`agentcore_slime_demo/snapshot_store.py` (`snapshot_workspace`/`restore_workspace`).

#### Demo 1 — microVM heartbeat (`HEARTBEAT=1`; formerly `demo_cpu_heartbeat`)

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
   [`code/demos/demo_gpu_single/`](../code/demos/demo_gpu_single/) with `SNAPSHOT_EVERY=1` (`snapshot_store.py`).

Demo arc: naive "stop invoking" (anti-pattern, silently loses the GPU instance) → heartbeat quiesce
that survives the idle timeout → resume → recover-if-reclaimed. This deliberately re-couples the
sandbox to the trainer's sync barrier — a **cost** optimization, not a correctness fix (async RL
episodes are otherwise designed to span weight versions).

#### Demo 2 — runtime-instance application-level snapshot & resume (`demo_gpu_single SNAPSHOT_EVERY=1`)

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
