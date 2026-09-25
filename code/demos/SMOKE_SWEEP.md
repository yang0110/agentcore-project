# Smoke-sweep experiment plan — all 17 demos, end-to-end

> **2026-09-25 (latest):** `demo_cpu_heartbeat` → the opt-in **`HEARTBEAT=1`** knob of every demo (probe
> `demo_cpu_tasks/heartbeat_demo.py`); `demo_cpu_customruntime` → `demo_cpu_tasks/compare_backends.py`;
> `demo_cpu_imagepool` → **`demo_cpu_tasks VARIANT=imagepool`**. The set is now **6 demos** (5 main: 3 CPU
> `demo_cpu_tasks` / `demo_cpu_terminalbench` / `demo_cpu_scale_lanes` + 2 GPU `demo_gpu_single` / `demo_gpu_farm`,
> plus the experimental `demo_cpu_branching`). Driver entries: offline probes `demo_cpu_tasks:heartbeat`,
> `demo_cpu_tasks:compare_backends`; trainers `demo_cpu_tasks:deepscaler+HEARTBEAT=1` (the `+KEY=VAL` suffix adds
> env knobs) and, in `run_smoke_plan.sh` PHASE=2, `demo_cpu_tasks+VARIANT=imagepool`. The log below keeps the old names.

> **2026-09-25 (later):** `demo_cpu_bigimage` → `demo_cpu_tasks VARIANT=bigimage`; `demo_gpu_snapshot` →
> `demo_gpu_single SNAPSHOT_EVERY=1`; `demo_gpu_customruntime` → `demo_gpu_single` (custom-Runtime angle, `LOCAL_GPU=1`).
> The set was then **9 demos** (8 main: 6 CPU + 2 GPU, plus the experimental, blocked `demo_cpu_branching`), each with
> step scripts `1_prepare_data.sh` … `6_cleanup.sh`. The offline probe entry is now `demo_gpu_single` → `snapshot_demo.py`.

> **2026-09-25:** `demo_gpu_shared_single`, `demo_gpu_shared_multi` and `demo_gpu_local` were merged into
> `demo_gpu_farm` (`FARM=fractional|whole`; launcher modes `fractional|whole`, old names still aliased).
> The log below keeps the names the runs used.

> **Historical plan (pre-2026-09-23).** Written when the set had 17 demos. The ten per-dataset CPU
> folders (`demo_cpu_mbpp`, `demo_cpu_math`, `demo_cpu_gsm8k` … `demo_cpu_mcp`) have since been
> merged into ONE demo, `demo_cpu_datasets` (now `demo_cpu_tasks`), selected with `DATASET=`. The set was then **15 demos**
> (9 CPU + 6 GPU); since the 2026-09-25 merges (`demo_gpu_farm`; `demo_cpu_mixtask` folded into `demo_cpu_tasks` as `DATASET=mixed`) it was **12** (6 since the later 2026-09-25 merges; see `README.md`). In `run_smoke_sweep.sh` those trainers are now entries of the form
> `demo_cpu_tasks:<ds>` (e.g. `demo_cpu_tasks:gsm8k`), logged as `demo_cpu_tasks-<ds>`.
> `demo_gpu_shared` was split into `demo_gpu_shared_single` / `demo_gpu_shared_multi` (2026-09-21).

Goal: run a short **smoke** of every demo to catch bugs end-to-end (dataset → harness →
sandbox → slime train step → final checkpoint), and record a per-demo PASS/FAIL with a captured
`smoke_run.log` in each demo dir.

## Constraints (why this is a serial, staged sweep, not a parallel loop)

- **No parallelism.** Every RL trainer claims all/most of the 8 GPUs, so the trainers run **strictly
  serially** (~8–12 min each).
- **Disk.** Each trainer writes a ~56 G final checkpoint. `/` is ~1.9 T and was recently at 100 %.
  The driver **deletes each run's checkpoint immediately after** verifying it, so peak extra usage is
  one checkpoint (~56 G) at a time, never the sum.
- **CPU demos are remote-only.** They run on the pinned CPU AgentCore sandbox ARNs
  (`acs_sandbox_cpu*`), round-robin (K=2). Needs live AWS creds in the container (present).
- **GPU demos.** No remote g6 required for the smoke: we run the standalone on-box demo
  **`demo_gpu_local`**, which serves the KernelBench GPU sandbox **locally** on GPUs 6,7 via
  `deploy/run_local_gpu_runtime.sh local` (a whole-GPU Ray farm: `B4_FARM_GPUS=2`,
  `B4_GPU_FRACTION=1.0` ⇒ 2 whole-GPU slots) while slime uses GPUs 0–5. Its `run.sh` needs **no
  `LOCAL_GPU` flag** — it is always local. This exercises the same AgentCore `/invocations` contract
  over localhost. (The remote-only `demo_gpu_single` / `demo_gpu_shared` need a real g6 and are not
  part of the on-box smoke.)
- **Shared code.** All CPU trainers use the identical `run.sh` / in-process-harness shape and differ
  only in dataset + hidden checker, so a per-demo failure is almost always dataset/checker-specific.

## Smoke size

Every trainer runs at `NR=3` ⇒ `NR-1 = 2` real train steps + 1 held-out eval + one final save.
Defaults otherwise (`RB=4`, `NS=8`, `MODE=B` fully-async, `EVAL_INTERVAL=10`, `SAVE_INTERVAL=1e6`
⇒ save-only-at-end). Offline demos run at their default tiny knobs (seconds).

## Phases (executed in this order)

**Phase 0 — offline PROBES (seconds each, no GPU/AWS).** Driver: `run_smoke_sweep.sh` inside `agentcore_p4`.
Runs each property demo's **retained probe script directly** (NOT `run.sh`, which is now the trainer);
logs to `offline_probe.log`. Behavior markers (not just exit 0) must appear.
- `demo_cpu_heartbeat` (now `demo_cpu_tasks:heartbeat`) — `python3 heartbeat_demo.py` (naive→heartbeat→recover arc)
- `demo_cpu_tasks:compare_backends` (was `demo_cpu_customruntime`, added 2026-09-25) — `python3 compare_backends.py --backends local`
- `demo_gpu_snapshot` (now `demo_gpu_single`) — `python3 snapshot_demo.py` (app-level snapshot/resume across recycle)
- `demo_cpu_scale_lanes`  — `python3 harness_scaling_demo.py …` (formerly `demo_cpu_harness_scale`) (`throughput = min(inference, sandbox, harness)` sweep)

**Phase 1 — CPU RL trainers (remote AgentCore CPU sandbox, ~10 min each, ckpt deleted after).** Same driver.
`demo_cpu_tasks:mbpp`, `demo_cpu_tasks:math`, `demo_cpu_tasks:mixed` (then `demo_cpu_mixtask`), `demo_cpu_tasks:gsm8k`,
`demo_cpu_tasks:math500`, `demo_cpu_tasks:deepscaler`, `demo_cpu_tasks:humaneval`,
`demo_cpu_tasks:gretelsql`, `demo_cpu_tasks:hotpotqa`, `demo_cpu_tasks:xlam`,
`demo_cpu_tasks:mcp` (formerly the ten `demo_cpu_<ds>` folders), plus the three CPU property trainers `demo_cpu_heartbeat`,
`demo_cpu_scale_lanes`, `demo_cpu_customruntime` (each now wraps a real slime run; since 2026-09-25 the
entries are `demo_cpu_tasks:deepscaler+HEARTBEAT=1` and `demo_cpu_scale_lanes` — customruntime was the same
deepscaler run on a custom Runtime and was dropped). The two GPU
property trainers (`demo_gpu_snapshot`, `demo_gpu_customruntime`) need a live g6 ARN → run with the
GPU trainers, not in this on-box sweep.

**Phase 2 — GPU RL trainer (local whole-GPU farm on 6,7).** Driven from the host (helper) + container (trainer).
- `demo_gpu_local`  — `run_local_gpu_runtime.sh local`  then `NR=3 run.sh` (whole-GPU Ray farm on the 2 local GPUs; no `LOCAL_GPU` flag)

## Success criteria

- **Offline probes:** probe script exits 0 **and** all per-demo behavior markers appear in
  `offline_probe.log` (e.g. quiesce's `>> RECOVERED`, snapshot's `2 rework step(s) saved`).
- **RL trainers:** the ray job log line `Job '...' succeeded` is present, AND the log has **no**
  `No space` / `CheckpointException` / `unexpected pos`, AND a final `iter_*` checkpoint dir was
  written (`latest_checkpointed_iteration.txt` present) before cleanup.
- Benign, ignored: trailing `resource_tracker: ... KeyError '/mp-*'` tracebacks (shutdown noise).

## Artifacts

- Per-demo captured log: `<demo>/smoke_run.log`.
- Consolidated results: `demos/smoke_sweep_results.tsv` (columns: demo, status, secs, note),
  appended live as each demo finishes.

## How to run

```bash
# Phase 0+1 (offline + CPU trainers), inside the container, in the background:
docker exec agentcore_p4 bash -lc 'nohup bash /work/code/demos/run_smoke_sweep.sh > /work/code/demos/smoke_sweep.driver.log 2>&1 & echo $!'

# Phase 2 (GPU), after Phase 1 finishes — the standalone on-box demo:
bash code/deploy/run_local_gpu_runtime.sh local       # host: start local whole-GPU farm on 6,7
docker exec agentcore_p4 bash -lc 'cd /work/code/demos/demo_gpu_local && NR=3 bash run.sh > smoke_run.log 2>&1'
bash code/deploy/run_local_gpu_runtime.sh stop        # host: tear down
```

Read progress with `tail -f demos/smoke_sweep_results.tsv` and each `<demo>/smoke_run.log`.
