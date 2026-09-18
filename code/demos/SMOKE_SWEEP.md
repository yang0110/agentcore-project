# Smoke-sweep experiment plan — all 16 demos, end-to-end

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
- **GPU demos.** No remote g6 required for the smoke: we serve the KernelBench GPU sandbox
  **locally** on GPUs 6,7 via `deploy/run_local_gpu_runtime.sh {single|shared}` and run the trainer
  with `LOCAL_GPU=1` (slime then uses GPUs 0–5). This exercises the same AgentCore `/invocations`
  contract over localhost.
- **Shared code.** All CPU trainers use the identical `run.sh` / in-process-harness shape and differ
  only in dataset + hidden checker, so a per-demo failure is almost always dataset/checker-specific.

## Smoke size

Every trainer runs at `NR=3` ⇒ `NR-1 = 2` real train steps + 1 held-out eval + one final save.
Defaults otherwise (`RB=4`, `NS=8`, `MODE=B` fully-async, `EVAL_INTERVAL=10`, `SAVE_INTERVAL=1e6`
⇒ save-only-at-end). Offline demos run at their default tiny knobs (seconds).

## Phases (executed in this order)

**Phase 0 — offline sims (seconds each, no GPU/AWS).** Driver: `run_smoke_sweep.sh` inside `agentcore_p4`.
- `demo_cpu_quiesce`      — `python3 quiesce_demo.py` (naive→heartbeat→recover arc)
- `demo_gpu_snapshot`     — `python3 snapshot_demo.py` (app-level snapshot/resume across recycle)
- `demo_cpu_harness_scale`— offline `throughput = min(inference, sandbox, harness)` sweep

**Phase 1 — CPU RL trainers (remote AgentCore CPU sandbox, ~10 min each, ckpt deleted after).** Same driver.
`demo_cpu_single`, `demo_cpu_math`, `demo_cpu_mixtask`, `demo_cpu_gsm8k`, `demo_cpu_math500`,
`demo_cpu_deepscaler`, `demo_cpu_humaneval`, `demo_cpu_gretelsql`, `demo_cpu_hotpotqa`,
`demo_cpu_xlam`, `demo_cpu_mcp`.

**Phase 2 — GPU RL trainers (local GPU sandbox on 6,7).** Driven from the host (helper) + container (trainer).
- `demo_gpu_single`  — `run_local_gpu_runtime.sh single`  then `LOCAL_GPU=1 NR=3 run.sh`
- `demo_gpu_shared`  — `run_local_gpu_runtime.sh shared`  then `LOCAL_GPU=1 NR=3 run.sh` (Ray farm on the one local GPU)

## Success criteria

- **Offline demos:** process exits 0.
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

# Phase 2 (GPU), after Phase 1 finishes — for each of single|shared:
bash code/deploy/run_local_gpu_runtime.sh single      # host: start local GPU sandbox on 6,7
docker exec agentcore_p4 bash -lc 'cd /work/code/demos/demo_gpu_single && LOCAL_GPU=1 NR=3 bash run.sh > smoke_run.log 2>&1'
bash code/deploy/run_local_gpu_runtime.sh stop        # host: tear down
```

Read progress with `tail -f demos/smoke_sweep_results.tsv` and each `<demo>/smoke_run.log`.
