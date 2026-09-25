# Smoke-test experiment plan — every demo: 10 train steps · 10 weight syncs · 1 eval · 1 checkpoint (at the end)

**Goal.** Run every existing demo (the 7 main demos + experimental `demo_cpu_branching`, 22 runs in
all) for the same short budget — **10 training steps, 10 weight syncs, exactly 1 held-out eval, and
exactly 1 checkpoint, written at the end** — capturing a
**detailed per-run log** (and parsed CSV/PNG) for each. This is a *smoke* (does it train end-to-end
and produce sane curves?), not a convergence run (that's what `convergence/` is for).

**Every demo is a standalone directory and is treated equally** — there are no tiers. All follow the
one per-run recipe below; the only per-demo differences are captured as data in the demo table
(substrate, whether it needs an ARN/deploy, whether it ships data). Run them one at a time.

## How the spec maps to knobs (all RL trainers share one `run.sh` shape)

| Spec | Knob | Value | Why |
|---|---|---|---|
| **10 train steps** | `NR` (`--num-rollout`) | **`NR=11`** | slime runs `range(1,NR)` = `NR−1` real steps (fresh torch_dist ckpt is iter 0). `NR=11` ⇒ 10 steps. **Already the default.** |
| **10 weight syncs** | *(none)* | automatic | Under the async/decoupled driver (`train_async.py`, no `--colocate`) weights sync to SGLang **once per train step** ⇒ 10 syncs for 10 steps, in both `MODE=A` (stock 1-step-async) and `MODE=B` (fully-async). No user-facing var. |
| **1 eval (near end)** | `EVAL_INTERVAL` / `EVAL_NS` | **`10` / `2`** | With `NR=11`, eval fires once at step 10. **Already the default.** `EVAL_INTERVAL=0` disables. |
| **1 checkpoint, at the end** | `SAVE_INTERVAL`, `SAVE` | **`1000000`**, `$DEMO_DIR/ckpt/...` | Huge interval disables periodic save; explicit `--num-rollout` disables the epoch-boundary save; only slime's forced final-rollout checkpoint is written (≈ 56 GB for Qwen3-4B; delete it after checking). **Already the default.** |

**Net: the spec == each demo's defaults.** Verified 2026-09-25: the `run.sh` of all 8 demos defaults
to `NR=11 EVAL_INTERVAL=10 EVAL_NS=2 SAVE_INTERVAL=1000000`. You do not need to pass anything except
`NR=11` explicitly (for clarity) and whatever sandbox ARN(s) that demo requires (see the table). Do
**not** raise `RB`/`NS` for a smoke except where the table says so (`demo_cpu_scale_lanes` needs
`RB=32` to fill its 256 workers; that is its default).

## Common setup (once)

All trainers run **inside the persistent `agentcore_p4` container** (slimerl/slime), repo mounted at
`/work`, each demo at `DEMO_DIR=/work/code/demos/<demo>`. Requirements:

- **AWS creds** live in the container (`AWS_REGION` default `us-west-2`; `AWS_BEARER_TOKEN_BEDROCK`
  or role creds) — every CPU trainer talks to a **remote** AgentCore sandbox and **rejects
  `SANDBOX=local`**.
- **Sandbox ARNs.** Most CPU demos ship a baked **K=2** default `AGENTCORE_SANDBOX_RUNTIME_ARNS`
  (`acs_sandbox_cpu…`), so they run with no ARN set. The table's **Needs** column flags the demos
  that require an extra deploy / ARN / map before they run.
- **Log directory.**

```bash
docker exec agentcore_p4 bash -lc 'mkdir -p /work/code/demos/_smoke_logs'
```

Each run tees full stdout/stderr to `/work/code/demos/_smoke_logs/<demo>.log` and then parses it to
`<demo>.csv` / `<demo>.eval.csv` / `<demo>.png` via the demo's `metrics.py`.

### The per-run recipe (identical for every demo)

```bash
DEMO=<demo>                       # e.g. demo_cpu_tasks — any dir under code/demos/
DS=                               # demo_cpu_tasks only: DATASET name, e.g. gsm8k (else leave empty)
TAG=$DEMO${DS:+-$DS}              # log label, e.g. demo_cpu_tasks-gsm8k
LOG=/work/code/demos/_smoke_logs/$TAG.log
EXTRA_ENV=()                      # per-demo env from the "Needs" column, e.g. (-e AGENTCORE_SANDBOX_RUNTIME_ARN="$ARN")

# 1. build data (only demos that have prepare_data.sh; others ship data/ already)
docker exec agentcore_p4 bash /work/code/demos/$DEMO/prepare_data.sh 2>/dev/null || true

# 2. run: 10 train + 1 eval, single final ckpt, FULL log tee'd
docker exec -e NR=11 -e EVAL_INTERVAL=10 -e EVAL_NS=2 ${DS:+-e DATASET=$DS} "${EXTRA_ENV[@]}" agentcore_p4 \
    bash -lc "cd /work && bash code/demos/$DEMO/run.sh 2>&1 | tee $LOG"

# 3. parse the log into CSVs + a curve PNG
docker exec agentcore_p4 python3 /work/code/demos/$DEMO/metrics.py $LOG \
    --csv /work/code/demos/_smoke_logs/$TAG.csv \
    --eval-csv /work/code/demos/_smoke_logs/$TAG.eval.csv \
    --plot /work/code/demos/_smoke_logs/$TAG.png
```

`NR/EVAL_INTERVAL/EVAL_NS` are passed explicitly for auditability even though they equal defaults.
`RB=4, NS=8, TEMP=1.0` defaults are fine for a smoke (scale_lanes: `RB=32`, 320 prompts ≤ its 400). Tiny
synthetic-zoo datasets (4–9 rows) wrap into more epochs; `--num-rollout` still keeps it to one save.

**Shared box (GPUs 0/1 taken by other users).** Serialize runs with `flock /tmp/acs_gpu.lock` and pin
slime with `-e SLIME_CUDA_VISIBLE=2,3,4,5,6,7 -e SLIME_NUM_GPUS=6` plus a GPU split that fits 6 GPUs
(column **GPUs** below). Terminal-Bench does not read the pin yet: run it only when all 8 GPUs are free.

---

## The runs: every existing demo (22 trainer runs)

Every demo ships the numbered step scripts (`1_prepare_data.sh` → `2_build_image.sh` → `3_push_ecr.sh`
→ `4_deploy_runtime.sh`, then `5_run.sh`, `6_cleanup.sh`); `4_deploy_runtime.sh` writes
`deploy/runtime.env`, which `5_run.sh` picks up. **Needs** = what to do before the recipe. **GPUs** =
a suggested split for the shared 6-GPU box (only rows 15 and 22 have run with it) (default = 4 train + 4 inference on all 8).

| # | Run (`DEMO[:DATASET][+knobs]`) | Substrate | Needs before running | GPUs on the shared box (suggested) | Last smoke |
|---:|---|---|---|---|---|
| 1 | `demo_cpu_tasks:mbpp` | default Code Interpreter (`SANDBOX=agentcore`), no image | — | 4 + 2 (`ROLLOUT_GPUS=2`) | own-Runtime 0.69 (09-18); CI re-run pending |
| 2 | `demo_cpu_tasks:gsm8k` | Code Interpreter | — | 4 + 2 | **CI 0.90** (09-25) |
| 3 | `demo_cpu_tasks:math` | Code Interpreter | — (synthetic zoo, tiny) | 4 + 2 | own-Runtime 1.00 |
| 4 | `demo_cpu_tasks:math500` | Code Interpreter | — | 4 + 2 | own-Runtime 0.89 |
| 5 | `demo_cpu_tasks:deepscaler` | Code Interpreter | — | 4 + 2 | own-Runtime 0.78 |
| 6 | `demo_cpu_tasks:humaneval` | Code Interpreter | — | 4 + 2 | own-Runtime 0.85 |
| 7 | `demo_cpu_tasks:gretelsql` | Code Interpreter | — | 4 + 2 | own-Runtime 0.60 |
| 8 | `demo_cpu_tasks:hotpotqa` | Code Interpreter | — | 4 + 2 | own-Runtime 0.65 |
| 9 | `demo_cpu_tasks:xlam` | Code Interpreter | — | 4 + 2 | own-Runtime 1.00 |
| 10 | `demo_cpu_tasks:mcp` | Code Interpreter | — | 4 + 2 | own-Runtime run |
| 11 | `demo_cpu_tasks:mixed` | Code Interpreter | — (math + sql + ctf + datasci) | 4 + 2 | own-Runtime run |
| 12 | `demo_cpu_tasks:libs` | Code Interpreter (pandas/scipy/sklearn preinstalled) | — (scores 0 on the stdlib own-Runtime image) | 4 + 2 | **CI 1.0** (09-25) |
| 13 | `demo_cpu_tasks:deepscaler+HEARTBEAT=1+SANDBOX=agentcore-runtime` | own CPU Runtime (baked K=2 ARNs) | — (heartbeat is a no-op on the CI) | 4 + 2 | pending |
| 14 | `demo_cpu_terminalbench` | ~229 per-task images, fresh microVM per episode, pre-open + heartbeat | fetch → `2_build_image.sh` → `4_deploy_runtime.sh` → `validate_oracle.py` → rebuild dataset (see its README); `TB_PREOPEN_BATCHES` × RB × NS (64) sessions booted ahead | **all 8 free** (no pin yet) | passed 09-23: 4/382 solves, `open_wait_ms` p95 0 |
| 15 | `demo_cpu_instance_pool` | warm pool of `SANDBOX_POOL=2` m5 Instances, ~4 GB image, per-episode slot uid | steps 2–4 (image → ECR → CPU capacity provider → Runtime); no baked default | 4 + 2 (as run) | **passed 09-25**: eval 0.7875, 2 cold boots, warm op p50 117 ms |
| 16 | `demo_cpu_scale_lanes` | 4 engines : `HARNESS_WORKERS=256` : `SANDBOX_SESSIONS=256` pooled microVMs, decoupled | — (baked K=2 ARNs; `RB=32` default → 256 episodes/step) | 2 + 4 (`ACTOR_GPUS=2 ROLLOUT_GPUS=4`) | pool check 64 × 64 ok; RL smoke pending (old 4×4×4 lanes: 0.358) |
| 17 | `demo_gpu_single` (remote g6) | ONE warm g6 Instance shared across episodes | steps 2–4 (Runtime on the g6 capacity provider) | 4 + 2 | trained; 09-25 remote smoke failed on `op=open` 500s |
| 18 | `demo_gpu_single+LOCAL_GPU=1` | local whole-GPU farm on GPUs 6,7 (no g6) | `LOCAL_GPU=1 bash 4_deploy_runtime.sh`; `LOCAL_GPU=1 bash 6_cleanup.sh` after | slime on 2–5 (`ACTOR_GPUS=2 ROLLOUT_GPUS=2`) | 0.65 (custom-Runtime probe) |
| 19 | `demo_gpu_single+SNAPSHOT_EVERY=1+LOCAL_GPU=1` | as 18, plus per-episode snapshot/resume | as 18 | as 18 | 0.625 (local farm) |
| 20 | `demo_gpu_farm_fractional` | local fractional Ray farm, GPU 6, 0.5 + 0.5 | host `bash code/deploy/run_local_gpu_runtime.sh fractional`; `… stop` after | slime on 2–5 | 0.65 (09-25) |
| 21 | `demo_gpu_farm_whole` | local whole-GPU Ray farm, GPUs 6,7, one GPU per session | host `bash code/deploy/run_local_gpu_runtime.sh whole`; `… stop` after | slime on 2–5 | 0.65 / 0.70 (09-21); re-run since the split pending |
| 22 | `demo_cpu_branching` (**experimental**) | real Claude Code CLI in a CPU Runtime (VPC, adapter dial-back `:18021`) | steps 2–4 (CLI image), `deploy/runtime.env` with the ARN + `ADAPTER_PUBLIC_HOST` | 4 + 2 | 3 blockers fixed 09-25; RL smoke running |

Rows 1–13 are one folder (`demo_cpu_tasks`) with one run per dataset: set `DS=<dataset>` in the
recipe, and add the `+KEY=VAL` knobs as `-e KEY=VAL`. The remaining synthetic-zoo datasets (`swe`
`tool_use` `sql` `ctf` `rag` `regex` `config` `data` `planning`) are optional extra rows of the same
shape. `run_smoke_plan.sh` drives rows 1–13 + 16 (`PHASE=1`) and 15 (`PHASE=2`);
`run_gpu_smoke.sh` / `run_b_queue.sh` drive 17–21. Old logs from merged folders:
`demo_cpu_tasks/logs/{heartbeat,customruntime,imagepool}/`, `demo_cpu_instance_pool/logs/bigimage/`,
`demo_gpu_single/logs/snapshot/`.

Isolation reminder for the reviewer (what each row's sandbox guarantees): rows 1–12 and 14 get a fresh
microVM per episode; 13 and 16 reuse a microVM **one episode at a time** with a fresh workdir; 15, 17–21
share one Instance across concurrent episodes via per-episode `sub_session`s (slot uid + `0700` dir
where the container has the capabilities; the AgentCore g6 container has none, so it falls back to one
shared uid).

### Offline pre-checks (seconds, no AWS/GPU)

These are part of the plan too, but they are not trainers (no steps / syncs / eval / checkpoint).
A few demos ship offline probes worth running first — they import the harness/glue and verify a
mechanism without AWS: property-demo animations (`demo_cpu_tasks`'s `heartbeat_demo.py` and `compare_backends.py --backends local`, `snapshot_demo.py`,
`harness_scaling_demo.py`, `demo_gpu_single`'s `gpu_torch_smoke.py`), `demo_cpu_terminalbench`'s warm-start probe
(`BACKENDS=local bash warmstart/run_warmstart.sh`), and `demo_cpu_instance_pool`'s `isolation_probe.py`
(per-slot-uid isolation, 16 checks; `--live` on the m5 after step 4), `demo_gpu_farm_fractional`'s `isolation_probe.py` (live on the local farm under the GPU lock), `probe_bigimage.py` (image-size vs the microVM cap + LocalSandbox contract
parity) and `netcheck.py` (safe-networking checklist). These are **not** the trainers — the trainers all run via the recipe above.

### Excluded
- **`convergence/`** — not a runnable demo; it holds per-demo wrappers that run each `train.sh` **to
  convergence** (the opposite of a smoke). Out of scope here.

---

## Running order

There is no tiering, but two ordering facts still hold:

1. **One demo at a time** — each trainer takes all 8 local GPUs (or the 6 pinned ones on the shared
   box), so hold `flock /tmp/acs_gpu.lock` for every trainer and farm probe.
2. **Prerequisites first** — a demo with a non-empty **Needs** cell (branching, `demo_cpu_instance_pool`, terminalbench,
   the g6 GPU demos, the gpu farm demos' host step) must have that deploy/ARN ready before its run. Demos with
   an empty **Needs** cell can run in any order with no extra setup.

A convenient sequence: run the empty-**Needs** demos first (no setup), then the ones whose deploy you
have provisioned, then the g6 GPU demos if a live g6 is available.

## Success criteria (check per log/CSV)

- **Completes 10 steps:** the log shows rollouts `1..10`.
- **Exactly 1 checkpoint:** one checkpoint write under `$SAVE`, at the end (no earlier one — save-at-end held).
- **Exactly 1 eval:** one `eval/<name>` block near step 10.
- **10 weight syncs:** one weight update/sync per train step in the async driver.
- **Sane learning signal:** `rollout/raw_reward` present and non-degenerate (not identically 0 — a
  stuck-at-0 `libs`/datasci family means it ran on the stdlib own-Runtime image instead of the Code Interpreter; in `branching` it can mean
  the CLI harness isn't producing valid submissions).
- **Latency split recorded:** `perf/sandbox_ms_*` vs `perf/sglang_gen_ms_*` columns populated
  (in-process harness demos); confirms the remote sandbox round-trips.
- **Artifacts saved:** `_smoke_logs/<demo>.log` + `.csv` + `.eval.csv` + `.png` exist for every run.
