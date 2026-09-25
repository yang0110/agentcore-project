# Smoke-test experiment plan — every demo: 10 train steps · ~10 weight syncs · 1 eval · save-at-end

**Goal.** Run each RL-training demo for a short, identical budget — **10 training steps, ~10 weight
syncs, exactly 1 held-out eval, and a single checkpoint written only at the end** — capturing a
**detailed per-run log** (and parsed CSV/PNG) for each. This is a *smoke* (does it train end-to-end
and produce sane curves?), not a convergence run (that's what `convergence/` is for).

**Every demo is a standalone directory and is treated equally** — there are no tiers. All follow the
one per-run recipe below; the only per-demo differences are captured as data in the demo table
(substrate, whether it needs an ARN/deploy, whether it ships data). Run them one at a time.

## How the spec maps to knobs (all RL trainers share one `run.sh` shape)

| Spec | Knob | Value | Why |
|---|---|---|---|
| **10 train steps** | `NR` (`--num-rollout`) | **`NR=11`** | slime runs `range(1,NR)` = `NR−1` real steps (fresh torch_dist ckpt is iter 0). `NR=11` ⇒ 10 steps. **Already the default.** |
| **~10 weight syncs** | *(none)* | automatic | Under the async/decoupled driver (`train_async.py`, no `--colocate`) weights sync to SGLang **once per rollout step** ⇒ ~10 syncs for 10 steps. No user-facing var. `MODE=B` (default) = fully-async. |
| **1 eval (near end)** | `EVAL_INTERVAL` / `EVAL_NS` | **`10` / `2`** | With `NR=11`, eval fires once at step 10. **Already the default.** `EVAL_INTERVAL=0` disables. |
| **Save only at end** | `SAVE_INTERVAL`, `SAVE` | **`1000000`**, `$DEMO_DIR/ckpt/...` | Huge interval disables periodic save; explicit `--num-rollout` disables the epoch-boundary save; only slime's forced final-rollout checkpoint is written. **Already the default.** |

**Net: the spec == each demo's defaults.** You do not need to pass anything except `NR=11`
explicitly (for clarity) and whatever sandbox ARN(s) that demo requires (see the table).

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
`RB=4, NS=8, CONC=4, TEMP=1.0` defaults are fine for a smoke. Keep `NR × RB ≤ 100` (all demos ≥ 100).

---

## The demos (flat, equal — one row each)

**Needs** column = what to do *before* the standard recipe. Every demo now ships the numbered step scripts (`1_prepare_data.sh` → `2_build_image.sh` → `3_push_ecr.sh` → `4_deploy_runtime.sh`, then `5_run.sh`, `6_cleanup.sh`; see `README.md`), so a Needs cell usually just means "run steps 1–4 first" — `4_deploy_runtime.sh` writes `deploy/runtime.env`, which `5_run.sh` picks up. When it says an ARN/map, you can also pass it via
`EXTRA_ENV` (recipe step 2). Blank Needs = just run the recipe (baked ARN, ships `data/` or has `prepare_data.sh`).

| Demo (standalone dir) | Substrate | Needs before running |
|---|---|---|
| `demo_cpu_tasks` (one run per `DATASET`: mbpp gsm8k math math500 deepscaler humaneval gretelsql hotpotqa xlam mcp mixed) | CPU sandbox (baked ARNs) | — (set `DS=<dataset>`; replaces the ten former `demo_cpu_<ds>` folders, merged 2026-09-23 as `demo_cpu_datasets`, and `demo_cpu_mixtask` = `DS=mixed`, merged 2026-09-25 — older `_smoke_logs/demo_cpu_<ds>.*` / `demo_cpu_datasets-*` / `demo_cpu_mixtask.*` files are those runs). **`VARIANT=bigimage`** (formerly `demo_cpu_bigimage`, merged 2026-09-25): a ~4 GB image on a CPU **capacity-provider Instance** — run `VARIANT=bigimage bash 2_build_image.sh` → `3_push_ecr.sh` → `4_deploy_runtime.sh` (image → ECR → CPU CP → Runtime → `deploy/runtime.bigimage.env`) first; no baked default. Old bigimage logs are in `demo_cpu_tasks/logs/bigimage/`. **`VARIANT=imagepool`** (formerly `demo_cpu_imagepool`, merged 2026-09-25): a pool of 3 custom images (pandas/scipy/sklearn), one Runtime each — run `VARIANT=imagepool bash 2_build_image.sh` → `3_push_ecr.sh` → `4_deploy_runtime.sh` first (→ `deploy/runtime.imagepool.env` with `AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE`); without the map, custom-image families import-fail → reward 0. **`HEARTBEAT=1`** (formerly `demo_cpu_heartbeat`, now an opt-in knob of every demo): e.g. `DS=deepscaler` + `-e HEARTBEAT=1`. The custom-vs-managed Runtime angle (formerly `demo_cpu_customruntime`) is the offline `compare_backends.py`. Old logs: `demo_cpu_tasks/logs/{heartbeat,customruntime,imagepool}/`. |
| `demo_cpu_scale_lanes` (formerly `demo_cpu_harness_scale`) | CPU sandbox (baked ARNs), 4 persistent microVMs | — (`LANES=4`: 4 workers : 4 TP1 engines on rollout GPUs 4–7 : 4 microVMs; the old smoke ran the pre-rework topology) |
| `demo_cpu_branching` (**experimental, blocked**) | real Claude Code CLI in a CPU Runtime | steps 2–4 deploy the CLI-capable image, or pass `-e AGENTCORE_SANDBOX_RUNTIME_ARN="$CLI_ARN"`. First run = bring-up (README header stale; trust `run.sh`). **Blocked:** the in-sandbox CLI exits 1 before any model call → no trajectories. |
| `demo_cpu_terminalbench` | pool of per-task CPU images (~229 Runtimes), fresh microVM per episode | `fetch_terminalbench.py`, then `deploy/deploy_terminalbench.py` (one image + Runtime per task; `TB_LIMIT=2` to validate), pass the resulting `AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE` map (see its README); run `validate_oracle.py` and rebuild the dataset from usable tasks. Look-ahead pre-open keeps `TB_PREOPEN_BATCHES` × RB × NS (default 64) sessions booted — check the account's concurrent-session quota first. First smoke passed 2026-09-23 (4/382 solves, `open_wait_ms` p95 0). Its `SessionKeeper` is always on. |
| `demo_gpu_farm` (`FARM=whole`) | LOCAL whole-GPU Ray farm (GPUs 6,7, 1 GPU/session) — was `demo_gpu_local` / `demo_gpu_shared_multi` | two-step: host `bash code/deploy/run_local_gpu_runtime.sh whole` first; run with `-e FARM=whole`; sync save; `bash …/run_local_gpu_runtime.sh stop` after. Warm-shared (`B4_SHARED_RUNTIME_SESSION`). Ships `data/`. |
| `demo_gpu_farm` (`FARM=fractional`) | LOCAL fractional-GPU Ray farm (1 GPU, 2×0.5 actors share it) — was `demo_gpu_shared_single` | two-step: host `bash code/deploy/run_local_gpu_runtime.sh fractional` first; run with `-e FARM=fractional` (default); `…/run_local_gpu_runtime.sh stop` after. Warm-shared (`B4_SHARED_RUNTIME_SESSION`). Ships `data/`. |
| `demo_gpu_single` (incl. `SNAPSHOT_EVERY=1` and the custom-Runtime angle — formerly `demo_gpu_snapshot` / `demo_gpu_customruntime`, merged 2026-09-25) | ONE warm g6 custom Runtime shared across episodes (remote) — OR local whole-GPU farm for on-box smoke | **Remote:** steps 2–4 (`4_deploy_runtime.sh` → Runtime on the g6 CP → `deploy/runtime.env`), then `5_run.sh`. **On-box smoke (no g6 needed):** `LOCAL_GPU=1 bash 4_deploy_runtime.sh` (starts the local whole-GPU farm on 6,7), `LOCAL_GPU=1 bash 5_run.sh` (slime on 0–5, sync save), `LOCAL_GPU=1 bash 6_cleanup.sh`. Warm-shared (`B4_SHARED_RUNTIME_SESSION`) either way. Add `SNAPSHOT_EVERY=1` for the snapshot/resume angle (default 0 = off; the store works on the local farm too). `gpu_torch_smoke.py` is the offline CUDA-contract probe. (A former customruntime on-box attempt OOM-killed the local trainer — reduce train GPUs / `--max-tokens-per-gpu` if it recurs.) |

### Offline pre-checks (seconds, no AWS/GPU)

A few demos ship offline probes worth running first — they import the harness/glue and verify a
mechanism without AWS: property-demo animations (`demo_cpu_tasks`'s `heartbeat_demo.py` and `compare_backends.py --backends local`, `snapshot_demo.py`,
`harness_scaling_demo.py`, `demo_gpu_single`'s `gpu_torch_smoke.py`), and `demo_cpu_tasks`'s `probe_bigimage.py`
(image-size vs the microVM cap + LocalSandbox contract parity) and `netcheck.py` (safe-networking
checklist). These are **not** the trainers — the trainers all run via the recipe above.

### Excluded
- **`convergence/`** — not a runnable demo; it holds per-demo wrappers that run each `train.sh` **to
  convergence** (the opposite of a smoke). Out of scope here.

---

## Running order

There is no tiering, but two ordering facts still hold:

1. **One demo at a time** — each trainer reserves all 8 local GPUs (4 train + 4 inference).
2. **Prerequisites first** — a demo with a non-empty **Needs** cell (branching, `demo_cpu_tasks VARIANT=bigimage|imagepool`, terminalbench,
   the g6 GPU demos, gpu_farm's host step) must have that deploy/ARN ready before its run. Demos with
   an empty **Needs** cell can run in any order with no extra setup.

A convenient sequence: run the empty-**Needs** demos first (no setup), then the ones whose deploy you
have provisioned, then the g6 GPU demos if a live g6 is available.

## Success criteria (check per log/CSV)

- **Completes 10 steps:** the log shows rollouts `1..10` and a final checkpoint write under `$SAVE`
  (and no earlier checkpoint — save-at-end held).
- **Exactly 1 eval:** one `eval/<name>` block near step 10.
- **~10 weight syncs:** one weight update/sync per rollout step in the async driver.
- **Sane learning signal:** `rollout/raw_reward` present and non-degenerate (not identically 0 — a
  stuck-at-0 family in `VARIANT=imagepool`/`DATASET=mixed` means a mis-mapped image ARN; in `branching` it can mean
  the CLI harness isn't producing valid submissions).
- **Latency split recorded:** `perf/sandbox_ms_*` vs `perf/sglang_gen_ms_*` columns populated
  (in-process harness demos); confirms the remote sandbox round-trips.
- **Artifacts saved:** `_smoke_logs/<demo>.log` + `.csv` + `.eval.csv` + `.png` exist for every run.
