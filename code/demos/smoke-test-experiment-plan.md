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
DEMO=<demo>                       # e.g. demo_cpu_gsm8k — any dir under code/demos/
LOG=/work/code/demos/_smoke_logs/$DEMO.log
EXTRA_ENV=()                      # per-demo env from the "Needs" column, e.g. (-e AGENTCORE_SANDBOX_RUNTIME_ARN="$ARN")

# 1. build data (only demos that have prepare_data.sh; others ship data/ already)
docker exec agentcore_p4 bash /work/code/demos/$DEMO/prepare_data.sh 2>/dev/null || true

# 2. run: 10 train + 1 eval, single final ckpt, FULL log tee'd
docker exec -e NR=11 -e EVAL_INTERVAL=10 -e EVAL_NS=2 "${EXTRA_ENV[@]}" agentcore_p4 \
    bash -lc "cd /work && bash code/demos/$DEMO/run.sh 2>&1 | tee $LOG"

# 3. parse the log into CSVs + a curve PNG
docker exec agentcore_p4 python3 /work/code/demos/$DEMO/metrics.py $LOG \
    --csv /work/code/demos/_smoke_logs/$DEMO.csv \
    --eval-csv /work/code/demos/_smoke_logs/$DEMO.eval.csv \
    --plot /work/code/demos/_smoke_logs/$DEMO.png
```

`NR/EVAL_INTERVAL/EVAL_NS` are passed explicitly for auditability even though they equal defaults.
`RB=4, NS=8, CONC=4, TEMP=1.0` defaults are fine for a smoke. Keep `NR × RB ≤ 100` (all demos ≥ 100).

---

## The demos (flat, equal — one row each)

**Needs** column = what to do *before* the standard recipe. When it says an ARN/map, pass it via
`EXTRA_ENV` (step 2). Blank Needs = just run the recipe (baked ARN, ships `data/` or has `prepare_data.sh`).

| Demo (standalone dir) | Substrate | Needs before running |
|---|---|---|
| `demo_cpu_mbpp` | CPU sandbox (baked ARNs) | — |
| `demo_cpu_gsm8k` | CPU sandbox (baked ARNs) | — |
| `demo_cpu_math` | CPU sandbox (baked ARNs) | — |
| `demo_cpu_math500` | CPU sandbox (baked ARNs) | — |
| `demo_cpu_deepscaler` | CPU sandbox (baked ARNs) | — |
| `demo_cpu_humaneval` | CPU sandbox (baked ARNs) | — |
| `demo_cpu_gretelsql` | CPU sandbox (baked ARNs) | — |
| `demo_cpu_hotpotqa` | CPU sandbox (baked ARNs) | — |
| `demo_cpu_xlam` | CPU sandbox (baked ARNs) | — |
| `demo_cpu_mcp` | CPU sandbox (baked ARNs) | — |
| `demo_cpu_mixtask` | CPU sandbox (baked ARNs) | — |
| `demo_cpu_heartbeat` | CPU sandbox (baked ARNs) | — (heartbeat on by default: `HEARTBEAT_KEEP_WARM=1`) |
| `demo_cpu_harness_scale` | CPU sandbox (baked ARNs) | — (`HARNESS_WORKERS` warm pool) |
| `demo_cpu_customruntime` | your custom CPU Runtime (`SANDBOX=agentcore-runtime`) | — (baked custom-Runtime ARN) |
| `demo_cpu_imagepool` | pool of custom CPU images | deploy 3 images (pandas/scipy/sklearn), pass `-e AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE='{...}'`. Without it, custom-image families import-fail → reward 0. |
| `demo_cpu_branching` | real Claude Code CLI in a CPU Runtime | deploy a CLI-capable image, pass `-e AGENTCORE_SANDBOX_RUNTIME_ARN="$CLI_ARN"`. First run = bring-up (README header stale; trust `run.sh`). |
| `demo_cpu_bigimage` | large image (>2 GB) on a CPU **capacity-provider Instance** | `python3 code/demos/demo_cpu_bigimage/deploy/deploy_bigimage.py` (builds image → ECR → CPU CP → Runtime), then pass `-e AGENTCORE_SANDBOX_RUNTIME_ARN="$BIGIMG_ARN"`. No baked default. |
| `demo_gpu_local` | LOCAL whole-GPU Ray farm (GPUs 6,7) | two-step: host `bash code/deploy/run_local_gpu_runtime.sh local` first; sync save; `bash …/run_local_gpu_runtime.sh stop` after. Ships `data/`. |
| `demo_gpu_shared_single` | LOCAL fractional-GPU Ray farm (1 GPU, 2×0.5 actors share it) | two-step: host `bash code/deploy/run_local_gpu_runtime.sh shared_single` first; `…/run_local_gpu_runtime.sh stop` after. Warm-shared (`B4_SHARED_RUNTIME_SESSION`). Ships `data/`. |
| `demo_gpu_shared_multi` | LOCAL whole-GPU Ray farm (GPUs 6,7, 1 GPU/session) | two-step: host `bash code/deploy/run_local_gpu_runtime.sh shared_multi` first; `…/run_local_gpu_runtime.sh stop` after. Warm-shared (`B4_SHARED_RUNTIME_SESSION`). Ships `data/`. |
| `demo_gpu_single` | ONE warm g6 shared across episodes (remote) — OR local farm for on-box smoke | **On-box smoke (no g6 needed):** two-step like `demo_gpu_local` — host `bash code/deploy/run_local_gpu_runtime.sh local`, then run with `-e LOCAL_GPU=1`, `…/run_local_gpu_runtime.sh stop` after. **Remote:** live g6 ARN instead. Warm-shared (`B4_SHARED_RUNTIME_SESSION`). |
| `demo_gpu_snapshot` | ONE warm g6 + app-level snapshot/resume (remote) — OR local farm for on-box smoke | **On-box smoke (no g6 needed):** two-step like `demo_gpu_local` — host `run_local_gpu_runtime.sh local`, then run with `-e LOCAL_GPU=1 -e SNAPSHOT_EVERY=1`, stop after. **Remote:** live g6 ARN. Warm-shared (`B4_SHARED_RUNTIME_SESSION`). Snapshot/resume persists to the store, so it works on the local farm too. |
| `demo_gpu_customruntime` | ONE warm remote g6 custom Runtime (KernelBench CUDA) — OR local farm for on-box smoke | **On-box smoke (no g6 needed):** two-step like `demo_gpu_local` — host `bash code/deploy/run_local_gpu_runtime.sh local` first, then run with `-e LOCAL_GPU=1` (local whole-GPU farm on GPUs 6,7, slime on 0-5), `…/run_local_gpu_runtime.sh stop` after. **Remote:** live g6 ARN instead. Warm-shared (`B4_SHARED_RUNTIME_SESSION`) either way. (Last on-box attempt OOM-killed the local trainer — reduce train GPUs / `--max-tokens-per-gpu` if it recurs.) |

### Offline pre-checks (seconds, no AWS/GPU)

A few demos ship offline probes worth running first — they import the harness/glue and verify a
mechanism without AWS: property-demo animations (`heartbeat_demo.py`, `snapshot_demo.py`,
`harness_scaling_demo.py`, `gpu_torch_smoke.py`), and `demo_cpu_bigimage`'s `probe_bigimage.py`
(image-size vs the microVM cap + LocalSandbox contract parity) and `netcheck.py` (safe-networking
checklist). These are **not** the trainers — the trainers all run via the recipe above.

### Excluded
- **`convergence/`** — not a runnable demo; it holds per-demo wrappers that run each `train.sh` **to
  convergence** (the opposite of a smoke). Out of scope here.

---

## Running order

There is no tiering, but two ordering facts still hold:

1. **One demo at a time** — each trainer reserves all 8 local GPUs (4 train + 4 inference).
2. **Prerequisites first** — a demo with a non-empty **Needs** cell (imagepool, branching, bigimage,
   the g6 GPU demos, gpu_local's host step) must have that deploy/ARN ready before its run. Demos with
   an empty **Needs** cell can run in any order with no extra setup.

A convenient sequence: run the empty-**Needs** demos first (no setup), then the ones whose deploy you
have provisioned, then the g6 GPU demos if a live g6 is available.

## Success criteria (check per log/CSV)

- **Completes 10 steps:** the log shows rollouts `1..10` and a final checkpoint write under `$SAVE`
  (and no earlier checkpoint — save-at-end held).
- **Exactly 1 eval:** one `eval/<name>` block near step 10.
- **~10 weight syncs:** one weight update/sync per rollout step in the async driver.
- **Sane learning signal:** `rollout/raw_reward` present and non-degenerate (not identically 0 — a
  stuck-at-0 family in `imagepool`/`mixtask` means a mis-mapped image ARN; in `branching` it can mean
  the CLI harness isn't producing valid submissions).
- **Latency split recorded:** `perf/sandbox_ms_*` vs `perf/sglang_gen_ms_*` columns populated
  (in-process harness demos); confirms the remote sandbox round-trips.
- **Artifacts saved:** `_smoke_logs/<demo>.log` + `.csv` + `.eval.csv` + `.png` exist for every run.
