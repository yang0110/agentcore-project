# Smoke-test experiment plan — every demo: 10 train steps · ~10 weight syncs · 1 eval · save-at-end

**Goal.** Run each RL-training demo for a short, identical budget — **10 training steps, ~10 weight
syncs, exactly 1 held-out eval, and a single checkpoint written only at the end** — capturing a
**detailed per-run log** (and parsed CSV/PNG) for each. This is a *smoke* (does it train end-to-end
and produce sane curves?), not a convergence run (that's what `convergence/` is for).

## How the spec maps to knobs (all RL trainers share one `run.sh` shape)

| Spec | Knob | Value | Why |
|---|---|---|---|
| **10 train steps** | `NR` (`--num-rollout`) | **`NR=11`** | slime runs `range(1,NR)` = `NR−1` real steps (fresh torch_dist ckpt is iter 0). `NR=11` ⇒ 10 steps. **Already the default.** |
| **~10 weight syncs** | *(none)* | automatic | Under the async/decoupled driver (`train_async.py`, no `--colocate`) weights sync to SGLang **once per rollout step** ⇒ ~10 syncs for 10 steps. No user-facing var. `MODE=B` (default) = fully-async. |
| **1 eval (near end)** | `EVAL_INTERVAL` / `EVAL_NS` | **`10` / `2`** | With `NR=11`, eval fires once at step 10. **Already the default.** `EVAL_INTERVAL=0` disables. |
| **Save only at end** | `SAVE_INTERVAL`, `SAVE` | **`1000000`**, `$DEMO_DIR/ckpt/...` | Huge interval disables periodic save; explicit `--num-rollout` disables the epoch-boundary save; only slime's forced final-rollout checkpoint is written. **Already the default.** |

**Net: the spec == each demo's defaults.** You do not need to pass anything except `NR=11`
explicitly (for clarity) and the sandbox ARN(s) each demo requires. `run_smoke_sweep.sh` currently
uses `NR=3` (2-step smoke) — this plan bumps that to `NR=11`.

## Common setup (once)

All trainers run **inside the persistent `agentcore_p4` container** (slimerl/slime), repo mounted at
`/work`, each demo at `DEMO_DIR=/work/code/demos/<demo>`. Requirements:

- **AWS creds** live in the container (`AWS_REGION` default `us-west-2`; `AWS_BEARER_TOKEN_BEDROCK`
  or role creds) — every CPU trainer talks to a **remote** AgentCore sandbox and **rejects
  `SANDBOX=local`**.
- **CPU sandbox Runtime ARNs.** The standard CPU demos ship a baked **K=2** default
  `AGENTCORE_SANDBOX_RUNTIME_ARNS` (`acs_sandbox_cpu…`), so they run with no ARN set. Two demos need
  extra routing (below): `demo_cpu_imagepool` (image→ARN map) and `demo_cpu_branching` (a single
  CLI-capable ARN).
- **Log directory.** Write one detailed log per run:

```bash
docker exec agentcore_p4 bash -lc 'mkdir -p /work/code/demos/_smoke_logs'
```

Each run below tees full stdout/stderr to `/work/code/demos/_smoke_logs/<demo>.log` and then parses
it to `<demo>.csv` / `<demo>.eval.csv` / `<demo>.png` via the demo's `metrics.py`.

### Per-run recipe (the pattern every trainer follows)

```bash
DEMO=<demo>                       # e.g. demo_cpu_gsm8k
LOG=/work/code/demos/_smoke_logs/$DEMO.log

# 1. build data (only demos that have prepare_data.sh)
docker exec agentcore_p4 bash /work/code/demos/$DEMO/prepare_data.sh

# 2. run: 10 train + 1 eval, single final ckpt, FULL log tee'd
docker exec -e NR=11 -e EVAL_INTERVAL=10 -e EVAL_NS=2 agentcore_p4 \
    bash -lc "cd /work && bash code/demos/$DEMO/run.sh 2>&1 | tee $LOG"

# 3. parse the log into CSVs + a curve PNG (detailed metrics artifact)
docker exec agentcore_p4 python3 /work/code/demos/$DEMO/metrics.py $LOG \
    --csv /work/code/demos/_smoke_logs/$DEMO.csv \
    --eval-csv /work/code/demos/_smoke_logs/$DEMO.eval.csv \
    --plot /work/code/demos/_smoke_logs/$DEMO.png
```

`NR/EVAL_INTERVAL/EVAL_NS` are passed explicitly for auditability even though they equal defaults.

---

## Group A — standard CPU per-task trainers (baked K=2 ARNs; no extra env)

Demos: `demo_cpu_mbpp`, `demo_cpu_gsm8k`, `demo_cpu_math`, `demo_cpu_math500`,
`demo_cpu_deepscaler`, `demo_cpu_humaneval`, `demo_cpu_gretelsql`, `demo_cpu_hotpotqa`,
`demo_cpu_xlam`, `demo_cpu_mcp`.

Each follows the per-run recipe verbatim. Example (`demo_cpu_gsm8k`):

```bash
docker exec agentcore_p4 bash /work/code/demos/demo_cpu_gsm8k/prepare_data.sh
docker exec -e NR=11 -e EVAL_INTERVAL=10 -e EVAL_NS=2 agentcore_p4 \
    bash -lc 'cd /work && bash code/demos/demo_cpu_gsm8k/run.sh 2>&1 | tee /work/code/demos/_smoke_logs/demo_cpu_gsm8k.log'
docker exec agentcore_p4 python3 /work/code/demos/demo_cpu_gsm8k/metrics.py \
    /work/code/demos/_smoke_logs/demo_cpu_gsm8k.log \
    --csv /work/code/demos/_smoke_logs/demo_cpu_gsm8k.csv \
    --eval-csv /work/code/demos/_smoke_logs/demo_cpu_gsm8k.eval.csv \
    --plot /work/code/demos/_smoke_logs/demo_cpu_gsm8k.png
```

Swap `demo_cpu_gsm8k` for each other name. All reject `SANDBOX=local`; none needs an ARN set (baked
default). `RB=4, NS=8, CONC=4, TEMP=1.0` are fine at defaults for a smoke.

---

## Group B — special CPU trainers (extra sandbox requirement)

### `demo_cpu_mixtask` (mixed families on one image; baked K=2 ARNs)
Runs like Group A — no extra env.
```bash
docker exec agentcore_p4 bash /work/code/demos/demo_cpu_mixtask/prepare_data.sh
docker exec -e NR=11 -e EVAL_INTERVAL=10 -e EVAL_NS=2 agentcore_p4 \
    bash -lc 'cd /work && bash code/demos/demo_cpu_mixtask/run.sh 2>&1 | tee /work/code/demos/_smoke_logs/demo_cpu_mixtask.log'
docker exec agentcore_p4 python3 /work/code/demos/demo_cpu_mixtask/metrics.py \
    /work/code/demos/_smoke_logs/demo_cpu_mixtask.log \
    --csv /work/code/demos/_smoke_logs/demo_cpu_mixtask.csv \
    --eval-csv /work/code/demos/_smoke_logs/demo_cpu_mixtask.eval.csv \
    --plot /work/code/demos/_smoke_logs/demo_cpu_mixtask.png
```

### `demo_cpu_imagepool` (pool of images; **requires** an image→ARN map)
Deploy the three sandbox images first (pandas/scipy/sklearn), then pass the map. **Without routing,
every custom-image family fails its `import` and scores reward 0** (the run.sh guard also errors).
```bash
export ARN_MAP='{"datasci":"arn:aws:bedrock-agentcore:us-west-2:ACCT:runtime/acs_sandbox_datasci-XXXX","scipy":"arn:aws:bedrock-agentcore:us-west-2:ACCT:runtime/acs_sandbox_scipy-XXXX","sklearn":"arn:aws:bedrock-agentcore:us-west-2:ACCT:runtime/acs_sandbox_sklearn-XXXX"}'
docker exec agentcore_p4 bash /work/code/demos/demo_cpu_imagepool/prepare_data.sh
docker exec -e NR=11 -e EVAL_INTERVAL=10 -e EVAL_NS=2 \
    -e AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE="$ARN_MAP" agentcore_p4 \
    bash -lc 'cd /work && bash code/demos/demo_cpu_imagepool/run.sh 2>&1 | tee /work/code/demos/_smoke_logs/demo_cpu_imagepool.log'
docker exec agentcore_p4 python3 /work/code/demos/demo_cpu_imagepool/metrics.py \
    /work/code/demos/_smoke_logs/demo_cpu_imagepool.log \
    --csv /work/code/demos/_smoke_logs/demo_cpu_imagepool.csv \
    --eval-csv /work/code/demos/_smoke_logs/demo_cpu_imagepool.eval.csv \
    --plot /work/code/demos/_smoke_logs/demo_cpu_imagepool.png
```
(Build/deploy of the three images: `for img in datasci scipy sklearn; do docker build -f
code/demos/demo_cpu_imagepool/deploy/Dockerfile.sandbox.$img -t <ecr>/acs-sandbox-$img:latest
code/demos/demo_cpu_imagepool/; done`, push to ECR, register one Runtime each — see the demo README.)

### `demo_cpu_branching` (real Claude Code CLI harness; **requires** a single CLI-capable ARN)
One episode fans out to many training samples via the real Claude Code CLI in the sandbox; needs a
CLI-capable sandbox image (no baked default ARN).
```bash
export CLI_ARN='arn:aws:bedrock-agentcore:us-west-2:ACCT:runtime/acs_sandbox_cli-XXXX'
docker exec agentcore_p4 bash /work/code/demos/demo_cpu_branching/prepare_data.sh
docker exec -e NR=11 -e EVAL_INTERVAL=10 -e EVAL_NS=2 \
    -e AGENTCORE_SANDBOX_RUNTIME_ARN="$CLI_ARN" agentcore_p4 \
    bash -lc 'cd /work && bash code/demos/demo_cpu_branching/run.sh 2>&1 | tee /work/code/demos/_smoke_logs/demo_cpu_branching.log'
docker exec agentcore_p4 python3 /work/code/demos/demo_cpu_branching/metrics.py \
    /work/code/demos/_smoke_logs/demo_cpu_branching.log \
    --csv /work/code/demos/_smoke_logs/demo_cpu_branching.csv \
    --eval-csv /work/code/demos/_smoke_logs/demo_cpu_branching.eval.csv \
    --plot /work/code/demos/_smoke_logs/demo_cpu_branching.png
```
> ⚠️ Verify `demo_cpu_branching` is a working demo before relying on this run (its README header is a
> stale copy; trust `run.sh`). Treat its first smoke as a *bring-up*, not a regression.

---

## Group C — GPU trainers (KernelBench; need a GPU substrate)

### `demo_gpu_local` — the on-box GPU smoke (LOCAL whole-GPU Ray farm, GPUs 6,7)
**Two-step launch.** Data ships in `data/` (no `prepare_data.sh`). Uses **sync save** (local runtime
seccomp blocks async `pidfd_getfd`). `SANDBOX=agentcore-ray-farm` (always local; rejects nothing).
```bash
# 1. HOST: serve the local GPU sandbox on reserved GPUs 6,7 (SKIP_BUILD=1 to reuse the image)
bash code/deploy/run_local_gpu_runtime.sh local
# 2. CONTAINER: 10 train + 1 eval, log tee'd
docker exec -e NR=11 -e EVAL_INTERVAL=10 -e EVAL_NS=2 agentcore_p4 \
    bash -lc 'cd /work/code/demos/demo_gpu_local && bash run.sh 2>&1 | tee /work/code/demos/_smoke_logs/demo_gpu_local.log'
docker exec agentcore_p4 python3 /work/code/demos/demo_gpu_local/metrics.py \
    /work/code/demos/_smoke_logs/demo_gpu_local.log \
    --csv /work/code/demos/_smoke_logs/demo_gpu_local.csv \
    --eval-csv /work/code/demos/_smoke_logs/demo_gpu_local.eval.csv \
    --plot /work/code/demos/_smoke_logs/demo_gpu_local.png
# 3. HOST teardown
bash code/deploy/run_local_gpu_runtime.sh stop
```
Farm defaults: `B4_FARM_GPUS=2`, `B4_GPU_FRACTION=1.0` ⇒ 2 whole-GPU slots. Keep `NR × RB ≤ 100`.

### `demo_gpu_shared` — ONE g6 shared by an in-instance Ray fractional farm (**needs a live g6 ARN**)
```bash
export G6_ARN='arn:aws:bedrock-agentcore:us-west-2:ACCT:runtime/acs_sandbox_gpu_shared-XXXX'
docker exec -e NR=11 -e EVAL_INTERVAL=10 -e EVAL_NS=2 \
    -e AGENTCORE_SANDBOX_RUNTIME_ARN="$G6_ARN" agentcore_p4 \
    bash -lc 'cd /work && bash code/demos/demo_gpu_shared/run.sh 2>&1 | tee /work/code/demos/_smoke_logs/demo_gpu_shared.log'
docker exec agentcore_p4 python3 /work/code/demos/demo_gpu_shared/metrics.py \
    /work/code/demos/_smoke_logs/demo_gpu_shared.log \
    --csv /work/code/demos/_smoke_logs/demo_gpu_shared.csv \
    --eval-csv /work/code/demos/_smoke_logs/demo_gpu_shared.eval.csv \
    --plot /work/code/demos/_smoke_logs/demo_gpu_shared.png
```
Client sets `B4_SHARED_RUNTIME_SESSION`; farm knobs (`B4_GPU_FRACTION`, `B4_FARM_GPUS`, …) live on
the g6, not here.

### `demo_gpu_single` — one whole g6 per session (**needs a live remote g6 ARN**; NOT in the on-box smoke)
```bash
export G6_ARN='arn:aws:bedrock-agentcore:us-west-2:ACCT:runtime/acs_sandbox_gpu-XXXX'
docker exec -e NR=11 -e EVAL_INTERVAL=10 -e EVAL_NS=2 \
    -e AGENTCORE_SANDBOX_RUNTIME_ARN="$G6_ARN" agentcore_p4 \
    bash -lc 'cd /work && bash code/demos/demo_gpu_single/run.sh 2>&1 | tee /work/code/demos/_smoke_logs/demo_gpu_single.log'
docker exec agentcore_p4 python3 /work/code/demos/demo_gpu_single/metrics.py \
    /work/code/demos/_smoke_logs/demo_gpu_single.log \
    --csv /work/code/demos/_smoke_logs/demo_gpu_single.csv \
    --eval-csv /work/code/demos/_smoke_logs/demo_gpu_single.eval.csv \
    --plot /work/code/demos/_smoke_logs/demo_gpu_single.png
```
`AGENTCORE_SANDBOX_RUNTIME_ARN` is mandatory (hard error if unset). Keep `NR × RB ≤ 100`. Run only
when a live g6 is provisioned; otherwise skip and note it in the results.

---

## Property demos — now REAL trainers (same 10-step / 1-eval / 1-ckpt spec)

These five were formerly offline sims / contract probes; they are now **real 10-step slime trainers**
(`NR=11` ⇒ 10 train + 1 eval + one final checkpoint) that layer their distinctive operational
property on top of the standard trainer as **env-gated glue that stays live during the run**. Each
demo's headline knob defaults **on**; setting it to its off value makes the run byte-identical to the
plain single-substrate trainer (`demo_cpu_mbpp` / `demo_gpu_single`). The old offline animation is
retained in each folder as a seconds-long probe (`heartbeat_demo.py`, `snapshot_demo.py`,
`harness_scaling_demo.py`, `gpu_torch_smoke.py`), invoked directly — **not** via `run.sh`.

The three **CPU** trainers carry the same baked CPU-sandbox ARNs as Group A, so they run in the
on-box smoke with **no extra env** (like the Group-A CPU demos). The two **GPU** trainers need a live
remote **g6** ARN (`AGENTCORE_SANDBOX_RUNTIME_ARN`) and belong with the GPU trainers (Phase 3);
`demo_gpu_customruntime` also has an on-box `LOCAL_GPU=1` path. Success criteria + metrics are the
standard ones below.

| Demo | Substrate | Property preserved (default-on knob) | Smoke command (data ships in `data/`) |
|---|---|---|---|
| `demo_cpu_heartbeat` | CPU sandbox (baked ARNs) | keep sandbox warm across the weight-sync barrier via `SessionKeeper`/`KeptSandbox` heartbeat (`HEARTBEAT_KEEP_WARM=1`) | `docker exec -e NR=11 agentcore_p4 bash -lc 'cd /work && bash code/demos/demo_cpu_heartbeat/run.sh 2>&1 \| tee /work/code/demos/_smoke_logs/demo_cpu_heartbeat.log'` |
| `demo_cpu_harness_scale` | CPU sandbox (baked ARNs) | warm in-process `run_episode` worker pool sized by `HARNESS_WORKERS` (`harness_pool_generate.generate`) | `... bash code/demos/demo_cpu_harness_scale/run.sh 2>&1 \| tee .../demo_cpu_harness_scale.log` |
| `demo_cpu_customruntime` | **your custom CPU Runtime** (`SANDBOX=agentcore-runtime`) | training runs unchanged on a custom Runtime, not the managed Code Interpreter | `... bash code/demos/demo_cpu_customruntime/run.sh 2>&1 \| tee .../demo_cpu_customruntime.log` |
| `demo_gpu_snapshot` | remote **g6** sandbox (needs ARN) | app-level snapshot/resume of the workspace at each episode boundary (`SNAPSHOT_EVERY=1`) | `docker exec -e NR=11 -e AGENTCORE_SANDBOX_RUNTIME_ARN="$G6_ARN" -e SNAPSHOT_EVERY=1 agentcore_p4 bash -lc 'cd /work && bash code/demos/demo_gpu_snapshot/run.sh 2>&1 \| tee .../demo_gpu_snapshot.log'` |
| `demo_gpu_customruntime` | remote **g6** sandbox (needs ARN) | KernelBench CUDA training only possible on a custom GPU Runtime (managed CI has no GPU) | `docker exec -e NR=11 -e AGENTCORE_SANDBOX_RUNTIME_ARN="$G6_ARN" agentcore_p4 bash -lc 'cd /work && bash code/demos/demo_gpu_customruntime/run.sh 2>&1 \| tee .../demo_gpu_customruntime.log'` |

> ⚠️ The two GPU trainers (`demo_gpu_snapshot`, `demo_gpu_customruntime`) are **offline-validated only**
> (import + `bash -n` + the offline probe); their end-to-end path on a live g6 is **UNVERIFIED on hardware**.

## Excluded
- **`convergence/`** — not a runnable demo; it holds per-demo wrappers that run each `train.sh` **to
  convergence** (the opposite of a smoke). Out of scope here.

---

## Run order (suggested)

1. **Phase 0 — offline probes (seconds, no AWS/GPU):** run each property demo's retained animation
   directly (`heartbeat_demo.py`, `snapshot_demo.py`, `harness_scaling_demo.py`, `gpu_torch_smoke.py`)
   for fast confidence the harness/glue imports and the mechanism is legible — **not** the trainers,
   which now run in the phases below.
2. **Phase 1 — CPU trainers (Group A + mixtask + the three CPU property demos):** run serially (each
   holds the 8 GPUs). Baked ARNs, no extra env. Includes `demo_cpu_heartbeat`,
   `demo_cpu_harness_scale`, `demo_cpu_customruntime`.
3. **Phase 2 — routed CPU trainers:** `demo_cpu_imagepool` (after deploying its 3 images),
   `demo_cpu_branching` (after a CLI image is ready + bring-up verified).
4. **Phase 3 — GPU trainers:** `demo_gpu_local` (on-box, two-step); `demo_gpu_shared` /
   `demo_gpu_single` / `demo_gpu_snapshot` / `demo_gpu_customruntime` only if a live g6 is provisioned
   (`demo_gpu_customruntime` also has an on-box `LOCAL_GPU=1` path).

Run trainers **one at a time** — each reserves all 8 local GPUs (4 train + 4 inference).

## Success criteria (check per log/CSV)

- **Completes 10 steps:** the log shows rollouts `1..10` and a final checkpoint write under
  `$SAVE` (and no earlier checkpoint — save-at-end held).
- **Exactly 1 eval:** one `eval/<name>` block near step 10.
- **~10 weight syncs:** one weight update/sync per rollout step in the async driver.
- **Sane learning signal:** `rollout/raw_reward` present and non-degenerate (not identically 0 —
  a stuck-at-0 family in `imagepool`/`mixtask` means a mis-mapped image ARN; in `branching` it can
  mean the CLI harness isn't producing valid submissions).
- **Latency split recorded:** `perf/sandbox_ms_*` vs `perf/sglang_gen_ms_*` columns populated
  (in-process harness demos); confirms the remote sandbox round-trips.
- **Artifacts saved:** `_smoke_logs/<demo>.log` + `.csv` + `.eval.csv` + `.png` exist for every run.
