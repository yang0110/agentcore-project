# Progress / Handoff — AgentCore + slime demos

_Last updated: 2026-09-25. Branch: `master`._

This file tracks the current work session so it can be picked up later. It has three
sections: **Done**, **In progress**, **Next steps (to run)**.

> **2026-09-25 — GPU farm demos merged + scale_sandboxes folded in (13 demos now: 9 CPU + 4 GPU).**
> `demo_gpu_shared_single`, `demo_gpu_shared_multi` and `demo_gpu_local` are now ONE demo,
> `demo_gpu_farm` (a `git mv` of `demo_gpu_shared_single`), selected with `FARM=fractional` (default;
> GPU 6, 0.5 + 0.5) or `FARM=whole` (GPUs 6,7, one GPU each — shared_multi and local had identical
> config). Host launcher modes are now `run_local_gpu_runtime.sh fractional|whole` (old names still
> aliased); convergence `13_gpu_farm_fractional.sh` / `14_gpu_farm_whole.sh`. The uncommitted
> `demo_cpu_scale_sandboxes` copy was folded into `demo_cpu_scale_lanes` (its only new files,
> `build_deepcoder.py` + `data/deepcoder_probe.csv`; DeepCoder checker not wired yet).
>
> **2026-09-23 — dataset demos merged (15 demos at the time; 13 since the 2026-09-25 `demo_gpu_farm` merge).** The ten per-dataset CPU folders
> (`demo_cpu_mbpp`, `demo_cpu_math`, `demo_cpu_gsm8k`, `demo_cpu_math500`, `demo_cpu_deepscaler`,
> `demo_cpu_humaneval`, `demo_cpu_gretelsql`, `demo_cpu_hotpotqa`, `demo_cpu_xlam`, `demo_cpu_mcp`)
> are now ONE demo, `demo_cpu_datasets` (a `git mv` of `demo_cpu_mbpp`), selected with
> `DATASET=<name>` (default `mbpp`; `FAMILY` still works as an alias). The set is 9 CPU + 6 GPU;
> see `code/demos/README.md`. Old folder names below are kept as history.
>
> **2026-09-23 — `demo_cpu_harness_scale` → `demo_cpu_scale_lanes`.** Reworked to
> `LANES=4` = 4 harness workers : 4 SGLang engines (TP1, rollout GPUs 4–7) : 4 persistent microVMs,
> strictly 1:1:1 (`LANE_SANDBOX=0` = old fresh session per episode); `[demo]` lane fields, `router.csv`
> `lane` column, `check_stickiness` 1:1:1 check; convergence `16_cpu_scale_lanes.sh`. **Smoke PASSED
> on AWS** (`_smoke_logs/lanes_tb_0923/scale_lanes.log`, 10 steps + eval, job succeeded): strict 1:1:1
> held (lanes 0–3 = 104/102/100/98 episodes, never crossed), raw_reward 0.19–0.50, eval/deepscaler
> 0.358. Only noise: resource_tracker KeyError at teardown + lane stop_session idle-reclaim fallback.
> Old names below (`harness_scale`) are history.
>
> **2026-09-23 — `demo_cpu_terminalbench`: first smoke PASSED on AWS.** Look-ahead pre-open
> (`tb_lookahead_source.py`, `TB_PREOPEN_BATCHES`=2 → 64 sessions ahead, `CONC`=⌈2W/E⌉). Deploy
> 218/229 Runtimes; oracle final **166/218 usable** (47 oracle-fail, 5 errored, 0 null-pass) →
> dataset 135 train + 31 eval. Quota: Active Session Workloads 5000 (L-3E5722B2) ≫ 64; 25 new/s is
> retried on throttle. Smoke (`_smoke_logs/lanes_tb_0923/terminalbench.log`, 382 episodes, ~70 min,
> job succeeded, 0 infra_error): `open_wait_ms` p95 **0** (boot fully hidden), open p50 0.66 s, 100 %
> warm hit; BUT `preopen_idle_ms` p50 7.3 min vs episode p50 21 s → ~95 % of microVM-s idle.
> Reward sparse: 4 solves/382 (train 0.088 / 0.026 on 2 steps, else 0), eval 0.0, 42 % pass ≥1 test.
> Next: try `TB_PREOPEN_BATCHES`=1 (cost), easy-first curriculum / partial-credit reward for signal,
> run `probe_coldstart.py`. Deck `docs/slides/terminalbench_deck.md` has the results.

---

## Done

### 1. New standalone demo: `demo_gpu_local` (whole-GPU Ray farm)
- Turned the old `LOCAL_GPU=1` local-6,7 fallback into its own demo (whole-GPU farm:
  `B4_FARM_GPUS=2`, `B4_GPU_FRACTION=1.0` → 2 whole-GPU sub-sessions, one GPU per episode).
- Removed the `LOCAL_GPU` fallback from `demo_gpu_single` / `demo_gpu_shared` (they are now
  **remote-only** — require a real g6 sandbox Runtime ARN).
- `code/deploy/run_local_gpu_runtime.sh` simplified to `local | stop` only; serves just the
  standalone `demo_gpu_local` farm. `bash -n` verified.

### 2. Report + deck: "one session = one whole instance" clarification
- Added the AgentCore constraint (1 AgentCore session = 1 whole instance; GPUs live on
  capacity-provider instances; you get the instance, not a GPU → can't natively point two
  runtimeSessionIds at "same 2-GPU box, one GPU each"). The right pattern is one shared
  session + an in-instance Ray farm with `FRACTION=1.0`/`FARM_GPUS=N`.
- Edited: `docs/report/agentcore_slime_report.md`, `report-based-deck/deck.md`, `.tex`.

### 3. Custom-runtime demo pair (managed vs custom Runtime contrast)
- `demo_cpu_customruntime/` — `compare_backends.py` runs the SAME MockPolicy episode against
  each backend, asserts identical reward, prints an operational-diff table. Punchline: same
  task-agnostic contract (`write_file`/`exec`/`read_file`) → same reward; only ops differ.
  Offline smoke passes (`local → solved=True reward=1.0`).
- `demo_gpu_customruntime/` — `gpu_torch_smoke.py` write→exec→read of a torch-CUDA probe:
  managed Code Interpreter has no GPU (`cuda_available=False`); custom Runtime on g6 does.
  Offline smoke round-trips end-to-end.
- Both: `py_compile` + `bash -n` clean; vendored `agentcore_slime_demo/` byte-identical to
  source demos. **Live AWS is documented-only** (drivers default `SANDBOX=local`).

### 4. Checkpoint cleanup (disk) — DONE 2026-09-18
- Deleted output checkpoints (~370G freed, 58% → 39% used):
  - `code/demos/demo_cpu_mbpp/ckpt/` (that folder is now `demo_cpu_datasets/`; its `ckpt/` holds only `.gitkeep`): Qwen3-4B_{ctf,gretelsql,rag,sql,tool_use}_async
  - `code/p4_slime_train/ckpt/`: Qwen3-4B_mbpp_grpo, Qwen3-4B_mbpp_learn
- **PRESERVED (do NOT delete):** `code/p4_slime_train/ckpt/Qwen3-4B_torch_dist` (7.5G) — the
  base Megatron checkpoint every run trains from (`CKPT_BASE`). Deleting it breaks all reruns.
- Note: ckpt files are owned by `root` (written by the training container) → delete with
  `sudo rm -rf` from the host, or from inside `agentcore_p4`.

### 5. Convergence run directory (`code/demos/convergence/`) — DONE
- 16 files: `README.md` + `commands.md` + 14 numbered wrappers (01–11 CPU, 12–14 GPU), one per
  trainable demo. Each wrapper is a thin, fully-overridable (`${VAR:=default}`) shell around the
  demo's own `run.sh` with a rich header (task/dataset, substrate, bottleneck hypothesis,
  convergence metric, prereqs, runtime/ckpt, teardown) and a `tee` to a timestamped log.
- Convergence defaults: CPU `NR=201` (200 steps) / GPU `NR=101` (100 steps); `RB=8`, `NS=8`,
  `EVAL_INTERVAL=20`, `EVAL_NS=4`, `SAVE_INTERVAL=100` (with loud ~56 GB/ckpt disk caveat).
  `12`/`13` remote-only with preflight; `13` defaults `B4_GPU_FRACTION=0.5`; `14` preflights the
  local farm `/ping`. Excluded (offline/infra): quiesce, snapshot, harness_scale, customruntime.
- All 14 scripts `bash -n`-clean and executable; no edits outside `code/demos/convergence/`.
- **Note:** this dir is for *full convergence* (100–200 steps). The `NR=11` "10-steps + eval +
  save" sweep below is a separate, shorter smoke/validation pass.

---

## Next steps (to run) — the "10 steps + 1 eval + save" re-run sweep

**Goal:** rerun every trainable demo at `NR=11` (= 10 train steps + 1 eval step + final save),
serially (each trainer claims ~all 8 GPUs, so no parallelism). `NR=11` is already the demos'
default, so plain `run.sh` does it.

**Where to run:** INSIDE the `agentcore_p4` container (`docker exec`), repo at `/work/code`.
Host launch dies on the `/root/slime` model-source permission error.

### Trainable demos
CPU (`SANDBOX=agentcore-runtime`; need AWS creds + CPU sandbox ARN, defaulted in each `run.sh`):
- demo_cpu_datasets, once per `DATASET=` gsm8k, math, math500, deepscaler, humaneval,
  gretelsql, hotpotqa, mcp, xlam, mbpp (formerly one folder each)
- demo_cpu_mixtask

GPU:
- demo_gpu_local  → start local farm first: `bash code/deploy/run_local_gpu_runtime.sh local`
  (host), then `bash code/demos/demo_gpu_local/run.sh` (inside container).
- demo_gpu_single → remote-only, needs a real g6 sandbox Runtime ARN.
- demo_gpu_shared → remote-only; ray-farm; known non-MIG wedge under load → raise
  `B4_GPU_FRACTION` toward 0.5 if it stalls.

### NOT part of the sweep (no standard training loop)
- demo_cpu_customruntime, demo_cpu_scale_lanes, demo_cpu_heartbeat,
  demo_gpu_customruntime, demo_gpu_snapshot

### Suggested driver
`code/demos/run_smoke_sweep.sh` already does a serial sweep with per-run ckpt cleanup (it used
`NR=3`). Reuse it with `NR=11` overridden, or run each `run.sh` one-by-one. Each CPU run ≈ short;
budget a few hours for the full serial sweep. Confirm AWS creds are live before starting.
