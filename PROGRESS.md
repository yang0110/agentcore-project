# Progress / Handoff — AgentCore + slime demos

_Last updated: 2026-09-26. Branch: `master`._

This file tracks the current work session so it can be picked up later. It has three
sections: **Done**, **In progress**, **Next steps (to run)**.

> **2026-09-26 — GPU farm folders merged; instance pool demoted → 5 main demos + 1 side note + 1 experimental.**
> - **Main demos (storyline order):** `demo_cpu_tasks`, `demo_cpu_terminalbench` (+ `warmstart/` probe),
>   `demo_cpu_scale_lanes`, `demo_gpu_single` (KernelBench, one session per instance), `demo_gpu_farm`.
> - **`demo_gpu_farm`** (merged from `demo_gpu_farm_fractional` + `demo_gpu_farm_whole`, identical code): ONE GPU
>   instance serving MANY sessions, `FARM=fractional` (default; GPU 6, 0.5 + 0.5, like a single-GPU `g6.xlarge`) or
>   `FARM=whole` (GPUs 6,7, one whole GPU per session, like a multi-GPU `g6.12xlarge`). Labelled **trusted-only:
>   shared kernel + shared GPU** (no MIG on L4/L40S, cooperative memory-fraction cap, no Landlock port, one shared
>   uid on AgentCore). Results kept as rows: `FARM=fractional` eval 0.65 (2026-09-25) / 0.775 (09-21);
>   `FARM=whole` 0.65 / 0.70 (09-21), old logs now in `demo_gpu_farm/logs/whole/`.
> - **Side note:** `demo_cpu_instance_pool` (warm m5 pool shared through per-episode sub_sessions). The Instances
>   are AgentCore, but the pooling is our own code and a shared instance can't give full isolation (Landlock-grade,
>   one kernel). Its "Isolation at a glance" summary is condensed into the report as a side-note subsection.
> - **Experimental:** `demo_cpu_branching` (unchanged).
> - **IN PROGRESS (not done):** `demo_cpu_tasks`' default is switching from the Code Interpreter to our own Runtime
>   (generic stdlib image, VPC) with the harness running inside the sandbox session; the Code Interpreter stays as a
>   zero-setup opt-in. Until that lands, the default and the measured results are the Code Interpreter path.
> - Docs updated: report (.md/.tex/.pdf), report-based deck (.md/.tex/.pdf), `docs/demos-program.md`,
>   `docs/harness-sandbox-decoupling.md`, `docs/sandbox-fresh-env-on-persistent-instance.md`.
>
> **2026-09-25 — storyline restructure → 7 demos + 1 experimental** (superseded 2026-09-26, above). In storyline order:
> 1. `demo_cpu_tasks`: every literature task on the **DEFAULT managed Code Interpreter** (`SANDBOX=agentcore`,
>    no image / Runtime; steps 2/3/4/6 are no-ops). The CI has pandas/scipy/sklearn/torch-cpu/gcc/node (no git,
>    no internet); all 13 families pass on it, and `DATASET=gsm8k` scored eval 0.90 on it. `VARIANT=imagepool` →
>    **`DATASET=libs`** on the CI (the 3 custom images are gone); `VARIANT=bigimage` moved out (item 3).
>    `HEARTBEAT=1` is a no-op on the CI (sessions end at `AGENTCORE_CI_SESSION_TIMEOUT`, default 3600 s);
>    it only helps `SANDBOX=agentcore-runtime`.
> 2. `demo_cpu_terminalbench`: one image + Runtime per task, look-ahead pre-open (`TB_PREOPEN_BATCHES=2`),
>    always-on heartbeat. It now carries the **`warmstart/` probe** (`bash warmstart/run_warmstart.sh`, results
>    in its README): pre-open cuts wait p50 774 → 5 ms (own Runtime) / 1715 → 0 ms (CI), makespan −20 % / −15 %,
>    only the first batch waits; heartbeat on keeps 3/3 own-Runtime sessions past the probe's shortened 60 s idle timeout (training uses the 15-min default; resume
>    104 ms), off → all silently reclaimed; the CI expires at its session timeout regardless. The short-lived
>    `demo_cpu_warmstart` folder was deleted with its Runtime/ECR.
> 3. `demo_cpu_instance_pool` (formerly `demo_cpu_tasks VARIANT=bigimage` / `demo_cpu_bigimage`): ~4 GB image on
>    one m5 capacity-provider Instance, warm pool `SANDBOX_POOL=N` (default 2) of shared sessions, each episode
>    a sub_session whose every exec runs in its own **Landlock jail** (own dir only, no TCP, signals/ptrace scoped,
>    256-proc / 4 GiB watchdog, tree killed when the exec ends, path-escape refusal, `AWS_*` stripped). Per-slot
>    uids (`:uidiso`) broke live: AgentCore drops every capability, so `chown` → EPERM on every open. Still
>    shared: kernel, netns (UDP/DNS), CPU/RAM, `ps`. `isolation_probe.py` **22/22 live on the m5** (Runtime v3,
>    `:landlock-20260925` = `:latest`); 13 families reward 1.0 live; RL smoke passed (eval 0.7875, pre-jail image).
> 4. `demo_cpu_scale_lanes`: now **decoupled tiers**, 4 SGLang engines : `HARNESS_WORKERS`=256 in-process workers :
>    `SANDBOX_SESSIONS`=256 pooled persistent microVMs (was 4 × 4 × 4 pinned lanes; `LANES`/`LANE_SANDBOX` removed).
>    Per episode: free worker pulls → borrows any free session (fresh `ep-<hex>`, `rm -rf` on return) → least-loaded
>    engine. Sessions open once at `SANDBOX_OPEN_RATE`=20/s; boto3 pool + executor sized to 256; `RB`=32 so 256
>    episodes/step. Live pool check (W=M=64, fake inference, real Runtimes): 1280 ops ok, engines 48/48/47/49, tiers
>    mix. RL smoke pending (needs 8 GPUs, or pin + `ACTOR_GPUS=2`).
> 5. `demo_gpu_single`.
> 6. (history; merged back into `demo_gpu_farm` 2026-09-26) `demo_gpu_farm_fractional` (GPU 6, 0.5 + 0.5; smoke eval/kernelbench 0.65) and
> 7. `demo_gpu_farm_whole` (GPUs 6,7, one GPU per session): `demo_gpu_farm FARM=…` split into two folders, each
>    hard-setting its mode; launcher `code/deploy/run_local_gpu_runtime.sh fractional|whole|stop`.
> - Experimental: `demo_cpu_branching`: the reward-0 cause was a missing `--sglang-tool-call-parser qwen25` (+ CLI `exit=1` on the 1024 per-turn cap); both fixed in `run.sh` 2026-09-25, RL re-run pending.
> - Convergence: 01–11 now say Code Interpreter; 13/14 point at the farm folders (no `FARM`; since 2026-09-26 back to `demo_gpu_farm FARM=…`); 15 pins
>   `SANDBOX=agentcore-runtime`; `17_cpu_tasks_imagepool.sh` → **`17_cpu_tasks_libs.sh`**; new
>   **`19_cpu_instance_pool.sh`** (`SANDBOX_POOL=2`, ARN from `deploy/runtime.env`).
> - Smoke drivers: trainers `demo_cpu_tasks:libs`, `demo_cpu_tasks:deepscaler+HEARTBEAT=1+SANDBOX=agentcore-runtime`;
>   `run_smoke_plan.sh` PHASE=2 = `demo_cpu_instance_pool` (was `demo_cpu_tasks+VARIANT=imagepool`);
>   `run_gpu_smoke.sh` / `run_b_queue.sh` use the two farm folders (since 2026-09-26 `demo_gpu_farm FARM=…`); offline probes `demo_cpu_terminalbench:warmstart`,
>   `demo_cpu_instance_pool:isolation`.
> - Decided (user, 2026-09-25: keep): the env-only `SLIME_CUDA_VISIBLE` / `SLIME_NUM_GPUS` pin runs the trainer on
>   a GPU subset of the shared box; unset = the unchanged hardcoded 8-GPU placement. Read by the run.sh of
>   tasks, instance_pool, branching, scale_lanes and the 3 GPU demos (terminalbench doesn't read it yet).
>
> **2026-09-25 — heartbeat / customruntime / imagepool folded into `demo_cpu_tasks` (6 demos at the time).**
> The set is **5 main demos** (3 CPU: `demo_cpu_tasks`, `demo_cpu_terminalbench`, `demo_cpu_scale_lanes`;
> 2 GPU: `demo_gpu_single`, `demo_gpu_farm`) **plus `demo_cpu_branching`, EXPERIMENTAL/blocked**. Merges:
> - `demo_cpu_heartbeat` → the opt-in **`HEARTBEAT=1`** knob of every demo (`session_keeper.maybe_keep_warm`
>   wraps each episode's sandbox in `generate.py`; scale_lanes also registers its lane sessions;
>   `HEARTBEAT_IDLE`=600, `HEARTBEAT_PING_FRACTION`=0.5; `HEARTBEAT_KEEP_WARM` still aliased). Default off ⇒
>   unchanged. Terminal-Bench keeps its own always-on keeper. Offline probe → `demo_cpu_tasks/heartbeat_demo.py`.
> - `demo_cpu_customruntime` → `demo_cpu_tasks/compare_backends.py` (same episode, same reward, custom vs managed Runtime).
> - `demo_cpu_imagepool` → **`demo_cpu_tasks VARIANT=imagepool`** (`DATASET=imagepool`, `eval/imagepool`,
>   `data/imagepool_*.jsonl` via `build_mixed_datasets.py --name imagepool`, `deploy/deploy_imagepool.py`
>   → `deploy/runtime.imagepool.env` with the image→ARN map).
> - Convergence: `15_cpu_heartbeat.sh` → `15_cpu_tasks_heartbeat.sh`; `17_cpu_customruntime.sh` (= 05 on a
>   custom Runtime) replaced by `17_cpu_tasks_imagepool.sh`.
> - Smoke drivers: entries take a `+KEY=VAL` suffix (`demo_cpu_tasks:deepscaler+HEARTBEAT=1`,
>   `demo_cpu_tasks+VARIANT=imagepool`); offline probes `demo_cpu_tasks:heartbeat` / `demo_cpu_tasks:compare_backends`.
> - Old logs/ckpts moved to `demo_cpu_tasks/logs/{heartbeat,customruntime,imagepool}/` and `demo_cpu_tasks/ckpt/`.
>
> **2026-09-25 — bigimage + GPU single/snapshot/customruntime merged; step scripts everywhere (9 demos at the time).**
> The set was **8 main demos** (6 CPU: `demo_cpu_tasks`, `demo_cpu_imagepool`, `demo_cpu_terminalbench`,
> `demo_cpu_customruntime`, `demo_cpu_heartbeat`, `demo_cpu_scale_lanes`; 2 GPU: `demo_gpu_single`,
> `demo_gpu_farm`) **plus `demo_cpu_branching`, marked EXPERIMENTAL/blocked** (the in-sandbox CLI exits 1
> before any model call → no trajectories). Merges:
> - `demo_cpu_bigimage` → **`demo_cpu_tasks VARIANT=bigimage`** (side note on the tasks track; `probe_bigimage.py`,
>   `netcheck.py`, `deploy/deploy_bigimage.py`, `Dockerfile.sandbox.bigcpu` moved in; old logs in `demo_cpu_tasks/logs/bigimage/`).
> - `demo_gpu_snapshot` → **`demo_gpu_single SNAPSHOT_EVERY=1`** (default 0 = off; old log in `demo_gpu_single/logs/snapshot/`).
> - `demo_gpu_customruntime` → **`demo_gpu_single`** (custom-Runtime angle, `gpu_torch_smoke.py`, `LOCAL_GPU=1` on-box farm with sync save).
> - Convergence: `18_gpu_snapshot.sh` → `18_gpu_single_snapshot.sh`; `19_gpu_customruntime.sh` deleted (= `LOCAL_GPU=1 12_gpu_single.sh`).
> - Smoke drivers: `run_gpu_smoke.sh` entries `demo_gpu_single` / `demo_gpu_single:snapshot`; offline probe → `demo_gpu_single` `snapshot_demo.py`.
>
> Every demo now ships the same six step scripts `1_prepare_data.sh` … `6_cleanup.sh` (see
> `code/demos/README.md`). Report, report-based deck, `slide_figures/` tables and the supporting docs
> were updated to the 9-demo set; smoke results are kept and attributed to the merged demo/variant.
>
> **2026-09-25 — single-dataset + mixed-task demos merged (12 demos at the time: 8 CPU + 4 GPU).**
> `demo_cpu_datasets` and `demo_cpu_mixtask` are now ONE demo, **`demo_cpu_tasks`** (a `git mv` of
> `demo_cpu_datasets`): a single dataset is the one-element mix. `DATASET=mbpp` etc. works as before;
> `DATASET=a,b,c` interleaves those datasets round-robin (`eval/mix`); `DATASET=mixed` is the old
> mixtask data (math+sql+ctf+datasci, `eval/mixed`, datasci routed via
> `AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE`). The vendored lib took mixtask's superset
> (`generate/sandbox/task/task_families.py`); every data row dispatches (3191/3191) and
> `tests/test_smoke.py` passes. Convergence `11_cpu_mixtask.sh` → `11_cpu_mixed.sh`; smoke entry
> `demo_cpu_mixtask` → `demo_cpu_tasks:mixed`.
>
> **2026-09-25 — GPU farm demos merged + scale_sandboxes folded in (13 demos at the time: 9 CPU + 4 GPU).**
> `demo_gpu_shared_single`, `demo_gpu_shared_multi` and `demo_gpu_local` are now ONE demo,
> `demo_gpu_farm` (a `git mv` of `demo_gpu_shared_single`), selected with `FARM=fractional` (default;
> GPU 6, 0.5 + 0.5) or `FARM=whole` (GPUs 6,7, one GPU each — shared_multi and local had identical
> config). Host launcher modes are now `run_local_gpu_runtime.sh fractional|whole` (old names still
> aliased); convergence `13_gpu_farm_fractional.sh` / `14_gpu_farm_whole.sh`. The uncommitted
> `demo_cpu_scale_sandboxes` copy was folded into `demo_cpu_scale_lanes` (its only new files,
> `build_deepcoder.py` + `data/deepcoder_probe.csv`; DeepCoder checker not wired yet).
>
> **2026-09-23 — dataset demos merged (15 demos at the time; 12 since the 2026-09-25 merges).** The ten per-dataset CPU folders
> (`demo_cpu_mbpp`, `demo_cpu_math`, `demo_cpu_gsm8k`, `demo_cpu_math500`, `demo_cpu_deepscaler`,
> `demo_cpu_humaneval`, `demo_cpu_gretelsql`, `demo_cpu_hotpotqa`, `demo_cpu_xlam`, `demo_cpu_mcp`)
> are now ONE demo, `demo_cpu_datasets` (a `git mv` of `demo_cpu_mbpp`; since 2026-09-25 `demo_cpu_tasks`), selected with
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
  - `code/demos/demo_cpu_mbpp/ckpt/` (that folder is now `demo_cpu_tasks/`; its `ckpt/` holds only `.gitkeep`): Qwen3-4B_{ctf,gretelsql,rag,sql,tool_use}_async
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
- demo_cpu_tasks, once per `DATASET=` gsm8k, math, math500, deepscaler, humaneval,
  gretelsql, hotpotqa, mcp, xlam, mbpp, mixed (formerly one folder each; `mixed` was demo_cpu_mixtask)

GPU:
- demo_gpu_local  → start local farm first: `bash code/deploy/run_local_gpu_runtime.sh local`
  (host), then `bash code/demos/demo_gpu_local/run.sh` (inside container).
- demo_gpu_single → remote-only, needs a real g6 sandbox Runtime ARN.
- demo_gpu_shared → remote-only; ray-farm; known non-MIG wedge under load → raise
  `B4_GPU_FRACTION` toward 0.5 if it stalls.

### NOT part of the sweep (no standard training loop)
- demo_cpu_customruntime, demo_cpu_scale_lanes, demo_cpu_heartbeat,
  demo_gpu_customruntime, demo_gpu_snapshot
  (2026-09-25: all but scale_lanes are now knobs/probes of demo_cpu_tasks / demo_gpu_single —
  `HEARTBEAT=1`, `compare_backends.py`, `SNAPSHOT_EVERY=1`, `LOCAL_GPU=1`)

### Suggested driver
`code/demos/run_smoke_sweep.sh` already does a serial sweep with per-run ckpt cleanup (it used
`NR=3`). Reuse it with `NR=11` overridden, or run each `run.sh` one-by-one. Each CPU run ≈ short;
budget a few hours for the full serial sweep. Confirm AWS creds are live before starting.
