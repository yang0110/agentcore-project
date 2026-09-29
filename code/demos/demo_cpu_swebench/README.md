# demo_cpu_swebench — RL on real GitHub issues (SWE-bench Verified), one arm64 image per dataset row, one fresh microVM per episode

**Sandbox tier 3: own Runtime, custom image — on the managed arm64 microVM fleet (2 GB image cap).**
Every SWE-bench instance pins its own repo commit + conda env, so every dataset row gets its own image
and Runtime. The images are Epoch AI's **arm64** rebuilds of the SWE-bench images, which fit under the
microVM fleet's 2 GB cap, so each Runtime is a plain microVM Runtime (no capacity provider). See the tier
table in [`../README.md`](../README.md#which-sandbox-does-a-task-need-three-tiers).

Two twins run the same dataset, harness, grader and RL recipe on other substrates, with the official
amd64 images: [`demo_cpu_swebench_eks`](../demo_cpu_swebench_eks/README.md) (one gVisor pod per episode)
and [`demo_cpu_swebench_ecs`](../demo_cpu_swebench_ecs/README.md) (one ECS task per episode on EC2).
Until 2026-09-28 this demo ran the amd64 images on m5 capacity-provider Instances (one m5 per episode);
that pool's state is kept in `swebench/m5_legacy/` for teardown (see step 6).

Split out of `demo_cpu_terminalbench` (where it was `BENCH=swebench`) on 2026-09-28; that demo is now
`demo_cpu_tasks DATASET=terminalbench`. It reuses that
demo's image-pool machinery unchanged (vendored): `TerminalBenchPool` (fresh session per episode,
pre-open, heartbeat, strict routing by `task.image` = instance id) and `terminalbench_pool_generate`.
What is SWE-bench-specific: the per-instance arm64 images and the hidden-test grader.

The policy is Qwen3-4B trained by slime (async/decoupled GRPO). The agent is the mini_claude
tool-calling loop, and it runs **inside the episode's session**: `InSessionHarness`
(`in_session_harness.py`) writes `session_runner.py` into the session, and the loop dials the
trainer's per-engine `AnthropicAdapter` back over the VPC (SG→SG rule on `ADAPTER_PORT_BASE`
18100-18163). There is no in-trainer harness and no fallback: a session that can't host the loop
(no `python3`, or the adapter unreachable) raises `HarnessPlacementError`. The loop needs
`--sglang-tool-call-parser qwen25 --sglang-reasoning-parser qwen3` (baked into `run.sh`). The
reward is the instance's own hidden FAIL_TO_PASS + PASS_TO_PASS tests (RLVR — no LLM judge).
**REMOTE-ONLY** (`SANDBOX=local` is rejected).

**Harness tier (the default in every demo).** The pool subclasses the shared `HarnessPool`
(`harness_pool.py`): `HARNESS_WORKERS` = W dispatch workers (default RB×NS = 32) pull episodes off one
queue, each episode takes its fresh microVM session first and then the least-loaded SGLang engine, sticky for
all its turns. No worker owns an engine. The harness scales with the sandbox (one loop per session),
not inside the trainer.

---

## Use case covered

RL on real GitHub issues: the agent edits a real repo at `/testbed` (django, sympy, sphinx, pytest,
astropy, requests, xarray, pylint, flask) and is rewarded when the instance's hidden tests pass.
The current subset is 20 instances, one Runtime each (16 train + 4 eval).

## Why this demo

- It is **the** standard agentic-coding benchmark.
- It is the extreme case of **one image per dataset row**: the dataset cannot be served by one
  generic image, because each row is a different repo, commit and dependency set.
- It shows that SWE-bench fits the **managed microVM fleet** as is: no capacity provider, no instances
  to pay for while idle, per-episode microVM isolation, and a ~25-30 s cold start instead of ~100 s for
  an m5. The two constraints are arm64 and a 2 GB image, and both are met by Epoch AI's arm64 rebuilds
  (`ghcr.io/epoch-research/swe-bench.eval.arm64.<id>`): 420 of the 500 Verified rows have one, all at
  most 1.8 GB compressed (`swebench/verified_image_sizes.csv`, `survey_image_sizes.py`). The default
  subset's images are 0.7-1 GB.

## Design considerations

- *The 2 GB cap is on the compressed image.* `probe_image_cap.py` asked AWS directly
  (`swebench/capprobe*.csv`): a 3 GB image of zeros (64 MB compressed) was accepted, 2041.7 MiB
  compressed was accepted and 2061.7 MiB rejected (`ServiceQuotaExceededException: maxImageSizeMb`).
  So the cap is 2048 MiB of `imageSizeInBytes` and on-disk size does not count. Selection keeps base
  images at most `SWE_MAX_IMAGE_MB` (1900), leaving room for the ~60 MB wrapper; deploy marks any image
  over 2048 MiB `too_large` and never registers it.
- *One Runtime per row.* `deploy/deploy_swebench.py` wraps each prebuilt arm64 image with a stripped
  arm64 standalone CPython at `/opt/acs/py` + the AgentCore HTTP server (cross-built with buildx +
  qemu), pushes it to ECR (`acs-swe-arm64-*`) and registers a VPC microVM Runtime (`acs_swearm_*`,
  `RUNTIME_EGRESS=none` by default, so no internet). The testbed env
  is not touched, so grading sees exactly the environment SWE-bench validated. Build/deploy cost grows
  linearly with the dataset (20 rows = 20 images, 20 repos, 20 Runtimes; full Verified would be 500).
- *Hidden tests.* The `test_patch` never enters the image or the dataset row; it lives only in the
  trainer-side `swebench/manifest.json`. After the agent's turns, `swebench.grade_swebench`:
  1. resets the files the patch touches to `base_commit`, discarding agent edits (added files are removed);
  2. `git apply`s the patch;
  3. runs the official per-repo/version `test_cmd` in conda env `testbed`;
  4. parses the log with vendored SWE-bench parsers.
- *Grading rule.* Solved iff every F2P and P2P test is PASSED/XFAIL, as in the official grader.
- *Deviation: no reinstall.* The official `install` step is not re-run, because repos are editable
  installs and the instance may lack PyPI egress. The oracle check catches any instance this breaks.
- *Oracle check.* `swebench_oracle_check.py` runs each instance twice on its own Runtime: the gold patch
  must solve it and the empty patch must not (`swebench/oracle.csv`, `usable=1`). Only usable instances
  go into `data/`. The m5 pool had checked 1 of 20; the arm64 images are checked afresh by step 4 (the local
  qemu check above works without AWS).
- *Selection.* `fetch_swebench.py` picks "<15 min fix" instances with 1-3 F2P, 1-150 P2P, at most
  2 test files and an arm64 image of at most 1.9 GB, from repos the vendored parsers support (quota:
  django 7, sympy 4, sphinx 3, pytest 2, one each for astropy, requests, pylint, flask = 20).
  matplotlib, sklearn and seaborn are excluded (slow suites); xarray has no arm64 image.
- *Why 20.* 20 is a smoke-sized default, not a limit: each row costs one image build, one Runtime and
  one oracle check. Raise it with `SWE_QUOTA` (e.g. `django/django:60,sympy/sympy:30,...`) or pick rows
  with `SWE_INSTANCES`; any of the 420 arm64 rows the parsers support can be added.
- *arm64 vs the official images.* The Epoch AI rebuilds are the same recipe on another architecture, so a
  test can in principle behave differently. The oracle check (gold solves, empty patch does not) runs on
  the arm64 image itself, so such an instance is caught and dropped.
- *Cold start.* Opening a session allocates a microVM and pulls the image: ~25-30 s for a 1-2 GB image
  (`capprobe_edge.csv`); a warm open is under 1 s. `TB_PREOPEN_BATCHES` hides it. An idle session costs
  nothing beyond its microVM.
- *Server shim.* The images run `deploy/acs_swe_http_server.py`, which accepts both chunked and
  Content-Length `/invocations` bodies (the m5 data plane sent chunked ones; the EKS / ECS twins reuse
  the same server).
- *Sparse reward.* A 4B policy solving real repo bugs is hard; expect a reward at least as sparse as
  Terminal-Bench's (4/382 on its first smoke). A curriculum or easier instances may be needed for signal.

## How an episode flows

```
slime → terminalbench_pool_generate.generate → TerminalBenchPool.submit
  harness worker pulls episode → route by task.image (= instance id) → that instance's Runtime ARN
  (at submit) open a BRAND-NEW session on that ARN while the episode waits   ← its own microVM
    SessionKeeper heartbeats it until it is stopped
      InSessionHarness writes session_runner.py into the session and polls it
        in-session loop in /testbed: LLM via the least-loaded engine's adapter (dial-back, policy under training), bash/write_file/read_file tools
    grade: swebench.grade_swebench (reset test files → apply hidden test_patch → test_cmd → parse F2P/P2P)
  finish_session → ONE loss-masked Sample (the longest); adapter.drop_session; StopRuntimeSession
```

## Step by step (numbered scripts; run from this dir on the HOST — container steps `docker exec` themselves)

```bash
bash 1_prepare_data.sh     # 1. container: fetch_swebench.py (select subset -> swebench/manifest.json)
                           #    then prepare_data.sh -> data/swebench_{train,eval}.jsonl
bash 2_build_image.sh      # 2. SWE_NO_REGISTER=1 deploy_swebench.py: wrap + push one arm64 image per instance (ECR acs-swe-arm64-*)
bash 3_push_ecr.sh         # 3. no-op: prints swebench/images.csv (step 2 already pushed)
bash 4_deploy_runtime.sh   # 4. deploy_swebench.py: one microVM Runtime per instance (<= 2 GB) -> swebench/arn_by_image.json,
                           #    then swebench_oracle_check.py -> swebench/oracle.csv, then fetch_swebench.py --data-only
bash 5_run.sh              # 5. container: 10 train steps + 1 eval through the pool; logs/<tag>.{log,csv,eval.csv,png}
bash 6_cleanup.sh          # 6. delete every Runtime in swebench/arn_by_image.json
```

- **Validate on 2 instances first** (real AWS $): `SWE_LIMIT=2 bash 2_build_image.sh && SWE_LIMIT=2 bash 4_deploy_runtime.sh`.
- `run.sh` loads `swebench/arn_by_image.json` itself, so `deploy/runtime.env` stays empty.
- The retired m5 pool (20 `acs_swe_*` Runtimes + capacity provider `acs_swe_cp`) is torn down with
  `SWE_POOL=swebench/m5_legacy/arn_by_image.json CP_NAME=acs_swe_cp DELETE_CP=1 bash 6_cleanup.sh`.
- Local check without AWS: `SWE_NO_PUSH=1 SWE_TASKS=<id> python3 deploy/deploy_swebench.py` builds the
  arm64 image, `SWE_TASKS=<id> python3 swebench_oracle_check.py --local` runs gold/empty under qemu.
- Train to convergence (inside the container): `train.sh` (`STEPS`, `EVAL_EVERY`, `SANDBOX_CONC`).

## Knobs (env vars; defaults in `run.sh`)

| Knob | Default | Meaning |
|---|---|---|
| `SANDBOX` | `agentcore-runtime` | fresh AgentCore session per episode. **`local` is rejected.** |
| `TB_PREOPEN_BATCHES` | `2` | upcoming batches kept booted ahead (`× RB × NS` = 64 microVM sessions). `0` = off |
| `HEARTBEAT` / `WARM_IDLE_TIMEOUT` / `WARM_PING_FRACTION` | `1` / `900` / `0.5` | heartbeat any session idle past `fraction × timeout` (on by default; `HEARTBEAT=0` = A/B baseline) |
| `HARNESS_WORKERS` | `RB × NS` (32) | harness tier: dispatch workers that pull episodes, start and poll the in-session loop; each episode picks the least-loaded engine |
| `CONC` | `⌈2W/E⌉` | episode-admission concurrency `S = CONC × num_engines` (W running + W queued) |
| `EPISODE_MAX_TURNS` / `EPISODE_MAX_TOKENS` | `12` / `1024` | agentic turns / tokens per episode |
| `TB_EXEC_TIMEOUT` | `180` | per bash-tool command timeout (s) |
| `MAX_TOKENS_PER_GPU` | `24576` | ≥ one full 12-turn trajectory |
| `RB` / `NS` / `NR` | 4 / 8 / 11 | GRPO prompts-per-step / group size / num-rollout (real steps `NR-1`) |
| `EVAL_INTERVAL` / `EVAL_NS` | 10 / 2 | held-out eval cadence / samples |

- Deploy: `SWE_TASKS`, `SWE_LIMIT`, `SWE_BUILD_CONC`, `SWE_REBUILD`, `SWE_NO_REGISTER`,
  `SWE_NO_PUSH` (local build; then `swebench_oracle_check.py --local`), `RUNTIME_EGRESS` (`none` | `nat`).
- Data: `SWE_QUOTA`, `SWE_INSTANCES`, `SWE_MAX_IMAGE_MB` (1900), `SWE_MAX_TURNS` (12), `SWE_TEST_TIMEOUT` (600 s, which keeps
  the grade exec under `INVOKE_READ_TIMEOUT` 900).

## What to watch

- `rollout/raw_reward` and `eval/swebench` — the learning signal / held-out solve rate.
- `[demo] fresh_session=1 preopened=… open_wait_ms=… preopen_idle_ms=…` — one line per episode;
  `open_wait_ms` near 0 means the microVM cold start was hidden by the pre-open. The same line carries
  `worker`, `qdepth` (> 0 = harness-bound), `engine` and `engine_inflight` (climbing = inference-bound).
- `perf/sandbox_ms_*` vs `perf/sglang_gen_ms_*` — the sandbox-vs-decode latency split.

## Folder layout (self-contained)

- `agentcore_slime_demo/` — vendored library (its pool is shared with `demo_cpu_tasks`' Terminal-Bench rows). Key pieces:
  `swebench.py` (task class + hidden grader), `terminalbench_pool.py` (fresh session per episode +
  pre-open + strict routing, on top of `harness_pool.py`), `terminalbench_pool_generate.py` (the slime hook), `tb_lookahead_source.py`
  (look-ahead pre-open), `session_keeper.py` (heartbeat).
- `fetch_swebench.py` — select the subset → `swebench/manifest.json` + `data/` (`--data-only` rebuilds data).
- `deploy/deploy_swebench.py` — wrap + push one arm64 image per instance, one microVM Runtime each.
- `survey_image_sizes.py` / `probe_image_cap.py` — the arm64 availability survey and the 2 GB cap probe.
- `deploy/acs_swe_http_server.py` — the in-image server (chunked bodies).
- `swebench_oracle_check.py` — gold/empty check per deployed instance → `swebench/oracle.csv`.
- `prepare_data.sh` (step 1) · `run.sh` (step 5) · `metrics.py` (run by step 5) · `train.sh` (convergence).
- `1_…6_*.sh` + `deploy/demo.env` — the numbered steps (BUILD/PUSH/DEPLOY/CLEANUP_CMD overrides).
- `swebench/` (gitignored) — `manifest.json` (holds the hidden test patches), `arn_by_image.json`,
  `images.csv`, `oracle.csv`, build logs, `m5_legacy/` (the retired m5 pool's state).
- `data/` — `swebench_train.jsonl` / `swebench_eval.jsonl`.
