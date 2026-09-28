# demo_cpu_swebench — RL on real GitHub issues (SWE-bench Verified), one image per dataset row, one isolated m5 per episode

**Sandbox tier 3: own Runtime, custom image — on a capacity-provider Instance.** Every SWE-bench
instance pins its own repo commit + conda env, so every dataset row gets its own image and Runtime.
The images are too big for the microVM fleet, so the Runtimes sit on m5 Instances. See the tier table
in [`../README.md`](../README.md#which-sandbox-does-a-task-need-three-tiers).

Split out of `demo_cpu_terminalbench` (where it was `BENCH=swebench`) on 2026-09-28. It reuses that
demo's image-pool machinery unchanged (vendored): `TerminalBenchPool` (fresh session per episode,
pre-open, heartbeat, strict routing by `task.image` = instance id) and `terminalbench_pool_generate`.
What is SWE-bench-specific: the substrate (m5 Instances), the per-instance images, and the hidden-test grader.

The policy is Qwen3-4B trained by slime (async/decoupled GRPO). The agent is the tool-calling loop
(`MiniClaudeHarness` → `AnthropicAdapter` → the policy under training; tools run in the sandbox), so it
needs `--sglang-tool-call-parser qwen25 --sglang-reasoning-parser qwen3` (baked into `run.sh`). The
reward is the instance's own hidden FAIL_TO_PASS + PASS_TO_PASS tests (RLVR — no LLM judge).
**REMOTE-ONLY** (`SANDBOX=local` is rejected).

---

## Use case covered

RL on real GitHub issues: the agent edits a real repo at `/testbed` (django, sympy, sphinx, pytest,
astropy, requests, xarray, pylint, flask) and is rewarded when the instance's hidden tests pass.
The current subset is 20 instances, one Runtime each (16 train + 4 eval).

## Why this demo

- It is **the** standard agentic-coding benchmark.
- It is the extreme case of **one image per dataset row**: the dataset cannot be served by one
  generic image, because each row is a different repo, commit and dependency set.
- It shows that the image-pool pattern also works for images the **microVM fleet cannot run**:
  SWE-bench images are amd64-only and 1-2 GB compressed (3-6 GB on disk), while microVMs are arm64 with a
  2 GB image cap. Each Runtime therefore sits on an **m5.xlarge capacity provider** (`acs_swe_cp`,
  60 GiB root). One session is one whole instance, so isolation is still per machine.

## Design considerations

- *One Runtime per row.* `deploy/deploy_swebench.py` wraps each prebuilt
  `swebench/sweb.eval.x86_64.<id>` image with a standalone CPython at `/opt/acs/py` + the AgentCore
  HTTP server, pushes it to ECR (`acs-swe-*`) and registers a Runtime (`acs_swe_*`). The testbed env
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
  go into `data/`. **As of 2026-09-28 only 1 of 20 is checked** (`django__django-15127`, usable); step 4
  runs the check for all of them.
- *Selection.* `fetch_swebench.py` picks "<15 min fix" instances with 1-3 F2P, 1-150 P2P, at most
  2 test files and an image of at most 3 GB, from repos the vendored parsers support (quota: django 6,
  sympy 4, sphinx 3, pytest 2, one each for the rest). matplotlib, sklearn and seaborn are excluded
  (images of 4-8 GB, slow suites).
- *Cold start.* Opening a session boots an m5 instance and pulls the image (~100 s measured), much
  slower than a microVM, so keep `TB_PREOPEN_BATCHES` on. Idle instances are released 600 s after their
  last session. Every pre-opened session is a whole m5 you pay for.
- *Server shim.* On Instances the data plane sends chunked `/invocations` bodies, which the shared
  `acs_http_server.py` misreads as a ping. The images therefore run `deploy/acs_swe_http_server.py`,
  which accepts both chunked and Content-Length bodies.
- *Sparse reward.* A 4B policy solving real repo bugs is hard; expect a reward at least as sparse as
  Terminal-Bench's (4/382 on its first smoke). A curriculum or easier instances may be needed for signal.

## How an episode flows

```
slime → terminalbench_pool_generate.generate → TerminalBenchPool.submit
  worker pulls episode → route by task.image (= instance id) → that instance's Runtime ARN
  (at submit) open a BRAND-NEW session on that ARN while the episode waits   ← its own m5 instance
    SessionKeeper heartbeats it until it is stopped
      MiniClaudeHarness loop in /testbed: LLM via adapter (policy under training), bash/write_file/read_file tools
    grade: swebench.grade_swebench (reset test files → apply hidden test_patch → test_cmd → parse F2P/P2P)
  finish_session → loss-masked GRPO samples (list); adapter.drop_session; StopRuntimeSession
```

## Step by step (numbered scripts; run from this dir on the HOST — container steps `docker exec` themselves)

```bash
bash 1_prepare_data.sh     # 1. container: fetch_swebench.py (select subset -> swebench/manifest.json)
                           #    then prepare_data.sh -> data/swebench_{train,eval}.jsonl
bash 2_build_image.sh      # 2. SWE_NO_REGISTER=1 deploy_swebench.py: wrap + push one amd64 image per instance (ECR acs-swe-*)
bash 3_push_ecr.sh         # 3. no-op: prints swebench/images.csv (step 2 already pushed)
bash 4_deploy_runtime.sh   # 4. deploy_swebench.py: capacity provider + one Runtime per instance -> swebench/arn_by_image.json,
                           #    then swebench_oracle_check.py -> swebench/oracle.csv, then fetch_swebench.py --data-only
bash 5_run.sh              # 5. container: 10 train steps + 1 eval through the pool; logs/<tag>.{log,csv,eval.csv,png}
bash 6_cleanup.sh          # 6. delete every Runtime in swebench/arn_by_image.json (DELETE_CP=1 also deletes acs_swe_cp)
```

- **Validate on 2 instances first** (real AWS $): `SWE_LIMIT=2 bash 2_build_image.sh && SWE_LIMIT=2 bash 4_deploy_runtime.sh`.
- `run.sh` loads `swebench/arn_by_image.json` itself, so `deploy/runtime.env` stays empty.
- Train to convergence (inside the container): `train.sh` (`STEPS`, `EVAL_EVERY`, `SANDBOX_CONC`).

## Knobs (env vars; defaults in `run.sh`)

| Knob | Default | Meaning |
|---|---|---|
| `SANDBOX` | `agentcore-runtime` | fresh AgentCore session per episode. **`local` is rejected.** |
| `TB_PREOPEN_BATCHES` | `2` | upcoming batches kept booted ahead (`× RB × NS` = 64 m5 sessions). `0` = off |
| `WARM_IDLE_TIMEOUT` / `WARM_PING_FRACTION` | `900` / `0.5` | heartbeat any session idle past `fraction × timeout` (always on) |
| `HARNESS_WORKERS` | `4` | warm in-process harness workers |
| `EPISODE_MAX_TURNS` / `EPISODE_MAX_TOKENS` | `12` / `1024` | agentic turns / tokens per episode |
| `TB_EXEC_TIMEOUT` | `180` | per bash-tool command timeout (s) |
| `MAX_TOKENS_PER_GPU` | `24576` | ≥ one full 12-turn trajectory |
| `RB` / `NS` / `NR` | 4 / 8 / 11 | GRPO prompts-per-step / group size / num-rollout (real steps `NR-1`) |
| `MODE` | `A` | stock async only — the pool returns a fan-out list; `B` is rejected |
| `EVAL_INTERVAL` / `EVAL_NS` | 10 / 2 | held-out eval cadence / samples |

- Deploy: `SWE_TASKS`, `SWE_LIMIT`, `SWE_BUILD_CONC`, `SWE_REBUILD`, `SWE_NO_REGISTER`,
  `SWE_NO_PUSH` (local build; then `swebench_oracle_check.py --local`), `SWE_INSTANCE`, `SWE_ROOT_GIB`.
- Data: `SWE_QUOTA`, `SWE_INSTANCES`, `SWE_MAX_TURNS` (12), `SWE_TEST_TIMEOUT` (600 s, which keeps
  the grade exec under `INVOKE_READ_TIMEOUT` 900).

## What to watch

- `rollout/raw_reward` and `eval/swebench` — the learning signal / held-out solve rate.
- `[demo] fresh_session=1 preopened=… open_wait_ms=… preopen_idle_ms=…` — one line per episode;
  `open_wait_ms` near 0 means the m5 cold start was hidden by the pre-open.
- `perf/sandbox_ms_*` vs `perf/sglang_gen_ms_*` — the sandbox-vs-decode latency split.

## Folder layout (self-contained)

- `agentcore_slime_demo/` — vendored library (shared with `demo_cpu_terminalbench`). Key pieces:
  `swebench.py` (task class + hidden grader), `terminalbench_pool.py` (fresh session per episode +
  pre-open + strict routing), `terminalbench_pool_generate.py` (the slime hook), `tb_lookahead_source.py`
  (look-ahead pre-open), `session_keeper.py` (heartbeat).
- `fetch_swebench.py` — select the subset → `swebench/manifest.json` + `data/` (`--data-only` rebuilds data).
- `deploy/deploy_swebench.py` — wrap + push one image per instance, capacity provider, one Runtime each.
- `deploy/acs_swe_http_server.py` — the in-image server (chunked bodies).
- `swebench_oracle_check.py` — gold/empty check per deployed instance → `swebench/oracle.csv`.
- `prepare_data.sh` (step 1) · `run.sh` (step 5) · `metrics.py` (run by step 5) · `train.sh` (convergence).
- `1_…6_*.sh` + `deploy/demo.env` — the numbered steps (BUILD/PUSH/DEPLOY/CLEANUP_CMD overrides).
- `swebench/` (gitignored) — `manifest.json` (holds the hidden test patches), `arn_by_image.json`,
  `images.csv`, `oracle.csv`, build logs.
- `data/` — `swebench_train.jsonl` / `swebench_eval.jsonl`.
