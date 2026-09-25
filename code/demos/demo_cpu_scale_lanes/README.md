# demo_cpu_scale_lanes — scaling the rollout as **4 × 4 × 4 lanes** (4 harness workers : 4 SGLang engines : 4 microVMs)

*(Formerly `demo_cpu_harness_scale`.)* The default topology is **4 lanes**. Each lane is one warm
harness worker, pinned to **its own SGLang engine** (1 GPU, TP1) and **its own persistent AgentCore
microVM** (one sandbox session, opened once and reused for every episode that worker drives, with a
fresh workdir per episode). So a run is exactly **4 harness workers : 4 inference engines : 4
microVMs**, on 4 train GPUs + 4 inference GPUs. `LANES=N` (1, 2 or 4) scales all three tiers
together; overriding one tier alone (`HARNESS_WORKERS`, `ROLLOUT_GPUS`) unbalances it so you can
watch it become the ceiling.

```
lane 0:  harness worker 0 ──► SGLang engine 0 (GPU 4) ──► microVM session lane0-…  (ARN[0])
lane 1:  harness worker 1 ──► SGLang engine 1 (GPU 5) ──► microVM session lane1-…  (ARN[1])
lane 2:  harness worker 2 ──► SGLang engine 2 (GPU 6) ──► microVM session lane2-…  (ARN[0])
lane 3:  harness worker 3 ──► SGLang engine 3 (GPU 7) ──► microVM session lane3-…  (ARN[1])
          ▲ episodes pulled off one shared queue by whichever worker is free
```

Every episode logs `[demo] worker=w … lane=w engine=… sandbox_session=… lane_episode=n`, and
`metrics.py` checks the lanes are 1:1:1 (each worker used one engine and one session; no session
shared), printing the worker → engine → session table.

The rest of this README explains the scaling law the lanes demonstrate.

The other demos hold the harness **in-process and single** and vary the *task* + *sandbox*. This
demo does the opposite: it fixes the task (one, end-to-end — **DeepScaleR** math) and makes the
**harness itself the scalable tier**, to demonstrate the async-RL rollout scaling law:

```
throughput  =  min( inference , sandbox , harness )
```

A rollout is produced by three replicated tiers. If you deepen the async buffer and scale inference
+ sandbox but keep a **single** process driving every episode, the *harness* silently becomes the
ceiling (its concurrency, the asyncio loop, file descriptors, RAM for in-flight trajectories). So the
harness has to scale too — and because a harness is **persistent/warm** while a sandbox is **ad-hoc
per episode**, "scale the harness" does **not** mean "make it ad-hoc": it means run a fixed set of
**W warm workers** and route episodes to them. Here `W = HARNESS_WORKERS` (default `LANES` = **4**),
matched by **4** SGLang engines (`ROLLOUT_GPUS=4`, one GPU each) and **4** persistent microVMs (one
session per worker, spread over the pinned `K=2` sandbox Runtime ARNs as `ARN[w % 2]`) — the
balanced 4+4+4 point. Lower one tier (e.g. `ROLLOUT_GPUS=2`) and throughput drops to that tier's
ceiling.

The harness driven here is the **"mimic Claude Code" `MiniClaudeHarness`** (the A3/B3 agent): 3 tools
(`write_file` / `bash` / `read_file`), speaking the **Anthropic Messages API** to slime's in-process
`AnthropicAdapter` (which serves completions from the policy under training and records the
trajectory), with the tools running in the remote CPU sandbox Runtime. Reward is **RLVR**: a hidden
`run_checks.py` numeric-asserts the answer (tol 1e-4) — no LLM judge.

- **Task:** DeepScaleR math, coding-shape (`solution.py` sets a module-level `answer`) · one task, end-to-end
- **Dataset:** `agentica-org/DeepScaleR-Preview-Dataset` — `data/deepscaler_train.jsonl` (400) + `data/deepscaler_eval.jsonl` (40)
- **The knob:** `LANES=N` (default 4) — N workers : N engines : N microVMs. `HARNESS_WORKERS=W` alone moves just the harness tier of `min(inference, sandbox, harness)`. The offline simulator also exposes a **balanced joint sweep** (`TIERS=N,…`) that scales **inference = sandbox = harness = N** in lockstep (the **4+4+4** case) — the complement to the harness-only sweep: because every tier rises together, no single one binds, so throughput scales ~linearly with `N` instead of plateauing.
- **Placement:** in-process asyncio worker pool (see the honest GIL caveat below)

---

## How prompts are routed to each tier (this is the point of the demo)

The three replicated tiers scale **and load-balance differently** — the asymmetry is deliberate:

| Tier | Replicas | State | Who load-balances | Granularity |
|---|---|---|---|---|
| **inference** | `E` SGLang engines (= `ROLLOUT_GPUS / GPUS_PER_ENGINE` = `4/1`, default 4) | **stateless** per request | each worker's adapter is **pinned** to engine `w % E` (bypassing the router's per-request LB) | per **worker** — all of an episode's turns hit its lane's engine |
| **sandbox** | `W` microVMs (one persistent session per worker, on ARN `w % K`) | **stateful** per episode — holds `solution.py`, `run_checks.py` in a fresh `/workspace/ep-<hex>` workdir, deleted after | the worker's **own** session (`LANE_SANDBOX=1`); `LANE_SANDBOX=0` restores a fresh session per episode round-robined over the `K` ARNs | per **worker** (lane), sticky for the run |
| **harness** | `W` warm workers (`HARNESS_WORKERS`) | stateless but **warm** (owns a reused `aiohttp` session) | a shared `asyncio.Queue`; whichever worker is **free pulls** the next episode (work-stealing) | per **episode**, by **availability** |

Why pull, not push, for the harness: multi-turn agentic episodes have **wide, variable latency**. A
push / hash assignment would hot-spot a worker stuck on a slow episode while others idle; **pull /
work-stealing balances by availability**, which is robust to that variance. A worker owns its episode
start→finish (it holds the `aiohttp` session and the adapter session id), so the episode is sticky to
its worker exactly as it is sticky to its sandbox.

slime admits up to `S = CONC × E` episodes concurrently, but only `W` are actively **driven by a
harness** at once. If `S > W`, admitted episodes queue for a free worker — so `W` is the harness
ceiling in `min(...)`. `run.sh` defaults `CONC = ceil(2W/E)`, so `S = 2W`: `W` episodes running (one
per lane) and `W` queued, so a lane never idles waiting for slime to admit its next episode.

**Why persistent microVMs are safe here.** The DeepScaleR checker only reads the episode's own
workdir, and each episode gets a fresh one, so reusing a microVM across episodes leaks nothing that
the reward depends on — and it removes the per-episode session cold start. (Tasks that change the
whole OS need a fresh session per episode instead: see `demo_cpu_terminalbench`.) If a lane's
session is idle-reclaimed (15 min), the next invoke on the same session id just boots a new microVM;
the per-episode workdir is recreated on demand. Lane sessions are stopped at process exit.

Full model + code: [`agentcore_slime_demo/harness_pool.py`](agentcore_slime_demo/harness_pool.py) and
[`../../../docs/harness-scaling.md`](../../../docs/harness-scaling.md).

### Honest ceiling caveat (in-process placement)

In-process asyncio workers lift the **IO-concurrency** ceiling only — episodes spend almost all
wall-clock **awaiting** (LLM turns over HTTP to the adapter, tool exec over HTTP to the sandbox
Runtime), which asyncio overlaps well. They do **not** lift the **CPU** ceiling: the GIL still
serialises the CPU-bound bits (tokenisation / trajectory flattening in the adapter's
`finish_session`). Lifting *that* needs separate processes or dedicated harness Runtimes — a
different placement, out of scope here. This is why the demo studies the **IO-bound** regime where
in-process warm workers are the right, cheap answer.

---

## Step by step (numbered scripts; run from this dir on the HOST — container steps `docker exec` themselves)

```bash
bash 1_prepare_data.sh     # 1. container: deepscaler 400/40 (ships in data/; re-pulls the HF rows)
bash 2_build_image.sh      # 2. arm64 image from deploy/Dockerfile.sandbox.cpu
bash 3_push_ecr.sh         # 3. -> ECR acs-scalelanes-cpu (deploy/images.env)
bash 4_deploy_runtime.sh   # 4. Runtime acs_scalelanes_cpu -> deploy/runtime.env (AGENTCORE_SANDBOX_RUNTIME_ARNS)
bash 5_run.sh              # 5. container: 10 train + 1 eval through the warm lanes (LANES=4); logs/<tag>.{log,csv,eval.csv,png}
bash 6_cleanup.sh          # 6. delete the Runtime
```

Without step 4, `5_run.sh` uses the baked K=2 `acs_sandbox_cpu` ARNs (lane `w` on `ARN[w % K]`).
`LANES=1|2|4` scales all three tiers together; `HARNESS_WORKERS` alone sweeps just the harness tier.
Stock driver only (`MODE=A`, pinned): the pool's fan-out LIST of Samples is incompatible with the
fully-async Mode B sort key, so `run.sh` rejects `MODE=B`.

**Offline scaling law (no AWS, no GPU)**, run in the container:
```bash
python3 harness_scaling_demo.py                            # sweep W: knee + plateau at min(inference, sandbox)
TIERS=1,2,4,8 EPISODES=24 python3 harness_scaling_demo.py  # balanced joint sweep: every tier = N, ~linear
REAL_SANDBOX=1 SANDBOX=agentcore-runtime AGENTCORE_SANDBOX_RUNTIME_ARN=<arn> \
  EPISODES=8 WORKERS=1,2,4 python3 harness_scaling_demo.py # sandbox tier = measured remote round-trip
```
The first sweep drives the real `run_episode` on a `LocalSandbox` through `W` warm asyncio workers
behind modelled inference/sandbox ceilings and prints which tier binds at each `W`. `REAL_SANDBOX=1`
opens one live remote sandbox **per warm worker** (reused across its episodes) and round-robins the
workers across the ARNs, so the `sandbox` ceiling is measured instead of the modelled `--tsbx`
(`--csbx` still applies). Keep `EPISODES` small — each episode is a real network call.

> **Adapter/tool-parser requirement (the A3/B3 dead-reward fix):** `run.sh` bakes in
> `--sglang-tool-call-parser qwen25 --sglang-reasoning-parser qwen3`. Without them the adapter cannot
> turn Qwen3's raw tool syntax into `tool_use` blocks, the mini-harness tool loop is dead, and reward
> is structurally 0. Do not remove them.

## Knobs (env vars; defaults in `run.sh`)

| Knob | Default | Tier / meaning |
|---|---|---|
| `LANES` | 4 | **N** lanes — sets `HARNESS_WORKERS` = `ROLLOUT_GPUS` = N (≤ 4 with 4 train GPUs) |
| `HARNESS_WORKERS` | `LANES` | **W** — warm harness workers (= persistent microVMs in lane mode) |
| `ROLLOUT_GPUS` / `GPUS_PER_ENGINE` | `LANES` / 1 | ⇒ **E** SGLang inference engines (4) |
| `LANE_SANDBOX` | 1 | 1 = one persistent microVM session per worker; 0 = fresh session per episode |
| `AGENTCORE_SANDBOX_RUNTIME_ARNS` | 2 pinned CPU ARNs | **K** sandbox Runtimes; lane `w` lives on `ARN[w % K]` |
| `CONC` | `ceil(2W/E)` | episode-admission concurrency; `S = CONC × E` (default `2W`: W running + W queued) |
| `MODE` | `A` (pinned) | stock async driver only; `B` (fully-async) is **rejected** — the pool returns a fan-out LIST of Samples the Mode-B sort key can't order |
| `NR` | 11 | num-rollout ⇒ real steps `NR-1` (10 train + 1 eval); keep `NR×RB ≤ 400` |
| `RB` / `NS` | 4 / 8 | GRPO prompts-per-step / group size (`GBS = RB×NS`) |
| `EVAL_INTERVAL` / `EVAL_NS` | 10 / 2 | held-out eval cadence / samples |
| `EPISODE_MAX_TURNS` | 4 | mini-harness turn budget per episode |
| `TEMP` | 1.0 | rollout sampling temperature |
| `SANDBOX` | `agentcore-runtime` | `agentcore-runtime` \| `agentcore` (Code-Interpreter). **`local` is rejected** (remote-only) |
| `HEARTBEAT` / `HEARTBEAT_IDLE` / `HEARTBEAT_PING_FRACTION` | 0 / 600 / 0.5 | `1` = a `SessionKeeper` pings each lane's persistent microVM (and per-episode sessions) once idle past `fraction × idle` s, so lanes survive the weight-sync stall (was `demo_cpu_heartbeat`) |

## Customize the sandbox runtime build (Dockerfile)

The sandbox image is built from [`deploy/Dockerfile.sandbox.cpu`](deploy/Dockerfile.sandbox.cpu) —
the same CPU sandbox server (`sandbox_server_agent.py`, a `BedrockAgentCoreApp` exposing
`/invocations` write_file/exec/read_file) the other CPU demos use. Edit it, then rebuild and redeploy
with the step scripts (`FORCE_RECREATE=1` replaces an existing Runtime):

```bash
bash 2_build_image.sh && bash 3_push_ecr.sh && FORCE_RECREATE=1 bash 4_deploy_runtime.sh
```

The adapter and the harness workers run **in the trainer process** (localhost), so — unlike the
removed two-Runtime variant — there is **no** harness Runtime to build and no public adapter port /
security-group to open. Only the *sandbox* Runtime is remote.

## Metrics → CSV + PNG

`5_run.sh` runs `metrics.py` on its log and writes `logs/<tag>.{csv,eval.csv,png}`.

Watch `rollout/raw_reward` (learning signal), `eval/deepscaler` (held-out solve-rate), and the
`perf/sandbox_ms_*` vs `perf/sglang_gen_ms_*` split. Each finished sample carries
`harness_worker` metadata (which of the `W` workers ran it) so you can confirm the pull-queue spread
work evenly.

## Next: DeepCoder for the 256-sandbox run (data prep only)

The 256-sandbox plan (4 lanes × 64 workers, one microVM per worker) needs a task whose *check* is
expensive enough to load the sandbox tier, which DeepScaleR's single numeric assert is not.
[`agentcore_slime_demo/build_deepcoder.py`](agentcore_slime_demo/build_deepcoder.py) prepares one from
DeepCoder (`agentica-org/DeepCoder-Preview-Dataset`, `primeintellect` subset): it runs a gold Python
solution over every hidden stdin/stdout test and keeps only problems that pass, recording how long the
full suite takes.

```bash
docker exec agentcore_p4 bash -lc 'cd /work/code/demos/demo_cpu_scale_lanes && \
    python3 agentcore_slime_demo/build_deepcoder.py --probe 1500'          # -> data/deepcoder_probe.csv
docker exec agentcore_p4 bash -lc 'cd /work/code/demos/demo_cpu_scale_lanes && \
    python3 agentcore_slime_demo/build_deepcoder.py --train 512 --eval 64 --min-suite-s 8'
```

`data/deepcoder_probe.csv` is the committed 1500-problem probe: 74 % pass with the gold solution,
gold suite p50 4.2 s, median 101 tests per problem; the run keeps the ≥ 8 s tail (13 % of passing problems). **Not wired into
training yet:** the rows carry `family: "deepcoder"`, but no `DeepCoderTask` checker exists in the
vendored `task.py`, so `FAMILY=deepcoder` does not run until that lands.

## Folder layout (self-contained)

- `agentcore_slime_demo/` — vendored shared library; **this demo adds** `harness_pool.py` (the warm
  worker pool + full routing model) and `harness_pool_generate.py` (the slime `--custom-generate-function-path`
  hook). `mini_agent.py` is the shared `MiniClaudeHarness` with one added `http=` param so a worker
  can reuse its warm `aiohttp` session across episodes.
- `harness_scaling_demo.py` — the **offline** scaling-law demo (no AWS/GPU)
- `deploy/Dockerfile.sandbox.cpu` — the CPU sandbox Runtime image
- `data/` — `deepscaler_train.jsonl` / `deepscaler_eval.jsonl`; `deepcoder_probe.csv` (DeepCoder probe, see above)
- `1_…6_*.sh` + `deploy/` (`demo.env`, `steps_common.sh`, `agentcore_deploy.py`) — the numbered steps
- `prepare_data.sh` — build the dataset (step 1) · `run.sh` — the RL run (step 5) · `metrics.py`

Shared by reference (NOT vendored): the base Megatron checkpoint
(`code/p4_slime_train/ckpt/Qwen3-4B_torch_dist`, override `CKPT_BASE`) and the slime install.
