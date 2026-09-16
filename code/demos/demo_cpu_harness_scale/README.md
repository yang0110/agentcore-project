# demo_cpu_harness_scale — scaling the **harness** tier (mini-Claude harness pool, one task)

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
**W warm workers** and route episodes to them. Here `W = HARNESS_WORKERS` (default **4**, symmetric
with the demo's 4 SGLang inference engines and 4 sandbox Runtimes).

The harness driven here is the **"mimic Claude Code" `MiniClaudeHarness`** (the A3/B3 agent): 3 tools
(`write_file` / `bash` / `read_file`), speaking the **Anthropic Messages API** to slime's in-process
`AnthropicAdapter` (which serves completions from the policy under training and records the
trajectory), with the tools running in the remote CPU sandbox Runtime. Reward is **RLVR**: a hidden
`run_checks.py` numeric-asserts the answer (tol 1e-4) — no LLM judge.

- **Task:** DeepScaleR math, coding-shape (`solution.py` sets a module-level `answer`) · one task, end-to-end
- **Dataset:** `agentica-org/DeepScaleR-Preview-Dataset` — `data/deepscaler_train.jsonl` (400) + `data/deepscaler_eval.jsonl` (40)
- **The knob:** `HARNESS_WORKERS=W` (default 4) — the harness tier of `min(inference, sandbox, harness)`
- **Placement:** in-process asyncio worker pool (see the honest GIL caveat below)

---

## How prompts are routed to each tier (this is the point of the demo)

The three replicated tiers scale **and load-balance differently** — the asymmetry is deliberate:

| Tier | Replicas | State | Who load-balances | Granularity |
|---|---|---|---|---|
| **inference** | `E` SGLang engines (= `ROLLOUT_GPUS / GPUS_PER_ENGINE`, default 4) | **stateless** per request | slime's **SGLang router** (cache-aware) across the `E` engines | per **request** — a single episode's turns may land on different engines |
| **sandbox** | `K` sandbox Runtimes (ARN list) | **stateful** per episode — holds `solution.py`, `run_checks.py` | `make_sandbox` **round-robins** a *fresh session* per episode across the `K` ARNs (`_next_runtime_arn`) | per **episode**, **sticky** for its whole life |
| **harness** | `W` warm workers (`HARNESS_WORKERS`) | stateless but **warm** (owns a reused `aiohttp` session) | a shared `asyncio.Queue`; whichever worker is **free pulls** the next episode (work-stealing) | per **episode**, by **availability** |

Why pull, not push, for the harness: multi-turn agentic episodes have **wide, variable latency**. A
push / hash assignment would hot-spot a worker stuck on a slow episode while others idle; **pull /
work-stealing balances by availability**, which is robust to that variance. A worker owns its episode
start→finish (it holds the `aiohttp` session and the adapter session id), so the episode is sticky to
its worker exactly as it is sticky to its sandbox.

slime admits up to `S = CONC × E` episodes concurrently, but only `W` are actively **driven by a
harness** at once. If `S > W`, admitted episodes queue for a free worker — so `W` is the harness
ceiling in `min(...)`. `run.sh` defaults `CONC` so `S ≈ W` (the balanced start point); raise `W` to
lift the harness tier, and watch throughput rise until it hits the scarcer of inference/sandbox.

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

## Reproduce (run one by one)

Everything runs INSIDE the persistent container `agentcore_p4` (`slimerl/slime`); the repo is at
`/work/code`. `data/` is already populated — step 1 only re-pulls it from scratch.

```bash
# 1. Download + prepare the dataset (one-time network to pull the HF rows) — builds deepscaler 400/40
docker exec agentcore_p4 bash /work/code/demos/demo_cpu_harness_scale/prepare_data.sh

# 2. See the scaling law offline (no AWS, no GPU): sweep W and watch the knee + plateau
docker exec agentcore_p4 python3 /work/code/demos/demo_cpu_harness_scale/harness_scaling_demo.py

# 3. Real RL train + eval (async/decoupled slime) through the WARM worker pool.
#    Smoke uses NR=3; converge with a larger NR. HARNESS_WORKERS sweeps the harness tier.
docker exec -e NR=3 -e EVAL_INTERVAL=2 -e HARNESS_WORKERS=4 agentcore_p4 \
    bash -lc 'cd /work && bash code/demos/demo_cpu_harness_scale/run.sh'
```

Step 2 is the cheapest way to *see* the claim: it drives the real `run_episode` on a `LocalSandbox`
through `W` warm asyncio workers behind modelled inference/sandbox ceilings, measures wall-clock
throughput, and prints which tier binds at each `W` (throughput rises ~linearly while the harness
binds, then plateaus at `min(inference, sandbox)`).

Real remote runs (step 3) need `AGENTCORE_SANDBOX_RUNTIME_ARN(S)` + AWS creds in the container
(`run.sh` ships a pinned CPU-sandbox ARNS default). `SANDBOX=local` runs fully offline.

> **Adapter/tool-parser requirement (the A3/B3 dead-reward fix):** `run.sh` bakes in
> `--sglang-tool-call-parser qwen25 --sglang-reasoning-parser qwen3`. Without them the adapter cannot
> turn Qwen3's raw tool syntax into `tool_use` blocks, the mini-harness tool loop is dead, and reward
> is structurally 0. Do not remove them.

## Knobs (env vars; defaults in `run.sh`)

| Knob | Default | Tier / meaning |
|---|---|---|
| `HARNESS_WORKERS` | 4 | **W** — warm harness workers (the tier this demo scales) |
| `ROLLOUT_GPUS` / `GPUS_PER_ENGINE` | 4 / 1 | ⇒ **E** SGLang inference engines (4) |
| `AGENTCORE_SANDBOX_RUNTIME_ARNS` | 2 pinned CPU ARNs | **K** sandbox Runtimes (round-robined per episode) |
| `CONC` | `ceil(W/E)` | episode-admission concurrency; `S = CONC × E` |
| `NR` | 11 | num-rollout ⇒ real steps `NR-1`; keep `NR×RB ≤ 400` |
| `RB` / `NS` | 4 / 8 | GRPO prompts-per-step / group size (`GBS = RB×NS`) |
| `EVAL_INTERVAL` / `EVAL_NS` | 10 / 2 | held-out eval cadence / samples |
| `EPISODE_MAX_TURNS` | 4 | mini-harness turn budget per episode |
| `TEMP` | 1.0 | rollout sampling temperature |
| `SANDBOX` | `agentcore-runtime` | `agentcore-runtime` \| `agentcore` (Code-Interpreter) \| `local` (offline) |

## Customize the sandbox runtime build (Dockerfile)

The sandbox image is built from [`deploy/Dockerfile.sandbox.cpu`](deploy/Dockerfile.sandbox.cpu) —
the same CPU sandbox server (`sandbox_server_agent.py`, a `BedrockAgentCoreApp` exposing
`/invocations` write_file/exec/read_file) the other CPU demos use. To rebuild + redeploy the CPU
sandbox Runtime and get fresh ARNs, use the shared deploy orchestrator:

```bash
# from the host (needs docker + AWS creds); builds this demo's Dockerfile, pushes to ECR, (re)creates
# the Runtime, and prints the ARN to feed AGENTCORE_SANDBOX_RUNTIME_ARNS.
python3 code/deploy/deploy_two_runtime.py --sandbox-only \
    --demo-dir code/demos/demo_cpu_harness_scale --dockerfile deploy/Dockerfile.sandbox.cpu
```

The adapter and the harness workers run **in the trainer process** (localhost), so — unlike the
removed two-Runtime variant — there is **no** harness Runtime to build and no public adapter port /
security-group to open. Only the *sandbox* Runtime is remote.

## Metrics → CSV + PNG

```bash
docker exec agentcore_p4 python3 /work/code/demos/demo_cpu_harness_scale/metrics.py <train.log> \
    --csv train.csv --eval-csv train.eval.csv --plot train.png
```

Watch `rollout/raw_reward` (learning signal), `eval/deepscaler` (held-out solve-rate), and the
`perf/sandbox_ms_*` vs `perf/sglang_gen_ms_*` split. Each finished sample carries
`harness_worker` metadata (which of the `W` workers ran it) so you can confirm the pull-queue spread
work evenly.

## Folder layout (self-contained)

- `agentcore_slime_demo/` — vendored shared library; **this demo adds** `harness_pool.py` (the warm
  worker pool + full routing model) and `harness_pool_generate.py` (the slime `--custom-generate-function-path`
  hook). `mini_agent.py` is the shared `MiniClaudeHarness` with one added `http=` param so a worker
  can reuse its warm `aiohttp` session across episodes.
- `harness_scaling_demo.py` — the **offline** scaling-law demo (step 2; no AWS/GPU)
- `deploy/Dockerfile.sandbox.cpu` — the CPU sandbox Runtime image
- `data/` — `deepscaler_train.jsonl` / `deepscaler_eval.jsonl`
- `prepare_data.sh` — build the dataset (step 1) · `run.sh` — the RL run (step 3) · `metrics.py`

Shared by reference (NOT vendored): the base Megatron checkpoint
(`code/p4_slime_train/ckpt/Qwen3-4B_torch_dist`, override `CKPT_BASE`), the slime install, and the
deploy orchestrator (`code/deploy/deploy_two_runtime.py`).
