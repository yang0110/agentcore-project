# demo_cpu_mixtask — mixed-task RL (one batch spans several tasks)

Every other CPU demo pins **one** task family per run. This one does the opposite: each GRPO
rollout **step trains a batch of prompts that spans several families at once** (default: **math +
sql + ctf + tool_use**). This is the realistic agentic-RL setting — you rarely train a policy on a
single task; you train it on a *mixture* and want one policy good at all of them.

The policy is Qwen3-4B trained by slime (async/decoupled GRPO). The agent harness is **in-process**
(a warm `run_episode` loop next to the trainer). The sandbox is a **remote, managed, serverless
AgentCore Runtime** session — **this demo is REMOTE-ONLY, it never runs a local sandbox** (see
[Remote-only](#remote-only-no-local-sandbox)).

This folder is **fully self-contained**: it owns its `data/`, `ckpt/`, `run.sh`, a **vendored copy of
`agentcore_slime_demo/`** (imported only from here — `PYTHONPATH` puts this dir first, `/work/code`
is not on the path), and its own `deploy/Dockerfile.sandbox.cpu`. You can read this one folder
end-to-end. The only things it uses from outside are prerequisites already present in the
`agentcore_p4` container: the slime install (`/root/slime`), the base Qwen3-4B Megatron checkpoint
(`CKPT_BASE`, default `/work/code/p4_slime_train/ckpt/Qwen3-4B_torch_dist`), and AWS credentials for
the remote sandbox. No other demo, doc, or script is required to understand or run this one.

---

## Why a mixed batch needs *no* change to the harness or the sandbox

Two properties make mixing essentially free:

1. **Every dataset row is self-describing.** A row is
   `{"prompt": ..., "label": ..., "metadata": {"task": {"family": ..., "task_id": ..., "spec": ...}}}`.
   At rollout time `agentcore_slime_demo.generate._task_from_sample` reads `metadata.task.family`
   and rebuilds the correct `EpisodicTask` subclass — which supplies that prompt's **seed files**,
   its **hidden RLVR checker**, and its **reward**. So `run_episode` already dispatches *per row*; a
   step containing math, sql, ctf and tool_use prompts side by side is transparent to it.

2. **The sandbox is task-agnostic.** The remote AgentCore Runtime runs a generic
   `write_file` / `exec` / `read_file` server (`agentcore_slime_demo/sandbox_server_agent.py`). All
   task specificity lives in the files the harness writes **into that episode's session** at episode
   start (`setup_files()` + the hidden `run_checks.py` from `runner_source()`). A math episode, a sql
   episode and a ctf episode are just *different files written into three identical, isolated
   sessions*.

### The one thing worth understanding: GRPO grouping
GRPO forms a group of `n_samples_per_prompt` (`NS`) repeats of the **same prompt** and normalizes
advantage **within that group**. Mixing happens **across groups within a step**, never inside a
group. So the very different reward *shapes* of the families — math (numeric tolerance, 0/1), sql
(result-set exact match, 0/1), ctf (flag accepted, 0/1), tool_use (arg-normalized call match, 0/1) —
**never need to be reconciled or rescaled**: within-group normalization makes each group's advantage
scale-free. Keep `NS` intact; only the prompt **sampler** mixes. This is *why* multi-task GRPO is
clean.

---

## How AgentCore runtime sessions handle the mixed batch

This is the part clients ask about. A rollout step admits `S = CONC × num_engines` episodes
concurrently, and those episodes are a heterogeneous mix of families. AgentCore handles it with **no
special casing**, because of how sessions map to work:

- **One runtime session per episode.** For each episode the harness opens a fresh
  `runtimeSessionId` (`sbx-<uuid>`) via `invoke_agent_runtime`. That session is a **stateful,
  isolated** microVM for the episode's whole life; the harness writes that episode's task files into
  it, runs the agent's turns, runs the checker, then the session is dropped. Because isolation is
  per-episode, a heterogeneous batch is just *different files in identical isolated sessions* — the
  Runtime image is the same generic CPU sandbox for all families.
- **Round-robin across `K` Runtimes, family-agnostic.** `make_sandbox` round-robins a fresh session
  across the `K` pinned Runtime ARNs (`AGENTCORE_SANDBOX_RUNTIME_ARNS`, `_next_runtime_arn`). So the
  `S` concurrent heterogeneous sessions spread evenly across the `K` endpoints; each Runtime just
  serves a mix. The session count — and therefore the cost — is **unchanged** by mixing (still one
  session per episode).
- **When you would route by family instead.** Only if a family needed a *different substrate*
  (e.g. SQL wanting a DB engine baked into the image, or a family needing a **GPU**, or network
  egress). Then you partition `K` by capability and pick the ARN by `family` rather than pure
  round-robin. This demo's four families are all pure-CPU / stdlib, so **one generic CPU image
  serves all of them** and plain round-robin is correct. (The GPU demos are the example of the
  capability-routed variant — their checker needs CUDA, so the sandbox is a GPU Runtime.)

---

## Remote-only (no local sandbox)

This is a **CPU** demo, so it runs **only** on a remote, managed, serverless AgentCore sandbox:
`SANDBOX=agentcore-runtime` (default) or `SANDBOX=agentcore` (Code-Interpreter microVM). `run.sh`
**rejects `SANDBOX=local`** — a CPU training run must never silently fall back off the AgentCore
substrate it is meant to exercise, and the serverless CPU sandbox is always available. (Only the GPU
demos keep a local fallback, because a remote g6 instance may be unavailable.)

`SANDBOX=agentcore-runtime` **requires** `AGENTCORE_SANDBOX_RUNTIME_ARNS` (comma-separated, one or
more CPU sandbox Runtime ARNs) and AWS credentials in the container. `run.sh` ships a pinned pair of
CPU ARNs as the default (`K=2`); override to point at your own.

---

## Reproduce (run one by one, inside the `agentcore_p4` container)

```bash
# 1. Build the MIXED dataset (offline, stdlib-only — no network, no GPU). Interleaves the families.
docker exec agentcore_p4 bash /work/code/demos/demo_cpu_mixtask/prepare_data.sh
#    -> data/mixed_train.jsonl (198 prompts: math+sql+ctf+tool_use, balanced) + data/mixed_eval.jsonl

# 2. REAL RL train + eval through the REMOTE AgentCore sandbox (async/decoupled slime). Smoke = NR=3.
#    Requires AGENTCORE_SANDBOX_RUNTIME_ARNS + AWS creds in the container (run.sh ships pinned ARNs).
docker exec -e NR=3 -e EVAL_INTERVAL=2 agentcore_p4 \
    bash -lc 'cd /work && bash code/demos/demo_cpu_mixtask/run.sh 2>&1 | tee code/demos/demo_cpu_mixtask/smoke_run.log'

# 3. Parse the log into CSVs + a PNG of the curves.
docker exec agentcore_p4 python3 /work/code/demos/demo_cpu_mixtask/metrics.py \
    /work/code/demos/demo_cpu_mixtask/smoke_run.log \
    --csv train.csv --eval-csv train.eval.csv --plot train.png
```

The harness extracts the agent's submission from a fenced code block (` ```python ` / ` ```sql `)
in each completion and writes it into the sandbox session — no Anthropic `tool_use` parsing is
involved, so no special SGLang tool-call parser is required here.

## Change the mix

```bash
# any subset of: math sql ctf tool_use swe rag regex planning config data
FAMILIES=math,sql,swe,rag,config BALANCED=1 REPEAT=6 \
    docker exec agentcore_p4 bash /work/code/demos/demo_cpu_mixtask/prepare_data.sh
```

- `BALANCED=1` (default) round-robin interleaves families so **every step spans them evenly**;
  `BALANCED=0` does a plain shuffle (proportional to each family's seed count).
- `REPEAT=N` replicates the mixed train pool so a rollout step's batch has enough distinct prompts to
  stay varied across the run.
- **Save-only-at-end:** the **only checkpoint written is the final one**. `run.sh` passes an explicit
  `--num-rollout` (so slime leaves `num_rollout_per_epoch = None` and its epoch-boundary save never
  fires) plus a huge `SAVE_INTERVAL` (so the periodic save never fires); the sole save is slime's
  forced final-rollout checkpoint. This holds regardless of `REPEAT` / dataset size.

## Knobs (env vars; defaults in `run.sh`)

| Knob | Default | Meaning |
|---|---|---|
| `SANDBOX` | `agentcore-runtime` | remote sandbox: `agentcore-runtime` \| `agentcore`. **`local` is rejected.** |
| `AGENTCORE_SANDBOX_RUNTIME_ARNS` | 2 pinned CPU ARNs | **K** sandbox Runtimes (round-robined per episode) |
| `RB` / `NS` | 4 / 8 | GRPO prompts-per-step / group size (`GBS = RB×NS`) |
| `NR` | 11 | num-rollout ⇒ real steps `NR-1`; keep the train set ≥ `NR×RB` (default mix = 198) |
| `CONC` | 4 | sandbox/episode concurrency; episodes in flight `S = CONC × num_engines` |
| `MODE` | `B` | `B` fully-async (deep buffer) \| `A` 1-step-async |
| `TEMP` | 1.0 | rollout sampling temperature |
| `EVAL_INTERVAL` / `EVAL_NS` | 10 / 2 | held-out (mixed) eval cadence / samples |

## What to watch
`rollout/raw_reward` (overall learning signal across the mix), `eval/mixed` (held-out solve-rate on
the interleaved eval set), and the `perf/sandbox_ms_*` vs `perf/sglang_gen_ms_*` split. `metrics.py`
also writes per-op sandbox/decode latency CSVs. Per-family eval files (`data/<family>_eval.jsonl`)
are emitted too, so you can score any single family in isolation with a one-off eval run.

## Folder layout (self-contained)

- `agentcore_slime_demo/` — vendored library (the in-process harness `generate.py` +
  `mini_agent.py`, the remote sandbox client `agentcore_runtime_sandbox.py`, the task registry
  `task_families.py`, the sandbox server `sandbox_server_agent.py`, …). Imported only from here.
- `build_mixed_datasets.py` — builds the interleaved multi-family dataset (offline, stdlib).
- `prepare_data.sh` — step 1 wrapper · `run.sh` — the remote RL run (step 2) · `metrics.py` — step 3.
- `deploy/Dockerfile.sandbox.cpu` — the CPU sandbox Runtime image (same generic exec server the other
  CPU demos use). Rebuild/redeploy with the shared deploy orchestrator to get fresh ARNs.
- `data/` — `mixed_train.jsonl` / `mixed_eval.jsonl` (+ per-family `<family>_eval.jsonl`).
