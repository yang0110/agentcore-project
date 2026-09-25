# demo_cpu_imagepool — mixed-task RL over a pool of prebuilt images (one image per task)

Every other CPU demo runs on **one** sandbox image. `demo_cpu_mixtask` mixes several task families
but they still share one generic image (plus a single pandas add-on). **This demo is the next step
the report calls a "described design, not [something] the shipped demos do":** a genuine
**capability-routed pool of images** — each rollout step's batch spans several families, and **each
family needs its *own* prebuilt sandbox image**. The harness **routes every episode to the image its
family needs**.

- `datasci` family → **pandas** image (`Dockerfile.sandbox.datasci`)
- `scipy` family → **scipy** image (`Dockerfile.sandbox.scipy`)
- `sklearn` family → **scikit-learn** image (`Dockerfile.sandbox.sklearn`)

None of those libraries is in the generic CPU image, so routing is not cosmetic: send a `scipy`
episode to the pandas image and its `solution.py`'s `import scipy` raises `ModuleNotFoundError`, the
hidden checker errors, and reward = **0**. Only the correctly-routed episode can score 1. Correct
routing is what makes a mixed-image batch *learnable*.

The policy is Qwen3-4B trained by slime (async/decoupled GRPO). The agent harness is **in-process**
(a warm `run_episode` loop next to the trainer). The sandboxes are **remote, managed, serverless
AgentCore Runtimes** — **one Runtime per image in the pool**. This demo is **REMOTE-ONLY**
(`SANDBOX=local` is rejected). It is **fully self-contained**: it owns its `data/`, `ckpt/`,
`run.sh`, a vendored copy of `agentcore_slime_demo/`, and its own `deploy/Dockerfile.sandbox.*`.

---

## Why this needs a pool of Runtimes (and mixtask did not)

The report's §"One image per Runtime, but a different env per row" gives three ways to reconcile
"one image per ARN" with "a different env per row":

- **A — per-session setup** (write files / install at seed time into one generic image). This is
  what `demo_cpu_mixtask` relies on: every family there is a pure-CPU / stdlib checker, so one
  generic image serves them all and the task specificity is just *files*.
- **B — a pool of Runtimes, one image per ARN, routed by row.** Used when the difference is
  *structural* — a library that cannot be a mere file. **This demo is B.** pandas, scipy and
  scikit-learn are heavyweight compiled deps; each is baked into its own image, one Runtime per
  image, and episodes are routed **by family**.
- **C — the hybrid** (shared base image per family + a cheap per-row delta at seed time).

So the Runtime count here tracks **distinct images (capabilities)**, not tasks: three images ⇒ three
Runtimes (times `K` if you also want throughput fan-out per image). A single `runtimeSessionId` still
maps 1:1 to one microVM booting that image.

## How routing works (already wired in the vendored library — no harness change)

The mechanism is entirely in the data + the sandbox factory; the harness and the trainer are
oblivious:

1. **Each task class carries an `image` tag.** `DataSciTask.image = "datasci"`,
   `ScipyTask.image = "scipy"`, `SklearnTask.image = "sklearn"` (`agentcore_slime_demo/task_families.py`).
   The dataset builder serializes `metadata.task.family`; `generate._task_from_sample` rebuilds the
   right `EpisodicTask` subclass per row and reads back its `image` tag.
2. **`generate.generate` passes `image=task.image` to `make_sandbox`.**
3. **`sandbox._runtime_arn_for_image(image)` maps the tag → a Runtime ARN** using the JSON env
   `AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE` (`{"datasci": <arn>, "scipy": <arn>, "sklearn": <arn>}`),
   falling back to the round-robin CPU pool (`AGENTCORE_SANDBOX_RUNTIME_ARNS`) for any unmapped tag.
4. AgentCore itself **never sees a task or an image at invoke time** — the image was fixed once at
   `CreateAgentRuntime`; an `invoke_agent_runtime` call carries only a `runtimeSessionId` + payload.
   The routing is entirely client-side (which ARN to open), by family.

GRPO grouping is unaffected: advantage is normalized **within** a group of `n_samples_per_prompt`
repeats of the *same* prompt (one family, one image); mixing happens **across** groups in a step, so
the families' 0/1 reward scales never need reconciling.

## Remote-only (no local sandbox)

`SANDBOX=local` is intentionally rejected: a CPU training run must never silently fall back off the
AgentCore substrate it is meant to exercise. Use `agentcore-runtime` (default; the pool of custom
Runtimes) or `agentcore` (Code-Interpreter microVM — but that has only the generic image, so it does
**not** exercise the pool).

---

## Reproduce (run one by one, inside the `agentcore_p4` container)

```bash
# 0. Build + deploy the THREE sandbox images (one Runtime per image). Build context = this demo dir.
for img in datasci scipy sklearn; do
  docker build -f code/demos/demo_cpu_imagepool/deploy/Dockerfile.sandbox.$img \
      -t <ecr>/acs-sandbox-$img:latest code/demos/demo_cpu_imagepool/
done
# push each to ECR and register a Runtime per image (CreateAgentRuntime, or your
# deploy_two_runtime.py --sandbox-only --dockerfile deploy/Dockerfile.sandbox.<img>), then collect
# the three ARNs into the image->ARN map:
export AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE='{"datasci":"arn:aws:bedrock-agentcore:us-west-2:ACCT:runtime/acs_sandbox_datasci-XXXX","scipy":"arn:aws:bedrock-agentcore:us-west-2:ACCT:runtime/acs_sandbox_scipy-XXXX","sklearn":"arn:aws:bedrock-agentcore:us-west-2:ACCT:runtime/acs_sandbox_sklearn-XXXX"}'

# 1. Build the MIXED dataset (offline, stdlib-only — no network, no GPU). Interleaves the families.
docker exec agentcore_p4 bash /work/code/demos/demo_cpu_imagepool/prepare_data.sh
#    -> data/mixed_train.jsonl (datasci+scipy+sklearn, balanced) + data/mixed_eval.jsonl

# 2. REAL RL train + eval through the REMOTE image pool (async/decoupled slime). Smoke = NR=3.
#    Requires AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE (step 0) + AWS creds in the container.
docker exec -e NR=3 -e EVAL_INTERVAL=2 \
    -e AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE="$AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE" agentcore_p4 \
    bash -lc 'cd /work && bash code/demos/demo_cpu_imagepool/run.sh 2>&1 | tee code/demos/demo_cpu_imagepool/smoke_run.log'

# 3. Parse the log into CSVs + a PNG of the curves.
docker exec agentcore_p4 python3 /work/code/demos/demo_cpu_imagepool/metrics.py \
    /work/code/demos/demo_cpu_imagepool/smoke_run.log \
    --csv train.csv --eval-csv train.eval.csv --plot train.png
```

The harness extracts the agent's submission from a fenced ` ```python ` block in each completion and
writes it into the sandbox session — no Anthropic `tool_use` parsing, so no special SGLang tool-call
parser is required here.

## Change the mix

```bash
# any subset of the custom-image families: datasci scipy sklearn
# (add generic-CPU families too, e.g. `math`, to show cpu + pool coexisting — but then also set a
#  cpu ARN via AGENTCORE_SANDBOX_RUNTIME_ARNS so the "cpu"-tagged episodes have somewhere to land)
FAMILIES=datasci,scipy,sklearn BALANCED=1 REPEAT=6 \
    docker exec agentcore_p4 bash /work/code/demos/demo_cpu_imagepool/prepare_data.sh
```

- `BALANCED=1` (default) round-robin interleaves families so **every step spans them evenly**;
  `BALANCED=0` does a plain shuffle (proportional to each family's seed count).
- `REPEAT=N` replicates the mixed train pool so a rollout step's batch stays varied across the run.
- **Save-only-at-end:** the **only checkpoint written is the final one**. `run.sh` passes an explicit
  `--num-rollout` (so slime leaves `num_rollout_per_epoch = None` and its epoch-boundary save never
  fires) plus a huge `SAVE_INTERVAL`; the sole save is slime's forced final-rollout checkpoint.

## Knobs (env vars; defaults in `run.sh`)

| Knob | Default | Meaning |
|---|---|---|
| `SANDBOX` | `agentcore-runtime` | remote sandbox: `agentcore-runtime` \| `agentcore`. **`local` is rejected.** |
| `AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE` | *(required)* | JSON `{image → ARN}` — the pool router (datasci/scipy/sklearn) |
| `AGENTCORE_SANDBOX_RUNTIME_ARNS` | *(empty)* | round-robin fallback pool for unmapped (`cpu`) tags |
| `RB` / `NS` | 4 / 8 | GRPO prompts-per-step / group size (`GBS = RB×NS`) |
| `NR` | 11 | num-rollout ⇒ real steps `NR-1`; keep the train set ≥ `NR×RB` |
| `CONC` | 4 | sandbox/episode concurrency; episodes in flight `S = CONC × num_engines` |
| `MODE` | `B` | `B` fully-async (deep buffer) \| `A` 1-step-async |
| `TEMP` | 1.0 | rollout sampling temperature |
| `EVAL_INTERVAL` / `EVAL_NS` | 10 / 2 | held-out (mixed) eval cadence / samples |

## What to watch

`rollout/raw_reward` (overall learning signal across the mix) and `eval/mixed` (held-out solve-rate).
Because each family lives on its own image, the **per-family eval files** (`data/<family>_eval.jsonl`)
are the sharp diagnostic: if one family sits at reward 0 while the others learn, its image ARN is
missing/mis-mapped in `AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE` (its `import` is failing in the wrong
sandbox). Also watch the `perf/sandbox_ms_*` vs `perf/sglang_gen_ms_*` split (`metrics.py` writes
per-op latency CSVs).

## Operating notes (AgentCore RL sandbox)

These are cross-cutting concerns this demo touches. They apply to any AgentCore-backed RL run; the
canonical implementation of each lives in the demo/report noted.

### Working around the ~2 GB image cap

The managed serverless microVM enforces two properties on the *built* image: it must be **arm64**
and stay **under ~2 GB** (measured). A pool of scientific images bumps into this — `scipy + numpy`
and `scikit-learn` (which pulls its own numpy/scipy) are heavy wheels, and a naive `pip install`
can approach or blow the cap. Five levers, cheapest first (report §"…arm64 + 2 GB limits" and
§"One image per Runtime…"):

1. **Defer the heavy install to `CMD`.** Keep the built image lean and `pip install` the big wheel
   when the container boots, before the server binds — the exact trick the GPU sandbox uses for the
   ~2.7 GB CUDA torch wheel (`deploy/Dockerfile.sandbox.gpu`, `SANDBOX_PYLIB`). Cost: added
   cold-start on the first session.
2. **Install at *seed* time** into the per-session ephemeral workspace (turn 0 `pip install`), so
   the bulk never lives in the image at all. Cost: per-episode latency + network egress.
3. **Slim the image:** multi-stage build, `--no-install-recommends`, drop build toolchains and pip
   caches after the wheel is in, start from a slim arm64 base. Often enough on its own.
4. **Split the capability across more ARNs** (this demo already routes by family) so no single image
   has to carry every library — one lean image per family instead of one fat image for all.
5. **Move to the amd64 Instances substrate**, which allows far larger images + a persistent EBS
   volume (a one-time install survives stop/resume). Cost: you spend a GPU-family instance for
   CPU-only work (there is no plain-CPU Instance family).

For this demo the practical answer is **#4 + #3**: one slim image per family (`datasci`/`scipy`/
`sklearn`), each installing only its own toolchain, keeps every image well under the cap.

### Keeping the sandbox warm across the weight-sync barrier

In async/decoupled slime, generation **pauses** while the trainer syncs updated weights into SGLang.
During that window no `invoke_agent_runtime` calls flow, and AgentCore reclaims a session purely on
**absence of traffic** (15-min microVM idle timeout / ~600 s GPU-instance idle timeout, reset on
each invoke). A long enough sync barrier — or a slow rollout tail — can therefore let an in-flight
sandbox be idle-reclaimed mid-episode, losing its workspace.

Two ways to hold a session warm through the barrier:

- **Heartbeat the health endpoint.** The sandbox server answers `GET /ping` (and the runtime client
  has a cheap `op: "ping"` no-op used at `__aenter__`); firing a `/ping` — or any no-op `exec` — at
  **half the idle timeout** resets the idle clock without doing real work. The canonical
  implementation is **`demo_cpu_heartbeat`**: a `SessionKeeper` (a process-wide registry + one
  background heartbeat loop that lives **outside** the harness) plus a `KeptSandbox` wrapper that
  gates new invokes for the sync window and pings any session going stale; outside the window it
  fires **zero** heartbeats (real traffic refreshes the same clock).
- **Don't tear the session down.** Each episode here mints a fresh `sbx-<uuid>` and calls
  `stop_runtime_session` at exit; set **`AGENTCORE_STOP_SESSION=0`** to keep sessions warm for reuse
  by the next episode instead of reclaiming them immediately (`agentcore_runtime_sandbox.py`).

Heartbeat backstops idle reclaim, not an instance *recycle* (a GPU-instance recycle kills the
session regardless) — for that you still need episode-level recovery (`recovery.resilient_episode`).

### Application snapshot at the weight-sync barrier (microVM & GPU instance)

Keeping a session *warm* (above) only defends the **current** session against idle reclaim. It does
nothing when the session is genuinely gone — a microVM idle-reclaimed after a long stall, or a
GPU-instance **recycle** (capacity reclaim / hardware replacement). To survive that you need to
**snapshot the sandbox's application state** (the per-session workspace: files written, corpus
`docs/`, any seed-time installs, cwd) and **restore** it into a fresh session. Restore is what makes
a long multi-turn episode — or a resumed RL run — pick up where it left off instead of starting the
env from scratch.

**Why piggyback on the weight-sync barrier.** In async/decoupled slime, generation **pauses** each
step while the trainer syncs updated weights into SGLang (the same barrier the keep-warm heartbeat
covers). During that window **no `invoke_agent_runtime` is mutating any sandbox** — every session is
quiescent. That makes the barrier the one moment a snapshot is guaranteed **consistent**: no torn
`write_file`, no half-finished `exec`. So the pattern is: at the barrier (or every *N*-th barrier, to
amortize cost), quiesce → snapshot each live session → resume. Sharing the snapshot cadence with the
trainer's own checkpoint keeps **policy weights and sandbox state consistent on resume** (both
rewind to the same step).

The mechanics differ by substrate because their persistence differs:

- **microVM (managed serverless) — snapshot must be APP-LEVEL.** The microVM has **no exposed
  disk/VM snapshot** and no persistent volume; it is ephemeral and reclaimed on idle. So "snapshot"
  means: at the barrier, `exec` a `tar` of the workspace dir (+ a small manifest of any seed-time
  `pip install`s) and `read_file` it **off the box** (S3 / local store); on the next session, at
  seed time `write_file` the tar back, unpack, and re-install the manifest. Cheap here precisely
  because at the barrier the workspace is small and static. This is the **only** durable option on
  the microVM and it is backend-portable.
- **GPU instance (EC2 capacity provider) — two levels.** (a) The **app-level** snapshot above works
  unchanged and is the portable, durable backstop. (b) The instance also has a **persistent EBS
  root**, so a plain **stop/resume keeps the workspace on disk** — no serialization needed across a
  pause. But a **recycle** still destroys that instance, so EBS persistence is *not* a substitute for
  the app-level snapshot; it only saves you the tar round-trip across a graceful stop. For a fast
  whole-disk restore you can additionally take an **EBS volume snapshot** at the barrier and boot the
  replacement instance from it (heavier, but no per-file replay).

The canonical implementation is **`demo_gpu_snapshot`** (app-level snapshot/resume across a
GPU-instance recycle). It composes with the two session-lifetime controls above: the keep-warm
heartbeat holds a session alive *within* a run, and `AGENTCORE_STOP_SESSION` governs teardown; the
snapshot is what lets you **rebuild** a session's workspace after the session itself is gone.

### One prompt, many samples (GRPO grouping)

GRPO forms a **group of `NS = n_samples_per_prompt` rollouts of the *same* prompt** and normalizes
advantage **within that group**. This "one prompt, many samples" shape shows up two ways here, both
handled without special-casing:

- **Sandbox side — `NS` isolated sessions per prompt.** The `NS` samples of one prompt are `NS`
  distinct `runtimeSessionId`s ⇒ `NS` separate microVMs (booting the family's image), so no sample
  can see another sample's files even though they share a prompt. They stream through the concurrency
  budget `S = CONC × num_engines` like any other episodes; a group is *not* required to run all at
  once.
- **Loss/advantage side — make it insensitive to sample count and length.** Because advantage is
  normalized *within* the same-prompt group, the very different reward *shapes* across families
  (pandas vs scipy vs sklearn, all 0/1 here) never need reconciling — within-group normalization is
  scale-free. On the loss, use a **rollout-level token-mean** (slime's default): pool all response
  tokens of a rollout, then average uniformly over rollouts, so the update is insensitive to *how
  many* samples a rollout produced (and robust when samples differ wildly in length). This is the
  same reconciliation `demo_cpu_branching` exercises when one rollout emits a variable number of
  sub-samples (report §"Loss normalization at the rollout level").

### Splitting a single GPU into fractions to serve concurrent sessions

This is a **GPU-substrate** concern (not exercised by this CPU demo, but the sibling GPU demos do
it — see `demo_gpu_farm`). The constraint: **one AgentCore session = one whole
instance** (session→instance is 1:1), so you **cannot** point two `runtimeSessionId`s at the same
box and get one GPU each — and one `runtimeSessionId` = one GPU means **no native cross-session
time-slicing**. To pack several concurrent episodes onto one GPU you must run **one shared session**
plus a **self-managed in-instance farm**: a Ray farm inside the sandbox Runtime, keyed by a
per-episode `sub_session`, that hands each episode a **fraction** of the GPU.

- `B4_SHARED_RUNTIME_SESSION` pins every episode to the same instance (one shared session id).
- `B4_FARM_GPUS=N` sets how many physical GPUs the in-instance farm manages.
- `B4_GPU_FRACTION=f` leases a fraction `f` of a GPU per `sub_session` ⇒ `≈ N/f` concurrent
  episodes (`f=1.0` ⇒ one whole GPU per sub_session, N concurrent; `f=0.25` ⇒ 4 per GPU).

Caveat from hardware: **non-MIG** GPUs (e.g. L4 / L40S) have no hardware partitioning, so fractions
are soft (MPS-style oversubscription); under sustained heavy load a non-MIG GPU can wedge, and a
smaller `B4_GPU_FRACTION` mitigates it. Use real **MIG** where the GPU supports it for hard isolation.

#### One whole GPU per session → 8 concurrent on an 8-GPU instance

The cleanest, isolation-safest case of the above is **one whole GPU per episode**, no fractioning:
lease an **8-GPU** instance (e.g. `g6.48xlarge` / a `g6e` 8-GPU box) and run **8 concurrent
episodes, one per GPU**. Set:

```bash
B4_FARM_GPUS=8       # the instance has 8 physical GPUs
B4_GPU_FRACTION=1.0  # one WHOLE GPU per sub_session (no soft-partitioning) => 8 concurrent
```

Mind the cardinality: at the **AgentCore** level this is still **one Runtime session = one
instance**, because a session maps 1:1 to an instance and you cannot pin two `runtimeSessionId`s to
one box. The "8 concurrent sessions" are **8 per-episode `sub_session` slots** that the in-instance
Ray farm hands out, each holding one whole GPU — *not* 8 AgentCore sessions. So you get 8-way GPU
concurrency from **one** shared `runtimeSessionId` (`B4_SHARED_RUNTIME_SESSION`), which is exactly
`demo_gpu_farm` at `B4_FARM_GPUS=8, B4_GPU_FRACTION=1.0`. Because each episode
owns a *whole* GPU, there is no time-slicing and no MIG needed — the non-MIG wedge caveat above does
not apply. To instead have AgentCore place each session on its *own* instance (true 1 session : 1
instance : 1 GPU), mint distinct `runtimeSessionId`s against a single-GPU-instance Runtime and let
the capacity provider spread them — you scale GPUs by adding *sessions/instances*, at the cost of
one instance per GPU.

## Folder layout (self-contained)

- `agentcore_slime_demo/` — vendored library (in-process harness `generate.py` + `mini_agent.py`,
  the remote sandbox client `agentcore_runtime_sandbox.py`, the **image-aware router**
  `sandbox._runtime_arn_for_image`, the task registry `task_families.py` incl. `DataSciTask` /
  `ScipyTask` / `SklearnTask`, the sandbox server `sandbox_server_agent.py`). Imported only from here.
- `build_mixed_datasets.py` — builds the interleaved multi-family dataset (offline, stdlib).
- `prepare_data.sh` — step 1 · `run.sh` — the remote RL run (step 2) · `metrics.py` — step 3.
- `deploy/Dockerfile.sandbox.{datasci,scipy,sklearn}` — the **three** custom sandbox images in the
  pool (each = the generic exec server + one baked toolchain). `Dockerfile.sandbox.cpu` is the
  generic image, kept for any stdlib family you add to the mix.
- `data/` — `mixed_train.jsonl` / `mixed_eval.jsonl` (+ per-family `<family>_eval.jsonl`).
