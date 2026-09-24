# Terminal-Bench on AgentCore

**Kaige Yang** — Senior Applied Scientist, AWS

September 23, 2026

---

## Outline

1. **Basic concepts:** image, Runtime, session, microVM, and how they relate
2. **The dataset:** Terminal-Bench rows, images, sizes, and why each row has its own image
3. **Training-time wiring:** how the Runtime is invoked, how sessions launch, how many exist, and how episodes are isolated
4. **Cold start:** the problem, the options, the best one, and what the demo uses
5. **The demo:** components, step-by-step flow, and the metrics we collect (what each one means and why we care)

Source: `code/demos/demo_cpu_terminalbench/` (deploy, pool, grader, probe, and `run.sh`).

# 1 · Basic concepts

## Image → Runtime → session → microVM

| Concept | What it is | Lifetime | Count in this demo |
|------------|----------------------------------------------------------|------------------------------------------|--------------|
| **Image** | arm64 container image in ECR: the task's own environment plus a thin wrapper | Built once | One per task |
| **Runtime** | An AgentCore *agent runtime*: a registered, versioned endpoint pinned to one image digest (ARN) | Long-lived; costs nothing when idle | One per image |
| **Session** | A `runtimeSessionId` on a Runtime; invokes with the same id reach the same sandbox | First invoke → `StopRuntimeSession` or ≈ 15 min idle | One per **episode** |
| **microVM** | A Firecracker VM (2 vCPU, 8 GB) that runs the image's container for exactly one session | Same as its session | One per session |

: Image, Runtime, session and microVM: definitions, lifetimes and counts

## How they relate

![](figures/tb_fig01_cardinality.pdf){height=46%}

*Figure 1: Cardinality: image 1:1 Runtime 1:N session 1:1 microVM.*

- **Session = microVM, 1:1**, terminated and memory-sanitized on stop. **The session is the isolation boundary.**
- **One image per Runtime**, so many environments need many Runtimes (quota: 1000 per account, adjustable) [Q1].
- **Serverless limits:** arm64 only · image ≤ 2 GB (fixed) · ≤ 25 new sessions/s per account · ≤ 15 min per invoke · 15 min idle → reclaimed [Q1].

## What the wrapper adds to each task image

Terminal-Bench ships a bespoke Dockerfile per task. We **leave the task environment untouched** and append one layer:

```
USER root
COPY --from=acs py /opt/acs/py            # standalone CPython 3.12 (54 MB), not the task's
COPY --from=acs acs_http_server.py sandbox_server_agent.py /opt/acs/
ENTRYPOINT []
CMD ["/opt/acs/py/bin/python3", "/opt/acs/acs_http_server.py"]   # :8080 /ping + /invocations
```

*Figure 2: The wrapper layer appended to every task Dockerfile.*

- **Why a standalone Python?** Task images range from python3.8 to having no Python at all. Running `pip install` in them would fail on some, and it would also **change the environment being graded**.
- **The server is stdlib-only.** It handles `exec`, `write_file`, `read_file` and `put_tar`. It reports the image's own WORKDIR (`/app`, `/app/astropy`, `/home/alice`, ...) and **boot timestamps** that let us decompose cold start.
- **The hidden tests are *not* in the image.** The grader injects them only after the agent finishes, as Terminal-Bench's own harness does, so the root agent cannot read them while solving.

# 2 · The dataset

## Terminal-Bench: 229 tasks = 229 images

| | Count |
|----------------------------------------------------------------------|----------------------------------------------------------------------|
| Tasks (dataset rows) | **229** (easy 59 · medium 113 · hard 57) |
| Distinct environment images needed | **229** (one per task; no two tasks share one) |
| Built for arm64 and ≤ 2048 MB → deployable | **218** |
| Exceed the 2048 MB cap (mteb-×3 at 6.6 GB, png-generation at 6.5 GB) | 4 |
| Don't build for arm64 (amd64-only base, qemu-system, x86 wheels, slurm) | 7 |
| Top categories | software-eng 46 · sysadmin 25 · data-sci 19 · sci-computing 18 · security 18 · debugging 17 |

: Terminal-Bench task and image counts

Each row is a natural-language instruction plus its own Docker environment, a hidden pytest suite (`tests/`, `run-tests.sh`) and a reference `solution.sh`. **The reward is 1 only if every test passes.**

**Train/eval split:** 80/20, disjoint by task and stratified by difficulty. Training rows are further filtered to **oracle-valid** tasks (see Section 5): **166 usable → 135 train + 31 held-out eval**.

## Image sizes (compressed, as pulled from ECR)

| Size bucket | Images |
|-----------------------|------|
| < 250 MB | **156** |
| 250 – 500 MB | 45 |
| 500 MB – 1 GB | 10 |
| 1 – 2 GB | 7 |
| > 2 GB (not deployable) | 4 |

: Image size distribution (compressed)

- Percentiles: min 61 MB · p25 74 MB · **p50 128 MB** · p75 302 MB · p90 492 MB · max 6.7 GB, over all 222 built images (74 GB). The 218 deployable images total **48 GB**.
- The heavy tail (ML models, conda, TeX, compilers) is what makes cold start vary by an order of magnitude.
- **Build cost:** 229 buildx arm64 builds under QEMU on a 192-core x86 host, 64 in parallel, took about 2 h wall-clock. Most finish in 1–3 min; C and C++ toolchains under emulation are the tail (`custom-memory-heap-crash`, a GCC build, alone took 2 h). The build is resumable: a wrapper-hash tag skips images that are already pushed.

## Why one image per dataset row?

- **The environment *is* the task.** "Fix this broken conda env", "recover this corrupted SQLite db" and "harden this nginx config" each need a specific OS, packages, files and even a broken state to start from. That state can't be recreated with a `pip install` at episode start.
- **Graded against its own environment.** Tests check the real system state (files, services, binaries). A shared "fat" image would have package conflicts between tasks, and some tasks deliberately *break* the base environment.
- **Setup is too heavy to do at runtime.** Tasks bundle compiled toolchains, models and datasets, up to GBs each. Doing that per episode would cost minutes, which is far worse than pulling an image.
- **Faithful to the benchmark.** Terminal-Bench scores are only comparable if each task runs in its official environment.

⇒ **One image → one Runtime → routed by task id.** The pool map `AGENTCORE_SANDBOX_RUNTIME_ARN_BY_IMAGE = {task → ARN}` sends each episode to its own task's Runtime. A task missing from the map is an **error**; it is never sent to some other task's Runtime.

# 3 · Training-time wiring

## How a Runtime is invoked and a session launched

![](figures/tb_fig03_episode-workflow.pdf){height=58%}

*Figure 3: Episode flow: routing, session launch, agent loop, grading, stop.*

- **There is no "create session" API call.** The first invoke carrying a new `runtimeSessionId` *is* the launch.
- All calls go through the IAM control plane with `networkMode=PUBLIC`: the sandbox needs internet egress (`run-tests.sh` installs uv and pytest), and it never dials back to the trainer.

## How many sessions and microVMs?

Default smoke config: `RB=4` prompts × `NS=8` GRPO samples gives **32 episodes per rollout step**.

| Quantity | Value | Why |
|----------------------------------|--------------------------|----------------------------------------------------------------------|
| Sessions per step | **32** = RB × NS | One fresh session per episode, never reused |
| microVMs per step | **32** | Session = microVM, 1:1 |
| Running concurrently | `HARNESS_WORKERS` (4) | Warm in-process agent workers |
| Pre-opened, booting ahead | ≤ P = 64 | The next **2 batches** (`TB_PREOPEN_BATCHES` × RB × NS), announced by the look-ahead data source |
| Peak live microVMs | ≈ 68 (W + P) | Bounded, so idle VMs are bounded |
| Distinct Runtimes touched per step | ≤ RB (4) | The NS siblings of a prompt share the task's Runtime but get **different** sessions |

: Sessions and microVMs per training step (smoke config)

**Scaling:** raise `HARNESS_WORKERS` for throughput; set `TB_PREOPEN_BATCHES` for how far ahead to boot. The ceilings are the account session quota (5000 active sessions [Q2]) and 25 new sessions per second (throttled opens are retried with backoff). A full run over 218 Runtimes stays well under the 1000-Runtime quota.

## Example: 4 prompts × 8 samples, shared tasks or not

**Sessions per step = RB × NS = 32, always.** Only the number of Runtimes used changes: a Runtime belongs to a task's image, not to a prompt.

| Batch (4 prompts) | Runtimes used | Sessions per Runtime | Sessions total |
|--------------------------------|---------------|------------------------------|---------------|
| A, B, C, D (all different) | 4 | A 8 · B 8 · C 8 · D 8 | **32** |
| A, A, B, C | 3 | A **16** · B 8 · C 8 | **32** |
| A, A, B, B | 2 | A 16 · B 16 | **32** |
| A, A, A, A | 1 | A **32** | **32** |

: Runtimes (= images) and sessions per step, shared tasks or not

- **Shared task → fewer Runtimes, never fewer sessions.** Each episode still gets its own fresh microVM.
- **No Runtime is created in training.** All 218 exist from deploy; unused ones cost nothing.
- **No per-Runtime session limit.** The limits are account-wide (concurrent sessions, 25 new/s). Live microVMs stay ≤ W + P = 68 in every case.
- *Sharing is rare here:* one row per task, so a batch repeats a task only across an epoch boundary.

## Who knows what: slime vs the pool vs AgentCore

**slime knows none of it.** It only knows prompts (dataset rows) and calls one `generate()` hook per sample. Tasks, images, Runtimes, sessions and the queue all live in *our* `TerminalBenchPool`, inside slime's rollout process.

| Layer | Knows | Does not know |
|-------------------------|-----------------------------------------------------|------------------------------------|
| **slime** | Rows of `tb_train.jsonl`; RB × NS; how many `generate()` calls may be in flight (`CONC × engines` = 8) | Which image or Runtime a row needs; that sessions exist |
| **Dataset row** | `metadata.task.spec.image` = the task name (plus prompt, timeouts) | ARNs (kept out of the data, so a redeploy needs no rebuild) |
| **`LookaheadDataSource`** (our slime data source) | slime's prompt order: dataset seed, epoch, offset → the prompts of the **next 2 rollouts**, before slime draws them | ARNs or sessions (it hands the list to the pool) |
| **`TerminalBenchPool`** (our generate hook) | Pool map `{task → ARN}` from env `..._ARN_BY_IMAGE` (set by `run.sh`); the ordered list of upcoming episodes; ready sessions per task; heartbeat set | Nothing about the model or GRPO |
| **AgentCore** | Each Runtime's image; which `runtimeSessionId`s are live microVMs | How many episodes are coming. No session count is ever declared |

: Who knows what, by layer

## How a queued episode finds its image, Runtime and session

1. **slime draws a batch** through our `LookaheadDataSource`. It also **peeks** the next 2 rollouts' prompts (replaying slime's seeded shuffle, without moving slime's position) and calls `pool.reserve(drawn 32, upcoming 64)`.
2. **Routing:** each upcoming episode's task (`spec.image` = task name) is looked up as `ARN = pool_map[spec.image]`. An unmapped task fails right away.
3. **Pre-open:** `_refill()` boots a fresh session (a `ping` with a new id `tbep-<32 hex>`) for each of the first **P** = `TB_PREOPEN_BATCHES` × RB × NS (64) upcoming episodes, in order, kept per task in `_ready[ARN]`.
4. **slime calls** `generate(sample)` for the batch, at most `CONC × engines` (8) in flight; our hook calls `pool.submit(task)`.
5. **Pickup:** a free harness worker (one of `HARNESS_WORKERS`) `_claim`s **any** ready session for its task. If none is ready, it opens one then. The claim moves the window on by one episode, so the next upcoming episode starts booting.
6. **End:** grade, then `StopRuntimeSession`, and slime gets the finished sample back.

**So the number of sessions per Runtime is never configured.** It emerges from the data. A task sampled in a step gets **NS** sessions on its Runtime (one per GRPO sibling), and a task not sampled gets 0. Live sessions are bounded by workers and the pre-open window (`TB_PREOPEN_BATCHES` × RB × NS), not by any per-Runtime setting.

## Terminal-Bench look-ahead pre-open (1/2): status and why

**Status.** The look-ahead has now run in real training on AWS (first smoke run, 2026-09-23): all 382 episodes got a pre-opened session and `open_wait_ms` was 0 at p95 (Results slide). With the defaults (4 prompts × 8 samples, 2 batches ahead), the pool keeps **64 microVMs booted ahead** of the episodes that are running.

**How the pool learns the next two batches' episodes.** slime would never tell it in time: it passes along only **8 episodes** of the current batch at once and doesn't draw the next batch until the current one finishes. But slime's prompt order is **predictable**: it walks a shuffled dataset in order, with a fixed seed for each epoch. So a data source can compute what slime will draw next.

## Terminal-Bench look-ahead pre-open (2/2): the five steps

1. A new **look-ahead data source** (`agentcore_slime_demo/tb_lookahead_source.py`, wired in `run.sh` with `--data-source-path`) wraps slime's normal one. Each time slime draws a batch, it also works out the prompts of the next 2 batches the same way slime will draw them, without moving slime's position.
2. It sends the pool the task list for each upcoming episode: the **32** just drawn, plus the next **64**.
3. The pool keeps the first **64** not-yet-started episodes booted, each on its own brand-new session on that task's Runtime. Sessions are matched by task, and each one still serves exactly one episode.
4. When a worker starts an episode, it takes a ready session for that task and the next episode in line starts booting. So 64 always stay booted ahead, which covers batch **N+2** before slime draws it. The existing heartbeat keeps waiting sessions alive.
5. If a prediction is wrong, the extra sessions are stopped at the next draw and those episodes just open on demand.

## Pre-opening sessions for upcoming episodes

**Runtimes are never pre-opened**: all 218 were created at deploy and stay `READY` at no idle cost. What is pre-opened is the **session**, which is the **microVM**.

| Episode state | Where in the pool | What happens |
|------------------|---------------------------------|------------------------------------------------------------|
| **upcoming** | `_demand` (drawn) / `_future` (peeked next 2 rollouts), as ARNs in run order | Announced by `LookaheadDataSource`; beyond the first 64, no session yet |
| **pre-opening** | `_ready[ARN]` (asyncio task) | New id `tbep-<hex>` → `InvokeAgentRuntime` ping → AgentCore allocates, pulls, boots. A failure is retried on a new id (3×) |
| **booted, idle** | same task, done | `SessionKeeper` runs a no-op exec every ~7.5 min (< 15-min idle timeout) |
| **claimed** | popped by a worker (any ready one for its task) | The worker awaits it; any boot left is `open_wait_ms` |
| **closed** | removed | Graded, then `StopRuntimeSession` terminates the microVM |

: Episode states inside TerminalBenchPool

**`_refill()`** (on every draw, `submit` and claim): keep one ready session per episode in the first P = 64 upcoming ones, pooled per task, and **stop** any ready session that fell out of the window (a wrong guess). Live microVMs ≤ W + P = 68.

## Worked example (1/2): one step with the next two booting

Step N: RB = 4 prompts (tasks **A** `broken-python`, **B** `chess-best-move`, **C** `build-pmars`, **D** `cancel-async-tasks`) × NS = 8 = 32 episodes. Steps N+1 (tasks E–H) and N+2 (I–L) follow. W = 4, P = 64, in-flight cap 8.

| # | Event | Workers run | Pre-opened (booted or booting) | Live microVMs |
|---|---------------------------------------------------------|-----------------|--------------|------------------|
| 0 | *While step N−1 ran*, its draw already announced steps N and N+1, so A–D and E–H were booted | N−1's episodes | A1–D8, E1–H8 | ≤ 68 |
| 1 | slime draws step N (A–D) and peeks N+1, N+2. `reserve` → window = A–D + E–H: already booted, so nothing new to open | – | 64 (A–H) | 64 |
| 2 | slime admits 8 `generate()` calls (A1–A8). Workers 0–3 `_claim` booted A sessions: `open_wait_ms ≈ 0` | A1–A4 | 60 | 64 |
| 3 | Each claim moves the window: the next upcoming episodes (I1, I2, …) of step N+2 start booting | A1–A4 | 64 (A5–H8 + I1–I4) | 68 |
| 4 | A2 ends: grade → `StopRuntimeSession` → slime admits B1; worker takes a booted A5; I5 starts booting | 4 | 64 | ≤ 68 |

: Worked example, step N with look-ahead (W = 4, P = 64)

## Worked example (2/2): the step boundary

| # | Event | Workers run | Pre-opened | Live microVMs |
|---|---------------------------------------------------------|-----------------|--------------|------------------|
| 5 | Steady state: each finish stops one VM, a worker claims a *booted* session, one more of step N+2 boots | 4 | 64 | ≤ 68 |
| 6 | Last samples (D5–D8) finish. Window = E–H (booted) + I1–J8 (booted by the claims) | ≤ 4 | 64 | falling to 64 |
| 7 | slime trains on 4 groups of 8; the 64 ready VMs idle through training (heartbeat keeps them) | – | 64 | 64 |
| 8 | Step N+1 draws E–H: **all already booted**, so its first episodes start with `open_wait_ms ≈ 0` | 4 | 64 | ≤ 68 |

: Worked example, across the step boundary (W = 4, P = 64)

**Tally for the step:** Runtimes A, B, C and D each served **8 sessions** (8 microVMs, one per GRPO sibling). The other Runtimes of the step served 0. Peak live microVMs: **68**. Nobody configured "8 per Runtime": it falls out of NS.

**Why it works:** every episode, including the first of each step, is booted a full batch or more before a worker needs it. The price is idle microVM time (next slides).

## How to set `TB_PREOPEN_BATCHES` (1/3): the sizing rule

W = `HARNESS_WORKERS`, P = `TB_PREOPEN_BATCHES` × RB × NS (the only pre-open knob; P follows from it), B = cold start (`open_ms`), E = episode time (agent + grade).

- In steady state a worker frees up every **E / W** seconds. An episode that starts booting is picked up after the **P** episodes ahead of it, so it has about **P · E / W** seconds to boot.
- **Hidden boot needs P · E / W ≥ B, i.e. P ≥ ⌈W · B / E⌉**, plus 1–2 for jitter. Across a step boundary the window must also cover the first W episodes of the next step, so **`TB_PREOPEN_BATCHES` ≥ 1** (P ≥ RB × NS) removes the per-step boot tail.
- **Each booted VM then idles for about P · E / W − B** (plus training time at a step boundary). That is paid microVM time; the heartbeat keeps it alive past the 15-min idle timeout, but it still costs money.

## How to set `TB_PREOPEN_BATCHES` (2/3): choosing a value

| `TB_PREOPEN_BATCHES` | P | Boot hidden? | Idle per VM (E = 120 s, W = 4) |
|------------------------------|------|------------------------------------|----------------------|
| 0 — look-ahead off (queue only, ≈ W) | 4 | Mid-step only; each step's first W pay a boot | ≈ 2 min − B |
| 1 — one batch ahead | 32 | Yes, including step starts | ≈ 16 min − B |
| **2 — default, two batches ahead** | **64** | Yes, with a full batch of slack | ≈ 32 min − B |

: Choosing TB_PREOPEN_BATCHES (illustrative E, until `coldstart.csv` is in)

*`TB_PREOPEN_BATCHES` is the one pre-open knob. `0` turns the look-ahead off: only episodes slime has already submitted are pre-opened (about W).*

**Measured in the smoke run** (P = 64): E ≈ 21 s, B ≈ 0.7 s (warm hit), booted VMs idle **7.3 min** p50. So `TB_PREOPEN_BATCHES` = 1 is the next setting to try.

## How to set `TB_PREOPEN_BATCHES` (3/3): tuning

- **`CONC` no longer has to follow:** the look-ahead comes from the data source, so `run.sh` sets `CONC = ⌈2W / engines⌉` (W running + W queued).
- **One draw per rollout assumed** (no dynamic filtering or over-sampling). A wrong guess only costs VMs stopped at the next draw.
- **Ceilings:** live microVMs ≤ W + P, against a quota of **5000** active sessions [Q2]; the first draw opens P at once, so keep P within 25 new sessions/s.

| Symptom (from `[demo]` lines) | Meaning | Action |
|---------------------------------------------------|------------------------------|-------------------------|
| `open_wait_ms` p50 ≫ 0 on `preopened=1` | Boots not finished at pickup | Raise `TB_PREOPEN_BATCHES` |
| `preopened=0` on many train episodes | Look-ahead guessed wrong, or is off | Check the `[tb_lookahead]` log lines |
| `preopen_idle_ms` ≫ B, often many minutes | VMs booted far too early | Lower `TB_PREOPEN_BATCHES` to 1 (saves microVM-seconds) |
| `open_wait_ms ≈ 0` and `preopen_idle_ms` small | Balanced | Keep |

: Tuning TB_PREOPEN_BATCHES from the collected metrics

## How far ahead can we pre-open?

**As far as the look-ahead reaches: the next `TB_PREOPEN_BATCHES` rollouts (default 2).** The pool by itself sees only what slime has submitted, and slime draws step N+1 only after step N's rollout. So we added a data source that **peeks at slime's data order**.

- slime's order is deterministic: a sequential offset through the dataset, reshuffled each epoch with `random.seed(seed + epoch)`. `LookaheadDataSource` replays that to list the next rollouts' prompts, across epoch boundaries, without advancing slime.
- **Units:** one batch = 4 prompts × 8 = **32 episodes = 32 microVMs**, so two batches ahead = **64 booted, idle, billed microVMs**.

| Option | Idea | Catch |
|-------------------------|----------------------------------------------|---------------------------------------|
| 1. Fully-async rollout | slime streams samples continuously; no step boundary to drain | Slightly older weights (bounded staleness) |
| **2. Peek at data order (used)** | Pool boots the next rows' VMs before slime asks | Guesses can be wrong; VMs idle through training |
| 3. Accept the tail | Nothing | About one boot per step (a few %) |

: Ways to pre-open across a step boundary

## Option 1 vs option 2: what actually runs early

**Option 1 moves the whole episode earlier; option 2 moves only its microVM boot.**

| | **1. Fully-async rollout** | **2. Peek at data order** |
|--------------------------|-----------------------------------|-----------------------------------|
| Who picks the next episodes | slime, which **really submits** them | Our pool, which **guesses** |
| What runs early | The whole episode (agent + grade) | Only the session / microVM boot |
| Policy weights | May be 1+ updates old (off-policy) | Current (on-policy, unchanged) |
| Look-ahead can be wrong? | No, every episode started is used | Yes (dynamic sampling, filtering, restart) → wasted VMs |
| Idle microVM time | Low: the episode starts after boot | High: idle through the step and training |
| Code change | slime rollout mode (as in demo_a) | New pool logic reading slime's data source |
| Removes | The whole step barrier | Only the per-step boot tail |

: Option 1 vs option 2 for cross-step look-ahead

**What the demo does:** option 2, because it keeps training on-policy and needs no slime change (a `--data-source-path` plugin). Watch `preopen_idle_ms`: if the idle cost dominates, lower `TB_PREOPEN_BATCHES` or move to option 1.

## Isolation: how episodes can't affect each other

| Layer | Mechanism |
|-------------------------------------------------------------|----------------------------------------------------------------------|
| **Across tasks** | Different image → different Runtime → different microVM fleet. Nothing is shared. |
| **Across episodes of the same task** (GRPO siblings, later steps) | A new `runtimeSessionId` per episode gives a **new microVM** from the pristine image. No reset or cleanup scripts to trust. |
| **After the episode** | `StopRuntimeSession` right away: the VM is terminated, its memory is sanitized, and it never serves another episode. |
| **Against reward hacking** | Tests are injected after the agent's last turn, and the reference solution is never in the image. |
| **Trainer ↔ sandbox** | The sandbox is reached only through IAM-signed invokes. No inbound path to the trainer, and no credentials inside the VM. |

: Isolation mechanisms by layer

**Why not reuse a warm session and reset its workdir?** Terminal-Bench tasks act on the *whole OS* (apt installs, `/etc`, services, `/tmp`, kernel parameters). A workdir reset would leak that state into the next episode and corrupt both the reward and the GRPO advantage.

# 4 · Cold start

## The challenge: every episode pays a boot

A fresh session means allocating a microVM, pulling the image and starting the container before the first command runs.

| Phase (measured from server boot timestamps) | Typical |
|----------------------------------------------------------------------|---------------------------------------------|
| `pull_boot`: request sent → server process up (allocate, pull, start) | **≈ 15 s** for a 447 MB image (early measurement) |
| `ready`: server up → first invoke delivered (health check, routing) | ≈ 0.7 s |
| **Warm hit** (AgentCore had already pre-provisioned a VM for this Runtime) | **≈ 0.5 s** open |

: Cold-start phases

- Scale: 32 episodes per step × about 15 s is **8 minutes of serial boot per step** if it sits on the critical path.
- The cost grows with image size (a long tail up to 2 GB), and it happens *in bursts*: all NS siblings of a task start at once.
- Second failure mode, **idle reclaim:** a session idle ≈ 15 min is reclaimed, and the weight-sync pause can leave a pre-opened or mid-episode session silent that long.

Per-Runtime numbers: `probe_coldstart.py` → `tasks/coldstart.csv` (Table 11).

## Options for dealing with cold start

| Option | Hides boot? | Isolation | Cost | Verdict |
|----------------------------------------------------------|------------------|----------------|-------------------------------|---------------------------------|
| A. Reuse one warm session per task, reset between episodes | ✓ | ✗ OS state leaks | Low | **Rejected**: breaks the reward |
| B. Big standing warm pool of pre-opened sessions per task | ✓ | ✓ | ✗ 218 tasks × NS idle VMs | Wasteful at this scale |
| C. Open on pickup (naive) | ✗ | ✓ | Low | Boot is on the critical path |
| D. Shrink images (slim bases, lazy pull) | Partly | ✓ | Engineering per task | Changes the benchmark environment |
| E. **Pre-open the next batches (data-order look-ahead) + heartbeat** | ✓ (including step starts) | ✓ | ≤ P = 64 extra VMs | **Best, and used** |

: Cold-start mitigation options

Option E also benefits from AgentCore's own behaviour: a Runtime keeps a pre-provisioned spare microVM. The first session on it is a **warm hit** (about 0.5 s), and E hides the rest.

## What the demo does (Option E)

1. **Look ahead two batches.** `LookaheadDataSource` (`--data-source-path`) tells the pool the episodes of the current and next `TB_PREOPEN_BATCHES` (2) rollouts on every draw, by replaying slime's deterministic data order.
2. **Pre-open a bounded window.** The pool keeps a fresh session booted for each of the first **P** = 64 (2 × 4 × 8) upcoming episodes, pooled per task; a worker takes any ready one for its task. That caps the idle microVMs you pay for, and wrong guesses are stopped at the next draw.
3. **Heartbeat.** A process-wide `SessionKeeper` pings every open session (pre-opened or running) at half the idle timeout, so none is reclaimed during the weight-sync stall.
4. **Retry on a new id.** If an open fails, it is retried on a **new** session (`TB_OPEN_ATTEMPTS=3`). An infra failure is logged as `infra_error` and is **never** confused with a wrong answer.
5. **Measure every boot.** The start line of each episode records `open_ms`, `pull_boot_ms`, `warm_hit`, `ready_ms`, `open_wait_ms` and `preopen_idle_ms`. That shows how much boot was hidden and how long a booted VM sat idle before pickup, which is what the heartbeat must cover.

# 5 · The demo

## Components

| Component | File | Role |
|----------------|------------------------------------------|----------------------------------------------------------------------|
| Fetch | `fetch_terminalbench.py` | Downloads the 229 tasks and writes `tasks/manifest.json` |
| Build + deploy | `deploy/deploy_terminalbench.py` | Per task: wrapped Dockerfile → buildx arm64 → ECR → one Runtime pinned to the digest. Writes `arn_by_image.json` and `images.csv` (size, layers, build_s, status) |
| Sandbox server | `deploy/acs_ctx/` `acs_http_server.py` | Stdlib AgentCore HTTP server in every image: exec, files, `put_tar`, boot timestamps |
| Oracle check | `validate_oracle.py` | Per task: *untouched env must fail* and *reference solution must pass* → `oracle.csv` |
| Cold-start probe | `probe_coldstart.py` | First and burst opens on every Runtime → `coldstart.csv` (open, pull+boot, warm hit × size) |
| Dataset | `build_terminalbench_dataset.py` | Deployed ∩ oracle-usable tasks → `tb_train` / `tb_eval` (disjoint, stratified) |
| Look-ahead | `…/tb_lookahead_source.py` | slime data source: peeks the next 2 rollouts' prompts → `pool.reserve` |
| Rollout pool | `…/terminalbench_pool.py` | Routing, fresh session, pre-open window, heartbeat, agent loop, grading, metrics |
| Agent | `mini_agent.py` + AnthropicAdapter | Bash-tool loop against the policy under training (qwen25 tool parser) |
| Trainer | `run.sh` | slime async GRPO, Qwen3-4B, 4 actor + 4 rollout GPUs (Mode A) |

: Demo components

## Step by step

1. **Fetch** the tasks, then **build and deploy** the pool: 229 builds → 218 deployable Runtimes (one per task).
2. **Validate the oracle** on the real substrate. Keep a task only if the reference solution passes **and** the untouched environment fails. Otherwise the reward is structurally 0 (the task can't be learned here) or structurally 1 (no signal).
3. **Probe cold start** across all Runtimes to get the boot vs image-size distribution.
4. **Build the dataset** from deployed ∩ usable tasks: 80% train and 20% held-out eval.
5. **Train.** Each step samples RB tasks × NS rollouts. Every rollout gets a fresh session on its task's Runtime, booted up to two batches ahead, runs up to 8 agent turns (bash tool, 180 s per command), and is graded by the injected hidden tests. Reward is {0,1}, then GRPO updates the weights.
6. **Evaluate** on the held-out tasks every `EVAL_INTERVAL` steps.
7. **Report** from the `[demo] episode_end` lines: solve rate by difficulty and category, and the cold-start and cost breakdown.

## Metrics we collect (1/2): learning and agent behaviour

| Metric | Meaning | Why we care |
|-----------------------------------|----------------------------------------------------------------------|----------------------------------------------------------------------|
| `rollout/raw_reward` | Mean episode reward (fraction of train episodes that solved their task) | The primary signal that the policy is learning |
| `eval/terminalbench` | Solve rate on **held-out** tasks | Generalization, not memorizing the train tasks |
| Solve rate by `difficulty` / `category` | Reward grouped from `episode_end` | Where the gains come from; whether hard tasks move at all |
| Mixed-group fraction | Share of GRPO groups (NS siblings) with some solved **and** some failed | Only mixed groups produce a gradient. All-0 or all-1 groups are wasted compute |
| Distinct tasks ever solved | Coverage of the train set | Shows whether RL expands the frontier or just sharpens a few tasks |
| `turns`, `tool_calls`, `stop` | Turns used, bash calls made, and why the episode ended (done, max_turns, max_tokens) | Whether the agent acts or just talks, and whether the 8-turn budget binds |
| `tests_passed` / `tests_total` | Partial credit from pytest (logged, not rewarded) | Detects "almost solved" progress that binary reward hides |

: Metrics: learning and agent behaviour

## Metrics we collect (2/2): grading, infrastructure, cost

| Metric | Meaning | Why we care |
|---------------------------------------|----------------------------------------------------------------------|----------------------------------------------------------------------|
| `open_ms` | First invoke sent → reply: the whole cold start this episode's session paid | The raw cost that pre-open must hide |
| `pull_boot_ms` / `ready_ms` | Sent → server up (allocate, pull, start) / server up → first invoke | Separates **image size** effects from **platform** overhead |
| `warm_hit` | 1 if the server was already up before we asked (pre-provisioned spare) | How often AgentCore's own warm VM saves the boot |
| `open_wait_ms` | How long the worker **waited** on the session at pickup | **Boot left on the critical path**. Near 0 means pre-open works |
| `preopen_idle_ms` | Booted VM idle before a worker picked it up | Must stay ≪ 15 min idle timeout; sizes the heartbeat and P |
| `open_attempts`, `infra_error` | Retries needed / episode lost to infra (open, grade) | Keeps infra failures **out of** the reward signal |
| `grade_ms`, `grade_timeout` | Test runtime (includes installing uv and pytest) / tests hit their limit | Grading is part of every episode's latency |
| `agent_ms`, `episode_ms`, `session_s` | Agent loop time / total episode time / microVM lifetime | `session_s` summed = **microVM-seconds = sandbox cost** |
| `perf/sandbox_ms_*` vs `perf/sglang_gen_ms_*` | Time in sandbox ops vs model generation | Which tier to scale: sandboxes or inference |

: Metrics: grading, infrastructure and cost

## Results (1/2): data and learning

| Item | Value | Source |
|------------------|--------------------------------------------------|------------------|
| Deployable Runtimes | 218 / 229 | `tasks/images.csv` |
| Oracle-usable tasks | **166 / 218** (47 oracle fails here · 5 errored/timeouts · **0** null-passes) | `tasks/oracle.csv` |
| Dataset | 135 train + 31 held-out eval | `data/tb_*.jsonl` |
| Train solve rate, steps 1–10 | 0 · 0 · 0.088 · 0 · 0 · 0 · 0 · 0.026 · 0 · 0 (4 solves / 382 episodes) | `rollout/raw_reward` |
| Held-out solve rate (step 9) | 0.0 (31 tasks × 2) | `eval/terminalbench` |
| Partial progress | 160 / 382 episodes pass ≥ 1 hidden test | `tests_passed` |

: Results: data and learning (smoke: RB = 4, NS = 8, 10 steps + eval)

## Results (2/2): sandbox and cost

| Item | Value | Source |
|------------------|--------------------------------------------------|------------------|
| Boot hidden? `open_wait_ms` | **p50 0 ms · p95 0 ms** · max 4.9 s (first 4 episodes only) | `[demo]` start lines |
| Session open `open_ms` | p50 0.66 s · p95 2.5 s · 100 % warm hit · 0 open retries | `[demo]` start lines |
| Episode time p50 / p90 | 21 s / 64 s (agent 9 s + grade 10 s) | `episode_end` |
| Pre-open idle p50 / max | **7.3 min** / 16.5 min (heartbeat kept all alive) | `preopen_idle_ms` |
| microVM-seconds per train step | ≈ 13 000 (32 × 411 s mean `session_s`); ≈ 95 % idle | Σ `session_s` |
| Infra errors | **0** episodes lost; 15 `read_file` tool errors on 4 tasks (agent-visible, not reward) | `infra_error`, `[sbx-op]` |
| Cold open by image size | *pending* (`probe_coldstart.py` not yet run) | `tasks/coldstart.csv` |

: Results: sandbox and cost (W = 4, `TB_PREOPEN_BATCHES` = 2)

## What the smoke run tells us

- **The pipeline works end to end:** 382 episodes, each on its own fresh microVM, 0 lost to infra, job succeeded in ≈ 70 min.
- **Pre-open fully hides the boot:** `open_wait_ms` is 0 at p95. But the price is high: a booted VM idles **≈ 7 min** for a **≈ 21 s** episode, so ≈ 95 % of microVM-seconds are idle. Per the tuning table, **`TB_PREOPEN_BATCHES` = 1** (or 0, since opens are ≈ 0.7 s warm hits) should keep boot hidden at a fraction of the cost.
- **Reward is sparse, not dead:** the tool loop runs (median 6 turns, 5 bash calls), 42 % of episodes pass some tests, but only 4 full solves (`acl-permissions-inheritance` ×3, `git-leak-recovery`). Qwen3-4B with 8 turns rarely produces mixed GRPO groups, so a longer run needs an easier curriculum (easy-first), more turns or a partial-credit reward.

## References

| Ref | Source | What it supports |
|------------|------------------------------------------------------------------|----------------------------------------------|
| [Q1] | AgentCore quotas: <https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/bedrock-agentcore-limits.html> | Total agents (Runtimes) per account **1,000**, adjustable · Docker image ≤ **2 GB**, fixed · new-session rate 25/s, account-wide · request timeout **15 min**, fixed · idle session timeout 15 min |
| [Q2] | Service Quotas for this account (`aws service-quotas list-service-quotas --service-code bedrock-agentcore`, us-west-2, 2026-09-23) | *Total Agents per Account* = 1000 (L-F4575653) · *Max Docker image size* = 2048 MB (L-0A9E32B3) · *Rate of new Runtime session creation* = 25/s (L-8EE2AEA2) · *Active Session Workloads per Account* = 5000 (L-3E5722B2) · *Control plane mutation APIs* = 50/s (L-DD349033) |
| [D3] | Runtime, how it works: <https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-how-it-works.html> | Session = dedicated microVM; ≈ 15 min idle timeout, 8 h max lifetime |

: References
