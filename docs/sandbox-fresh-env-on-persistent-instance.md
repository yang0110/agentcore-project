# Fresh env on a persistent (reused) sandbox instance

**Question.** An AgentCore Runtime on a capacity-provider Instance takes ~60–80 s to cold-boot. If
we keep ONE instance warm and run many RL episodes on it, how do we stop one episode's state from
leaking into the next — without throwing away the warm-instance speedup?

**Short answer.** Invoke every episode under **one shared `runtimeSessionId`** (so they all land on
the same warm instance), give every episode a per-episode **`sub_session`**, and inside the server
give each `sub_session` a **fresh environment** — its own workdir *and* `HOME`/`TMPDIR`/caches,
wiped on close. The CPU farm isolates *everything* (nothing warm to protect); the GPU farm isolates
writes but keeps caches *shared* (to preserve the warm compiled-CUDA-kernel cache). This is
implemented in [`agentcore_slime_demo/sandbox_farm_env.py`](../code/demos/demo_cpu_bigimage/agentcore_slime_demo/sandbox_farm_env.py)
+ [`sandbox_cpu_farm.py`](../code/demos/demo_cpu_bigimage/agentcore_slime_demo/sandbox_cpu_farm.py)
(CPU) and the opt-in block in [`sandbox_ray_farm.py`](../code/demos/demo_cpu_bigimage/agentcore_slime_demo/sandbox_ray_farm.py) (GPU).

## When reuse matters: Instances, not microVMs

This is a **substrate-specific** concern. A managed **Code Interpreter microVM** cold-starts in a
second or two, so we **deliberately do not reuse it** — every episode mints its own session (a fresh
microVM), which is cheaper, faster, and perfectly isolated for free. Reuse only pays off on the
**Instances** compute type — a capacity-provider EC2 box that takes **~60–80 s to cold-boot** — and
that applies to **both** Instance flavours:

- the **GPU `g6`** (`demo_gpu_shared` / `demo_gpu_local`), and
- the **CPU `m5`** (`demo_cpu_bigimage`, the big-image demo this doc ships with).

Both use the *same* shared-session + `sub_session` mechanism below; they differ only in what the
`sub_session` multiplexes (a fractional-GPU actor vs. an isolated CPU env) and the cache policy.

## Why reuse needs a freshness story at all

AgentCore maps **1 `runtimeSessionId` → 1 Instance** (1:1). Two ways to run N episodes:

| Pattern | Sessions | Instances | Cold-boots | Isolation |
|---|---|---|---|---|
| **Fresh per episode** (`SANDBOX=agentcore-runtime`) | N (one per episode) | N (one boots per session) | **N** — boot dominates wall-clock | perfect (a new box each time) |
| **Warm reuse** (`SANDBOX=agentcore-ray-farm`) | **1 shared** + N `sub_session`s | **1** warm instance | **1** for the whole run | must be *engineered* (this doc) |

Warm reuse pays the boot once, but now episodes share a live box — so we must reconstruct the
isolation the fresh-instance path got for free.

## The four levels of freshness

An episode can leak state to the next one through four channels. From "already handled" to "the real
work":

1. **In-process Python / GPU state** — never an issue: every `exec` spawns a **fresh subprocess**, so
   no interpreter globals or file handles survive a command, let alone an episode.
2. **The working directory** — each `sub_session` gets its own `{root}/{sub}` dir, created fresh on
   `open` and `shutil.rmtree`'d on `close` (`wipe_session`). Files written under the workdir never
   cross episodes, and nothing lingers on the instance after the episode ends.
3. **`$HOME` and `$TMPDIR` / `/tmp`** — the real leak vector. A checker that writes `~/.foo` or
   `/tmp/scratch` would, on a shared instance, be visible to the next episode. `fresh_env` redirects
   `HOME` → `{sub}/home` and `TMPDIR`/`TMP`/`TEMP` → `{sub}/tmp` (plus `PYTHONPYCACHEPREFIX` so
   `.pyc` writes stay in the session too). Since those live under the workdir, level 2's wipe cleans
   them as well.
4. **Write caches** (`pip`, `HF_HOME`, `TORCH_HOME`, `XDG_CACHE_HOME`) — the subtle one, because
   redirecting `HOME` (level 3) silently drags the default `~/.cache` into the session dir too. That
   is either exactly what you want or exactly what you *don't*, depending on the substrate — see next.

## CPU vs GPU: the one real asymmetry

**CPU farm (`SANDBOX_FARM=cpu`, `isolate_caches=True`).** There is nothing warm worth keeping — the
big-image checker is CPU Python, no compiled-kernel cache. So isolate **everything**: caches follow
`HOME` into the session dir and get wiped with it. Freshness is total; the only cost is that pip/HF
re-download per episode, which this demo doesn't do anyway (deps are baked into the image).

**GPU farm (`SANDBOX_FARM=gpu`, `isolate_caches=False`, opt-in via `SANDBOX_ISOLATE_HOME=1`).** The
whole point of the GPU farm is a **warm actor process** with a hot CUDA context and a
**compiled-CUDA-kernel cache** — recompiling per episode would erase the speedup. Those caches are
**content-addressed and read-mostly**: identical regardless of which episode produced them, so
sharing them is both safe and desirable. So the GPU path redirects `HOME`/`TMPDIR` for **write
isolation** but pins the caches to a **shared** warm root (`SANDBOX_SHARED_CACHE`,
default `/workspace/_shared_cache`) — explicitly, because otherwise the per-session `HOME` would move
`~/.cache` off the shared path and cold-start every kernel.

> Note the GPU block is **default-OFF** (`SANDBOX_ISOLATE_HOME=1` to enable) so the
> hardware-verified B4 path stays byte-identical unless you opt in. The CPU farm isolates by default.

```
                         HOME / TMPDIR         caches (pip/HF/torch/xdg)
CPU farm  (isolate=True)   -> {sub}/home,tmp     -> {sub}/home/.cache   (wiped on close)
GPU farm  (isolate=False)  -> {sub}/home,tmp     -> /workspace/_shared_cache (SURVIVES — warm)
```

## Wiring (what runs where)

- **Client** — `RayFarmRuntimeSandbox` (`SANDBOX=agentcore-ray-farm`): one shared `runtimeSessionId`
  from `B4_SHARED_RUNTIME_SESSION` (minted once per run in each demo's `run.sh`, ≥33 chars) + a
  per-episode `sub_session`. `__aenter__` sends `open`, `__aexit__` sends `close`.
- **Server** — `sandbox_server_agent.py` routes any op carrying a `sub_session` to `_farm()`, which
  builds `CpuFarm` (`SANDBOX_FARM=cpu`) or `RayGpuFarm` (`gpu`, default) once and reuses it.
- **Freshness** — both farms call `fresh_env(session_dir, env, isolate_caches=...)` per `exec` and
  `wipe_session(session_dir)` per `close`.

## Verifying the win

`metrics.py` splits sandbox latency into **cold** (`perf/sandbox_open_*`, the session-open `ping`)
and **warm** (`perf/sandbox_*`, data ops). With per-episode sessions the cold-boot **count** equals
the episode count; with the shared-instance farm it drops to **1** for the whole run, while warm-op
latency (~150 ms) is unchanged. That count is the headline metric that proves reuse without breaking
per-episode freshness.

## Related

- [`demo_cpu_bigimage/README.md`](../code/demos/demo_cpu_bigimage/README.md) — the CPU demo that
  carries this farm (§ "Warm-instance reuse").
- [`sandbox-concurrency-ablation.md`](sandbox-concurrency-ablation.md) — the CONC sweep the shared
  instance makes cheap to run.
- [`gpu-snapshot.md`](gpu-snapshot.md) — why we reuse a *live* warm instance rather than snapshot one.
