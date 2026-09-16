# demo_cpu_quiesce — pause/resume the sandbox across the weight-sync barrier

A **standalone, self-contained** demo of the *quiesce/resume* pattern from
[`docs/demos-program.md`](../../../docs/demos-program.md) ("quiesce / resume the sandbox across the
weight-sync barrier"). During slime's trainer→inference **weight sync**, generation stalls, so open
AgentCore sandbox sessions sit idle — and AgentCore reclaims an idle session purely on **absence of
invokes** (there is no "pause" primitive and no snapshot; see
[`docs/gpu-snapshot.md`](../../../docs/gpu-snapshot.md)). This demo keeps the session warm through
that window with a **heartbeat that lives outside the harness**, and backstops a genuine loss with
recovery.

Unlike the Tier-1/2 training demos, the deliverable here is a **coordination mechanism**, not a
convergence curve — so it runs in seconds, offline, with no AWS and no GPU.

## What it shows — the three-act arc
`python3 quiesce_demo.py` plays the exact arc from the demos program:

1. **ACT 1 — naive "stop invoking" (anti-pattern).** During the sync we simply stop touching the
   sandbox. The idle timer trips, AgentCore reclaims the session, and on resume it is **gone** — the
   harness would crash.
2. **ACT 2 — heartbeat quiesce.** A `SessionKeeper` (outside the harness) **gates** new invokes for
   the sync window and **heartbeats** the session at half the idle timeout. It survives a pause far
   longer than the timeout; resume is instant. The demo also proves the gate: an invoke issued
   *during* the sync blocks until `sync_end`.
3. **ACT 3 — recover-if-reclaimed.** Even heartbeated, a **GPU instance recycle** kills the session
   (no snapshot to reattach). The keeper is *necessary, not sufficient*: the death is detected and
   the episode **recovers by re-seeding** on a fresh session instead of crashing.

## The design (the point of the demo)
The heartbeat deliberately does **not** live in the harness — no per-harness keepalive feature.
Two reusable pieces in the shared library (`agentcore_slime_demo/session_keeper.py`) carry it:

- **`SessionKeeper`** — a process-wide registry + one background heartbeat loop, driven by the
  trainer's `sync_begin()` / `sync_end()` barrier. During the sync window it holds new invokes at a
  **gate** (the "lightweight quiesce") and pings any session going stale. Outside the window it
  fires **zero** heartbeats (real traffic refreshes the same idle clock).
- **`KeptSandbox`** — composes *any* `Sandbox` (Code Interpreter, Runtime, Local) with the keeper:
  every real op awaits the gate and touches the clock; the heartbeat is a raw no-op op (`exec
  "true"` = one `executeCommand`/`invoke_agent_runtime` round-trip) issued **below** the gate, since
  a heartbeat during quiesce must defeat the gate, not wait on it.

The offline stand-in for AWS's idle timer is `agentcore_idle_sim.py` (`AgentCoreIdleSim` +
`ReclaimableLocalSandbox`) — it reclaims an idle `LocalSandbox` in seconds and can `force_recycle`
to model a GPU instance recycle. Recovery reuses the real contract in `agentcore_slime_demo/recovery.py`.

## Run it
```bash
python3 quiesce_demo.py                    # defaults: IDLE=4s PAUSE=10s TICK=0.5s
IDLE=6 PAUSE=30 python3 quiesce_demo.py    # a much longer pause; still survives on the heartbeat
```
No dependencies beyond the vendored `agentcore_slime_demo/` library and the Python stdlib. Also
runnable via `bash run.sh` (defaults) inside the `agentcore_p4` container.

## Run against real AgentCore
Delete the local idle simulator and let AWS be the clock:
- Point `KeptSandbox` at an `AgentCoreSandbox` (Code Interpreter) or `AgentCoreRuntimeSandbox` (GPU
  Runtime) instead of `ReclaimableLocalSandbox`; the heartbeat becomes a real
  `executeCommand`/`invoke_agent_runtime`.
- Set `idle_timeout` to the deployed value (**900 s** Code Interpreter, **600 s**
  `idleRuntimeSessionTimeout` for the GPU Runtime — `deploy_two_runtime.py:402,430`).
- To *see* a reclamation in ACT 1 you must either pause longer than that idle timeout, or deploy the
  Runtime with a low `idleRuntimeSessionTimeout`. ACT 2 (heartbeat survival) and ACT 3 (recovery)
  work at any pause length.

The **GPU-instance tier** is where this pays off (idle GPU time is real money). On serverless CPU
microVMs the same coordination runs but there is no cost payoff — it's shown purely as the pattern.

## Wiring it into a real slime run
`SessionKeeper` is driven by the weight-sync barrier, so a real integration is two calls around
slime's trainer→inference weight update: `keeper.sync_begin()` before the update, `keeper.sync_end()`
after (and wrap each episode's sandbox in `KeptSandbox`). This deliberately re-couples the sandbox to
the sync barrier — a **cost** optimization, not a correctness fix (async RL episodes are otherwise
designed to span weight versions).

## Folder layout (self-contained, layout B)
- `quiesce_demo.py` — the standalone three-act driver (this is "the run")
- `agentcore_idle_sim.py` — offline AgentCore idle-timeout + instance-recycle simulator
- `agentcore_slime_demo/` — vendored shared library (imports only this copy); adds `session_keeper.py`
- `deploy/Dockerfile.sandbox.cpu` — the CPU sandbox image (for the real-AgentCore variant)
- `run.sh` — convenience runner

References (not copied): `docs/demos-program.md`, `docs/gpu-snapshot.md`, and the shared
`agentcore_slime_demo/recovery.py` contract.
