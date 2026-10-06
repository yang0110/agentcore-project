# Fault tolerance & auto-recovery: surviving sandbox failures during agentic RL

> **Historical (2026-09-30).** This doc was written for demos and code dirs that were removed on 2026-09-30 (and the microVM-only `demo_cpu_terminalbench` on 2026-10-06); the only live demo is [`code/demos/demo_terminalbench_kernelbench`](../code/demos/demo_terminalbench_kernelbench/README.md).

In agentic RL you run thousands of rollouts, each booting a fresh sandbox session. Sessions
**will** die mid-training — idle/lifetime timeouts, microVM crashes, severed transport, throttling.
If a dying sandbox crashes the rollout coroutine, you lose the whole training step. This doc covers
(1) what AgentCore/E2B actually do on failure, (2) the recovery strategy this repo implements, and
(3) how it's tested offline. **Every external claim has a source URL.**

Companion code: [`recovery.py`](../code/demos/demo_terminalbench_kernelbench/agentcore_slime_demo/recovery.py) (vendored in the Terminal-Bench demo lib). The offline
demo `code/scripts/recovery_smoke.py` and the tests in `code/tests/` were removed on 2026-09-30 (git history).

---

## 1. AgentCore Runtime — recovery semantics (grounded)

AgentCore's managed microVMs give **automatic session recovery**, which is exactly what an RL
trainer wants: you don't rebuild infra, you just re-invoke.

| Fact | Source |
|------|--------|
| Each session runs in its own dedicated, isolated microVM; after completion the microVM is **terminated and memory sanitized**. | [runtime-sessions](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-sessions.html) |
| A session **Stops** on: **15-min inactivity** (default), **8-h max lifetime** (default), explicit `StopRuntimeSession`, or **being deemed unhealthy by health checks**. | [runtime-sessions](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-sessions.html) |
| After a stop/timeout/crash, the session **"transitions back to Active on the next invocation and a new compute is provisioned"** — the logical session survives until the runtime ARN is deleted. | [runtime-sessions](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-sessions.html) |
| Compute is **ephemeral by default**: "Any data stored in memory or written to disk persists only for the compute lifecycle." Persist across stop/resume only with **session storage**. | [runtime-sessions](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-sessions.html) |
| **Session affinity** requires resending the session ID; otherwise requests hit a new microVM (cold start). | [runtime-sessions](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-sessions.html) |
| Health check is **`GET /ping`**, used for "service monitoring to detect and remediate issues" and **"automated recovery through AWS's managed infrastructure"**. Status ∈ **`Healthy` / `HealthyBusy`** (HealthyBusy keeps the session alive during async work). | [http-protocol-contract](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-http-protocol-contract.html) |
| **409 `RetryableConflictException`** (session op in progress) is transient — "Retry with short exponential backoff. The AWS SDKs auto-retry this exception." **429 `ThrottlingException`**, **424 `RuntimeClientError`** (your container returned 4xx/5xx), **500** internal. | [http-protocol-contract](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-http-protocol-contract.html) |
| Sessions last **up to 8 h on microVMs, 14 days on Instances**. | [runtime-sessions](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-sessions.html) |

Code Interpreter tool SDK surfaces (cloned repo `bedrock-agentcore-sdk-python`):
`CodeInterpreter.get_session().status ∈ {READY, TERMINATED}` (health poll), and `invoke()`
**auto-starts a new session if none is active** (`tools/code_interpreter_client.py:308-465`).

> **Key implication for RL:** recovery = **restart the rollout on a fresh session**, not resume.
> Because in-VM state is ephemeral by default, a dead session means the agent's workspace is gone;
> the correct move is to re-provision and re-run the episode from scratch (or restore from session
> storage/S3 if you configured it). That's what `resilient_episode` does.

## 2. E2B — recovery semantics (for contrast; slime's reference backend)

| Fact | Source |
|------|--------|
| Reconnect to a live sandbox via `Sandbox.connect(sandboxId)` (list with `Sandbox.list()`). | [connect](https://docs.e2b.dev/sandbox/connect) |
| **Pause saves filesystem AND memory; resume restores exact state**; paused sandboxes have **no TTL** and are never auto-killed. | [persistence](https://docs.e2b.dev/sandbox/persistence), [lifetime](https://docs.e2b.dev/faq/sandbox-lifetime.md) |
| Default timeout **5 min**; `onTimeout:'kill'` (default) terminates and **state is unrecoverable**; `onTimeout:'pause'` preserves state. | [lifetime](https://docs.e2b.dev/faq/sandbox-lifetime.md) |

**AgentCore vs E2B recovery:** AgentCore auto-reprovisions compute under a surviving *logical*
session (state ephemeral unless session storage); E2B can **pause/resume with full memory+fs
snapshot** (true resume). For a from-scratch coding task, restart-on-fresh-session is fine on
either; for expensive-to-rebuild environments, E2B pause/resume or AgentCore session storage wins.

## 3. RL best practices for environment failures (grounded)

| Practice | Source |
|----------|--------|
| **Agent Lightning** explicitly handles rollout failures — "agent crashes, network interruptions, or invalid outputs" and "long-hanging tool calls" — and **"Failed tasks can be retried or reassigned to other agent instances"** so "these failures do not disrupt the overall training process" (§3.4.2 Error Handling and Robustness). | [arXiv 2508.03680](https://arxiv.org/html/2508.03680v1) |
| **Ray RLlib** worker fault tolerance: `restart_failed_env_runners=True` restarts failed EnvRunners; `restart_failed_sub_environments` restarts a single env "without escalating errors"; `num_consecutive_env_runner_failures_tolerance` bounds consecutive failures before failing the job. | [rllib-fault-tolerance](https://docs.ray.io/en/latest/rllib/rllib-fault-tolerance.html) |
| **Gymnasium AutoReset**: env auto-resets on termination/truncation (episode-boundary handling). | [wrappers](https://gymnasium.farama.org/api/wrappers/misc_wrappers/) |
| **verl** recommends checkpointing precisely because "there could be training errors or machine failure during the whole RLHF training process." | [verl checkpoint](https://verl.readthedocs.io/en/latest/advance/checkpoint.html) |

slime already embodies these in its shipped stack (cloned repo):
- **RPC-level retry** with backoff+jitter for transient transport errors, and *no* retry for
  non-idempotent ops after a severed response — `E2BSandbox._rpc_retry` (`slime/agent/sandbox.py:237-271`).
- **Detached exec + done-marker polling** so a long tool run survives a severed stream —
  `exec_and_wait` (`slime/agent/sandbox.py:82-145`).
- **Boot retries** recreating the sandbox on failure — `boot_agent_sandbox` (`examples/coding_agent_rl/generate.py:95-134`).
- **Clean abort** of an unrecoverable rollout → sample marked `ABORTED`, `remove_sample=True`, so
  the trainer drops it and the step still runs — `_abort_result` (`examples/coding_agent_rl/generate.py:316-333`).

## 4. What this repo implements

`recovery.py` gives the same three layers, provider-agnostic:

1. **RPC-level self-heal** (`AgentCoreSandbox._call`): transient transport errors
   (throttle/unavailable/500/timeout/connection) retry the **same op on the same session** with
   exponential backoff; genuine **session-death** markers (terminated/expired/not found) are *not*
   retried in place — they propagate so the episode restarts on a fresh session (in-VM state is gone).
2. **Episode-level recovery** (`resilient_episode`): on any recoverable failure — boot or
   mid-episode — recreate the sandbox + policy and **restart the episode on a fresh session**, up to
   `max_boot_retries`, with jittered exponential backoff.
3. **Clean abort** (`_aborted`): if recovery is exhausted or the error is fatal (auth/config), return
   an **ABORTED** `Trajectory` (reward 0, `status="aborted"`) — **never raise**. The trainer drops it.

Error classification (`is_recoverable`): terminated/expired/throttle/unavailable/timeout/connection
→ recoverable; `AccessDenied`/unauthorized and explicit `FatalSandboxError` → fatal.

```
resilient_episode(task, make_policy, make_sandbox, max_boot_retries=3)
  ├─ attempt 1: boot → run_episode → ✅ return
  ├─ transient fail → backoff(jitter) → attempt 2: fresh sandbox + fresh session
  ├─ ...
  └─ exhausted / fatal → ABORTED trajectory (reward 0), trainer drops the sample
```

## 5. Tested offline (no AWS/GPU)

`FlakySandbox` injects failures into `LocalSandbox` to exercise every path.
`python3 code/scripts/recovery_smoke.py`:

```
[healthy (baseline)      ] status=completed solved=True  reward=1.0 boots=1
[2 transient boot fails  ] status=completed solved=True  reward=1.0 boots=3   # recreated, then solved
[mid-episode death       ] status=completed solved=True  reward=1.0 boots=2   # fresh session, restarted
[fatal (unrecoverable)   ] status=aborted   solved=False reward=0.0 boots=1   # clean abort, no crash
```

Tests (`code/tests/test_smoke.py`, run `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest code/tests -q`):
`test_recovery_from_transient_boot_failures`, `test_recovery_from_mid_episode_session_death`,
`test_unrecoverable_aborts_cleanly_without_raising`, `test_error_classifier` — all green (9/9).

## 6. Production checklist

- [ ] Set `max_boot_retries` and a per-rollout **wall-clock guard** (slime uses `SWE_ROLLOUT_GUARD_SEC`)
      so a hung session can't stall a step forever.
- [ ] Treat aborted rollouts as **dropped, zero-reward** samples — never let them poison the batch.
- [ ] Poll `GET /ping` / `get_session().status` for health; recycle `HealthyBusy`-stuck sessions.
- [ ] Retry **409/429/5xx** with backoff (SDK auto-retries 409); do **not** blind-retry non-idempotent
      tool actions after a severed stream (slime's `idempotent=False` rule).
- [ ] If your environment is expensive to rebuild, enable **AgentCore session storage** (or E2B
      **pause/resume**) so recovery restores state instead of restarting from scratch.
- [ ] Watch **concurrent-session quotas** — a `HealthyBusy` keepalive bug can pin sessions to
      MaxLifetime and exhaust quota ([http-protocol-contract](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-http-protocol-contract.html)).

---

### Note on the reference paper ID
The robustness quotes in §3 are from **arXiv 2508.03680** (the original *Agent Lightning*, whose
HTML renders on arXiv). The v1.0 report *Towards Harnessed Agentic RL* is cited by the repo as
**arXiv 2608.17528** (`repos/agent-lightning/README.md`); at research time its arXiv HTML did not
resolve, so §3's verifiable quotes are taken from 2508.03680.
