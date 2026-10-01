# Demo Plan — Minimalist Agentic RL: slime + AgentCore Runtime sandbox + Claude Code harness

> **Historical (2026-09-30).** This doc was written for demos and code dirs that were removed on 2026-09-30; the only live demo is [`code/demos/demo_cpu_terminalbench`](../code/demos/demo_cpu_terminalbench/README.md).

**Goal.** Train a policy with slime on a coding task where the agent loop is **Claude Code**, the
agent's *brain* is the **slime-served policy** (SGLang, not Anthropic's API), and every tool action
(Bash/Edit) executes inside an **AWS AgentCore Runtime sandbox**. Prove one full RL step end-to-end
on a single task, then scale.

This plan reuses slime's existing agent stack unchanged — the only new code is a sandbox adapter.
Companion docs: [`agentcore-slime-integration.md`](./agentcore-slime-integration.md) (the
`AgentCoreSandbox` class + `Sandbox` Protocol) and [`sandbox-comparison.md`](./sandbox-comparison.md).

> **Reference paper (confirmed):** *Agent Lightning v1.0: Towards Harnessed Agentic RL*, Microsoft,
> arXiv **2608.17528** — converted to [`agent-lightning-v1-harnessed-agentic-rl.md`](./agent-lightning-v1-harnessed-agentic-rl.md),
> repo cloned at `repos/agent-lightning`. This plan follows its three-component design (Trainer /
> API Gateway / Rollout Controller), its "train with real harnesses, ZERO changes" principle, and
> its unified per-turn trajectory / credit-assignment data model. The role mapping is in §1a below.

> **Status:** a runnable implementation exists under [`../code/`](../code/) — see §11. The offline
> path (LocalSandbox + a scripted policy) passes end-to-end; the AWS/slime path is the same code
> with two backends swapped in.

> **Update — real-CLI harness removed; harness is in-process.** This plan was originally written
> around the **real Claude Code CLI** as the harness (a real-CLI generate entry + a custom
> multi-user container). That code and its CLI toolchain have since been **removed** from the repo;
> the live harness is now our **mini Claude-Code-like harness** (`MiniClaudeHarness`,
> `mini_agent.py`), run **in-process** next to the trainer (see
> [`harness-sandbox-decoupling.md`](./harness-sandbox-decoupling.md)). The earlier variant that also
> hosted the harness on its own AgentCore Runtime has been dropped — only the *sandbox* runs as an
> AgentCore Runtime. The principle below is unchanged — the slime policy is the agent's brain,
> served through the adapter, with tool exec in an AgentCore sandbox — only the harness
> implementation differs. Read §1/§1a/§5–§6 as the original CLI-based framing.

---

## 1. What "Claude Code as the harness between" actually means

The three roles are cleanly separated — this is the whole point of the demo:

| Role | Who | Where it runs |
|------|-----|---------------|
| **Brain** (policy being trained) | slime's SGLang engine | slime GPU cluster |
| **Harness** (agent loop, tool decisions) | **Claude Code CLI** (`claude -p ... --output-format stream-json`) | inside the AgentCore sandbox |
| **Hands** (tool execution) | Bash/Edit/Write tools | inside the same AgentCore sandbox microVM |
| **Bridge** (records tokens for RL) | slime's Anthropic adapter | slime host, reachable from the sandbox |

The trick (verified in `repos/slime/slime/agent/harness/claude_code.py:57-71`): Claude Code is
launched with `ANTHROPIC_BASE_URL=<slime adapter url>` and `ANTHROPIC_AUTH_TOKEN=<session_id>`. So
every "Anthropic" completion Claude Code requests is actually served by the **slime policy** via the
adapter, which records the exact tokens + logprobs for training. Claude Code never talks to
Anthropic; it just provides the agent scaffolding and tool protocol.

```
┌── slime GPU cluster ─────────────────────────┐        ┌── AWS: AgentCore sandbox (microVM) ──┐
│  SGLang policy  ◄──/generate──┐               │        │  Claude Code CLI (harness)           │
│                               │               │        │   ANTHROPIC_BASE_URL ─────┐          │
│  Anthropic adapter  ◄─────────┘               │        │   tool calls: Bash/Edit   │          │
│   (records tokens, loss-masks tool output) ◄──┼── HTTP ┼───(reverse dial-back)─────┘          │
│                               ▲               │        │   executes in this microVM           │
│  Megatron trainer ◄── Samples ┘               │        │   git diff ← reward via tests        │
└───────────────────────────────────────────────┘        └──────────────────────────────────────┘
```

Loop (from `repos/slime/examples/coding_agent_rl/generate.py:182-268`, the four-stage orchestrator):
`open_session → boot sandbox + install Claude Code → prepare_workspace → harness.run (Claude Code
solves the task) → git_diff → run_evaluation (reward) → adapter.finish_session → list[Sample]`.

---

## 1a. Mapping to Agent Lightning v1.0 (arXiv 2608.17528)

The AL report names three lightweight components and one principle. Each maps 1:1 onto this stack —
that's *why* the demo is shaped this way:

| Agent Lightning v1.0 | AL implementation | This demo |
|---|---|---|
| **Trainer** — builds samples, updates the policy | verl + vLLM | **slime** (SGLang rollout + Megatron train), fed via `--custom-generate-function-path` |
| **API Gateway** — proxies model requests, captures training data (returns token ids + logprobs, no retokenization drift) | AL proxy | slime's **`AnthropicAdapter`** — Claude Code's `ANTHROPIC_BASE_URL` points at it; it serves the SGLang policy and records tokens |
| **Rollout Controller** — runs agents locally or as K8s Jobs, keeps tools/env in the loop | AL controller | **`ClaudeCodeHarness`** driving the **AgentCore sandbox**; one rollout = one isolated session |
| **Real harness, ZERO changes** | unmodified agents | **Claude Code CLI** runs unmodified; only `ANTHROPIC_BASE_URL`/`_MODEL` env are set |
| **Unified per-turn trajectory / credit assignment** | MDP transitions, LightningRL | the adapter's segment split → loss-masked `Sample`s (assistant tokens trained, tool/observation masked) |

The offline demo under `../code/` stands in `SGLangPolicy→MockPolicy` and `AgentCore→LocalSandbox`,
but keeps the same decoupling: the harness only emits a trajectory; sample construction is separate.

---

## 2. Components (all real, cited)

| Component | File | What it does | Change for demo |
|-----------|------|--------------|-----------------|
| Per-sample rollout | `examples/coding_agent_rl/generate.py` | orchestrates the loop | copy → swap sandbox |
| Sandbox contract | `slime/agent/sandbox.py:27-59` (`Sandbox` Protocol); `:160` `E2BSandbox` | `exec/write_file/read_file` | **implement `AgentCoreSandbox`** |
| Claude Code harness | `slime/agent/harness/claude_code.py` | installs + runs `claude -p`, wires `ANTHROPIC_BASE_URL` | reuse as-is |
| Anthropic↔SGLang adapter | `slime/agent/adapters/{anthropic,common}.py` | serves policy tokens, records turns | reuse as-is |
| Trajectory builder | `slime/agent/trajectory.py:269` | loss-masked `list[Sample]` | reuse as-is |
| SWE task/reward | `examples/coding_agent_rl/swe.py` | workspace prep, `git_diff`, `run_evaluation` | reuse; use 1 tiny task |
| Sandbox adapter (**new**) | `code/agentcore_slime_demo/sandbox.py` (`AgentCoreSandbox`) | AgentCore → `Sandbox` Protocol | implemented; drop-in for `E2BSandbox` |

**New code = `AgentCoreSandbox` + a thin generate** (`generate.py`, the in-process harness path).
Everything else — adapter, trajectory builder — is slime's shipped stack, unchanged.

---

## 3. The one wiring decision: sandbox reachability

`generate.py:144-149,171` **requires** `ADAPTER_PUBLIC_HOST` — the host IP the sandbox dials back to
reach slime's adapter. This is the single non-trivial integration point with AgentCore. Two options:

- **Option A — AgentCore Code Interpreter, PUBLIC network (simplest).** The sandbox has outbound
  internet; expose slime's adapter on a publicly reachable host/port (a bastion, an ALB, or a tunnel
  for the demo). Set `ADAPTER_PUBLIC_HOST` to that address. Install Claude Code inside via
  `sb.write_file` + `sb.exec("npm i -g ...")` or pre-uploaded tarballs (matches
  `claude_code.py:36-42`, which installs from host tarballs `SLIME_AGENT_NODE_TARBALL` /
  `SLIME_AGENT_CC_TARBALL`).
- **Option B — AgentCore custom Runtime + VPC.** Bake Node + Claude Code into a Runtime container
  (the samples repo already has `agentcore-samples/.../04-coding-agents/` doing exactly this), run it
  in the **same VPC** as the slime adapter, and set `ADAPTER_PUBLIC_HOST` to the private IP. More
  setup, but no public exposure and no per-rollout CLI install.

**Recommendation:** start with **Option A** for the minimalist demo; move to **B** for real training
(no public exposure, faster boot, controllable egress).

---

## 4. Prerequisites

- slime installed and a small policy checkpoint that already does basic tool/code following
  (e.g. a Qwen-2.5-Coder-3B/7B-Instruct-class model — big enough to emit valid tool calls).
- SGLang rollout + Megatron train working on your GPUs (standard slime setup).
- AWS account with Code Interpreter IAM actions (`bedrock-agentcore:CreateCodeInterpreter /
  StartCodeInterpreterSession / InvokeCodeInterpreter / StopCodeInterpreterSession / Delete* /
  List* / Get*` — integration doc §2.1) and a region where AgentCore is available.
- `pip install bedrock-agentcore boto3` in the rollout image.
- A publicly reachable (Option A) or VPC-reachable (Option B) address for the adapter.
- Node 22 + Claude Code CLI tarballs (or npm access from inside the sandbox).
- **One** tiny SWE-style task instance (repo + failing test + expected fix) to keep it minimal.

---

## 5. Build phases (each phase ends in something you can run)

**Phase 0 — Sanity: sandbox alone (30 min).**
Implement/borrow `AgentCoreSandbox` (integration doc §4). Verify outside slime:
```python
async with AgentCoreSandbox(region="us-west-2") as sb:
    print(await sb.exec("echo hello && python3 -c 'print(2+2)'"))   # (0, "hello\n4\n", "")
    await sb.write_file("/tmp/x.py", "print('hi')")
    print(await sb.exec("python3 /tmp/x.py"))                        # (0, "hi\n", "")
```
✅ Exit: `exec/write_file/read_file` return correct `(exit_code, stdout, stderr)`.

**Phase 1 — Claude Code installs & runs in the sandbox (1–2 h).**
Point `ClaudeCodeHarness.install_cli` at the AgentCore sandbox. Confirm
`claude --version` runs inside the microVM, and that `claude -p "print hello world in python and run it"`
with `ANTHROPIC_BASE_URL` set to a **mock** endpoint produces tool calls.
✅ Exit: Claude Code launches headless in the sandbox and makes outbound calls to the adapter host.

**Phase 2 — Adapter reachability (1 h).**
Stand up slime's `AnthropicAdapter` (via `_AdapterService`), set `ADAPTER_PUBLIC_HOST` per §3, and
confirm the sandbox's Claude Code reaches it and the adapter forwards to SGLang (`/generate`).
✅ Exit: one Claude Code turn round-trips through the adapter → SGLang → back, tokens recorded.

**Phase 3 — One full rollout (2–3 h).**
Use the implemented `code/agentcore_slime_demo/generate.py` (the in-process harness path). Run
`generate()` on the single task in **eval mode** (`evaluation=True`) — no training yet.
✅ Exit: prints `reward=... applied=... agent_exit_code=... elapsed=...` for the task.

**Phase 4 — One training step (1–2 h).**
Launch slime with the custom generate function (below), `--rollout-batch-size 1`,
`--n-samples-per-prompt 4`, 1 train step. Confirm loss-masked `Sample`s flow to the trainer and a
gradient step runs.
✅ Exit: a training step completes; reward logged; no shape/assertion errors.

**Phase 5 — Minimal learning signal (optional, longer).**
Run ~20–50 steps on a handful of tasks; watch mean reward trend upward. That's the "it works" moment.

---

## 6. Launch command

```bash
# --- sandbox / adapter env ---
export SWE_AGENT=claude_code                 # picks ClaudeCodeHarness + AnthropicAdapter
export AGENTCORE_REGION=us-west-2
export ADAPTER_PUBLIC_HOST=<host sandbox can reach>   # REQUIRED (generate.py:144-149)
export ADAPTER_PORT=18001
export SLIME_AGENT_NODE_TARBALL=/host/node22.tgz      # or enable npm in-sandbox
export SLIME_AGENT_CC_TARBALL=/host/claude-code.tgz
# + AWS creds with Code Interpreter IAM actions

python train.py \
  --hf-checkpoint /path/to/policy \
  --prompt-data one_task.jsonl \
  --rollout-batch-size 1 --n-samples-per-prompt 4 \
  --custom-generate-function-path agentcore_slime_demo.generate.generate \
  --rm-type none \
  ... your usual megatron / GRPO / optim args ...
```

(`--custom-generate-function-path` dispatch: `sglang_rollout.py:283-294`. Reward is filled by
`swe.run_evaluation` inside `generate`, so `--rm-type none`.)

---

## 7. Success criteria

- **Phase 3:** a real `reward` (0 or 1) computed from tests running *in the AgentCore sandbox*, via
  Claude Code driven by the slime policy.
- **Phase 4:** loss-masked `Sample`s (tool/observation tokens masked, `loss_mask=0`) pass
  `_convert_samples_to_train_data` (`ray/rollout.py:306-354`) and a train step runs.
- **Phase 5:** mean reward trends up — end-to-end RL loop confirmed.
- Each rollout = one isolated AgentCore session (distinct `runtimeSessionId`/session name).

---

## 8. Risks & mitigations (kept honest)

| Risk | Why | Mitigation |
|------|-----|-----------|
| Sandbox can't dial back to adapter | AgentCore microVM egress + private slime host | §3 Option A (public addr/tunnel) or Option B (VPC) |
| Claude Code needs a full Linux + Node runtime | Code Interpreter is code-snippet oriented | pre-upload Node/CC tarballs and install via `sb.exec`; or Option B container |
| Session-start + CLI-install latency per rollout | slows throughput | pool sessions / Option B baked image; raise `SWE_BOOT_CONCURRENCY` |
| Concurrent-session quotas (not GPUs) | AgentCore service limits | request quota increase before scaling fan-out |
| API result field shapes drift | `executeCode`/`writeFiles` shapes from samples, not a formal schema | pin SDK version; Phase 0 smoke test |
| Long agent runs hit 8 h serverless cap | big tasks | keep tasks small; use runtime *instances* (14 d) if needed |

---

## 9. Milestone checklist

- [ ] P0 `AgentCoreSandbox` passes `exec/write_file/read_file` smoke test
- [ ] P1 Claude Code installs & launches headless inside the AgentCore sandbox
- [ ] P2 sandbox reaches slime adapter; one turn round-trips to SGLang
- [ ] P3 one eval rollout prints a real reward from in-sandbox tests
- [ ] P4 one training step completes on loss-masked Samples
- [ ] P5 mean reward trends up over ~20–50 steps

---

## 10. Reference paper (resolved)

Confirmed as **Agent Lightning v1.0: Towards Harnessed Agentic RL** (Microsoft), arXiv
**2608.17528** — downloaded, converted to
[`agent-lightning-v1-harnessed-agentic-rl.md`](./agent-lightning-v1-harnessed-agentic-rl.md) (the
original 2508.03680 is in [`agent-lightning-original.md`](./agent-lightning-original.md)); repo
cloned at `repos/agent-lightning`. Terminology and the harness/gateway/trainer split are reconciled
in §1a above.

---

## 11. Implemented code (`../code/`)

A runnable implementation lives under [`../code/agentcore_slime_demo/`](../code/). Two paths, same
loop and same decoupling:

| Path | Entry | Policy | Sandbox | Runs offline? |
|------|-------|--------|---------|---------------|
| **Offline demo** | `scripts/smoke.py` → `harness.run_episode` | `MockPolicy` (scripted buggy→fixed) | `LocalSandbox` | ✅ yes — no AWS/GPU/slime |
| **slime training (CPU Code-Interpreter sandbox)** | `generate.generate` | `SGLangPolicy` | `AgentCoreSandbox` | needs slime + AWS |
| **slime training (remote GPU sandbox Runtime)** | `generate.generate` (`SANDBOX=agentcore-runtime`) | `SGLangPolicy` | `AgentCoreRuntimeSandbox` | needs slime + AWS + g6 |

- **Multi-turn task:** iterative code-repair — write `solution.py` → run checker *in the sandbox* →
  read failure → fix → repeat. Reward = checker exit code (`task.py`). Tasks in `data/tasks.jsonl`.
- **`AgentCoreSandbox` is a verified drop-in for slime's `E2BSandbox`:** `tests/test_smoke.py`
  asserts `isinstance(AgentCoreSandbox(), slime.agent.sandbox.Sandbox)` and matching `exec`
  signature, so any slime generate hook can swap the sandbox class unchanged.
- **Live harness — in-process `MiniClaudeHarness`:** the real Claude Code CLI path (which needed
  `useradd`/`setsid`/npm/multi-user, impossible on the restricted **Code Interpreter**, so it
  targeted a custom multi-user container) has been **removed**. The live harness is
  `MiniClaudeHarness` (`mini_agent.py`), run **in-process** next to the trainer — no CLI, no custom
  multi-user container, and no separate harness Runtime. Only the *sandbox* runs as an AgentCore
  Runtime (Code Interpreter for the CPU path, a g6 GPU Runtime for the GPU path). See
  [`harness-sandbox-decoupling.md`](./harness-sandbox-decoupling.md).
- **Verify offline now:** `python3 code/scripts/smoke.py`,
  `python3 code/scripts/recovery_smoke.py`, and
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest code/tests -q` (all pass: 2/2 tasks solved,
  9/9 tests). See [`../code/README.md`](../code/README.md).

## 12. Fault tolerance & observability

Two companion docs make the training loop production-shaped, both grounded in slime code + primary
AWS/SGLang docs:

- **[`fault-recovery.md`](./fault-recovery.md)** — sandboxes die mid-training; `recovery.py` gives
  RPC self-heal, episode restart on a fresh session, and clean ABORTED trajectories the trainer
  drops. Tested offline with injected failures (`scripts/recovery_smoke.py`).
- **[`observability.md`](./observability.md)** — which metrics to collect at each of the five stages
  (**trainer / harness / AgentCore sandbox / SGLang inference / weight sync**), with exact metric
  keys and `file:line` for slime, CloudWatch metric names for AgentCore, and Prometheus names for
  SGLang. Includes a minimal 5-row dashboard.
