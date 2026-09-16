# agentcore_slime_demo — multi-turn agentic RL: slime × AgentCore sandbox × pluggable harness

A minimal, **runnable** demo of using **AWS Bedrock AgentCore Runtime as the execution
sandbox** for agentic RL with **slime** (THUDM/slime) as the trainer, structured on the
**Agent Lightning v1.0** design principles ("Towards Harnessed Agentic RL", arXiv 2608.17528).

- **Task (multi-turn):** iterative code-repair — the agent writes `solution.py`, runs a
  checker *in the sandbox*, reads the failure, fixes it, and repeats until tests pass.
- **Sandbox:** AgentCore **Code Interpreter** microVM (`AgentCoreSandbox`), with a
  `LocalSandbox` fallback so the whole loop runs **offline, with no AWS and no GPU**.
- **Reward:** deterministic — the checker's exit code (1.0 iff all checks pass).

## Why this shape (Agent Lightning principles)

| AL v1.0 component | Role here | Where |
|---|---|---|
| **Trainer** (verl+vLLM → here slime+SGLang+Megatron) | updates the policy from samples | slime, via `generate.py` |
| **Rollout Controller** + real harness | runs the agent, keeps tools/env in the loop | `harness.py` (`run_episode`) |
| **API Gateway** (proxies model calls, captures tokens) | returns token ids + logprobs, no retokenization drift | `policy.py` (`SGLangPolicy`) |
| Unified per-turn **trajectory / MDP** | one transition per turn, only actions trained | `harness.py` `Turn` → `generate.py` `build_token_stream` |

Execution is **fully decoupled** from training: the harness only produces a `Trajectory`;
turning it into loss-masked slime `Sample`s happens separately in `generate.py`.

## Layout

```
code/
  agentcore_slime_demo/
    sandbox.py    # Sandbox Protocol; LocalSandbox (offline) + AgentCoreSandbox (AWS)
    task.py       # CodingTask: prompt, sandbox-run checker, exit-code reward
    policy.py     # Policy: MockPolicy (offline) + SGLangPolicy (slime router)
    harness.py    # run_episode: the multi-turn write->run->observe->fix loop
    generate.py   # slime entry (simplified harness): loss-masked token stream
    mini_agent.py # MiniClaudeHarness: our ~1-file Claude-Code-like agent loop (A3/B3 harness)
    recovery.py   # fault tolerance: RPC self-heal, resilient_episode, clean abort, FlakySandbox
    profiles.py   # SandboxProfile: the two demo cases (CPU_SANDBOX vs GPU_SANDBOX provisioning)
    data/tasks.jsonl        # CPU code-repair tasks
    data/gpu_tasks.jsonl    # GPU task (needs CUDA inside the sandbox)
  scripts/smoke.py           # offline end-to-end run
  scripts/demo_cases.py      # CPU-sandbox vs GPU-supported-Runtime demo cases (both offline)
  scripts/recovery_smoke.py  # offline fault-recovery demo (injected failures)
  tests/test_smoke.py
  requirements.txt
```

## Run it offline (no AWS, no GPU, no slime)

```bash
python3 code/scripts/smoke.py          # full multi-turn loop, LocalSandbox + MockPolicy
python3 -m pytest code/tests -q        # or: python3 code/tests/test_smoke.py
```

Expected: each task fails on turn 0 (buggy attempt), passes on turn 1, `solved=True reward=1.0`.

## Two demo cases: CPU sandbox vs GPU-supported Runtime

```bash
python3 code/scripts/demo_cases.py          # both cases, offline
python3 code/scripts/demo_cases.py --case gpu   # just the GPU case (uses this host's GPU if present)
```

- **CPU sandbox** (the RL default): tool actions run on a serverless Code Interpreter microVM; the
  policy's GPUs stay on slime's cluster. Tasks `two_sum`/`factorial`.
- **GPU-supported Runtime**: an AgentCore Runtime instance on a GPU capacity provider (e.g.
  `g6.xlarge`); the task `gpu_vector_add` only passes if a real CUDA GPU is visible *inside* the
  sandbox. Use only when the task itself needs a GPU.

Full write-up + provisioning CLI: [`../docs/demo-cases.md`](../docs/demo-cases.md).

## Fault tolerance (sandboxes die during training)

Sandboxes fail mid-rollout (session timeout, microVM crash, throttling). `recovery.py` keeps the
trainer alive: transient errors self-heal (retry same session), session death restarts the episode
on a fresh session, and unrecoverable failures return a clean **ABORTED** trajectory (reward 0,
dropped by the trainer) instead of raising. Verified offline with injected failures:

```bash
python3 code/scripts/recovery_smoke.py
# healthy→solved, 2 boot fails→solved (3 boots), mid-episode death→solved (2 boots), fatal→aborted
```

Grounded in AgentCore's real recovery semantics (session auto-reprovisions on next invoke; /ping
health; 409/429 retry) and slime's shipped retry/abort machinery — see
[`../docs/fault-recovery.md`](../docs/fault-recovery.md).

## Observability (what to watch across all five stages)

[`../docs/observability.md`](../docs/observability.md) details the metrics to collect at each stage
of the loop — **trainer / harness / AgentCore sandbox / SGLang inference / weight sync** — with exact
slime metric keys (`file:line`), AgentCore CloudWatch metric names, and SGLang Prometheus names, plus
a minimal 5-row dashboard. Top cross-stage KPIs: `train/rollout_logprob_abs_diff` (weight-sync
correctness), `perf/wait_time_ratio` (pipeline balance), `rollout/zero_std/count_*` (wasted GRPO
advantage), and the ABORTED-sample rate (sandbox health).

## Switch to the real AgentCore sandbox

Same loop, one env var (the harness never knows which sandbox it holds):

```bash
pip install bedrock-agentcore boto3
export SANDBOX=agentcore AGENTCORE_REGION=us-west-2   # + AWS creds w/ Code Interpreter IAM
python3 code/scripts/smoke.py                          # now runs the checker in an AWS microVM
```

## Wire into slime for training

```bash
export SGLANG_ROUTER_URL=http://<router-ip>:<port>
export POLICY_MODEL_PATH=/path/to/policy
export SANDBOX=agentcore AGENTCORE_REGION=us-west-2

python train.py \
  --hf-checkpoint /path/to/policy \
  --prompt-data code/agentcore_slime_demo/data/tasks.jsonl \
  --custom-generate-function-path agentcore_slime_demo.generate.generate \
  --rollout-batch-size 1 --n-samples-per-prompt 4 --rm-type none \
  ...usual megatron/GRPO args...
```

`generate` runs one episode in the sandbox, flattens the trajectory into `(tokens, loss_mask)`
(assistant tokens trainable, prompt+observation masked) and attaches `reward`. Dispatch point:
`slime/rollout/sglang_rollout.py:283-294`.

## Notes / honesty

- `MockPolicy` is scripted (buggy→fixed) purely to make the offline demo deterministic and
  genuinely multi-turn; it does no reasoning. Real learning comes from `SGLangPolicy`.
- `AgentCoreSandbox` field shapes (`structuredContent.stdout/exitCode`, `writeFiles`/`readFiles`)
  follow the AWS samples repo, not a formal schema — pin your SDK version and keep the Phase-0
  smoke test. See `../docs/agentcore-slime-integration.md` §4 and `../docs/demo-plan.md`.
- **Mini Claude-Code harness (in-process):** `MiniClaudeHarness` (`mini_agent.py`) is our ~1-file
  Claude-Code-like agent loop — call the model through slime's `AnthropicAdapter`, pick a tool, run
  it in the sandbox, repeat. It runs **in-process** next to the trainer and talks to whichever
  AgentCore sandbox the demo uses (a CPU Code-Interpreter microVM, or a GPU Runtime instance —
  `AgentCoreSandbox` / `AgentCoreRuntimeSandbox`). `tests/test_smoke.py` proves `AgentCoreSandbox` is
  a genuine drop-in via `isinstance(..., slime.agent.sandbox.Sandbox)`. (Earlier variants hosted the
  harness on its *own* AgentCore Runtime, and before that ran the real Claude Code CLI as the
  harness; both paths have been removed in favor of the in-process mini harness — see
  `../docs/harness-sandbox-decoupling.md`.) This demo's `run_episode` is the offline-runnable,
  simplified stand-in for the agent loop.
