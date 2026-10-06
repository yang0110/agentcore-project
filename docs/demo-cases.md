# Demo cases — CPU-based vs GPU-supported AgentCore Runtime sandbox

> **Historical (2026-09-30).** This doc was written for demos and code dirs that were removed on 2026-09-30 (and the microVM-only `demo_cpu_terminalbench` on 2026-10-06); the only live demo is [`code/demos/demo_terminalbench_kernelbench`](../code/demos/demo_terminalbench_kernelbench/README.md).

Two runnable demo cases that share **one harness and one reward mechanism** and differ only in
how the sandbox is provisioned on AWS AgentCore and what the task demands of it. Both run offline
today (LocalSandbox); both map onto a real AgentCore backend by swapping the sandbox class and
setting `SANDBOX=agentcore`.

| | **Case 1 — CPU sandbox** | **Case 2 — GPU-supported Runtime** |
|---|---|---|
| Compute | CPU microVM | GPU EC2 (e.g. `g6.xlarge`) |
| AgentCore mode | serverless **Code Interpreter** | **Runtime instance** + GPU **capacity provider** |
| Max session | ~8 h (serverless) | up to 14 d (Runtime instance on your EC2) |
| Task | code-repair (`two_sum`, `factorial`) | `gpu_vector_add` — needs CUDA *inside* the sandbox |
| Reward | checker exit code (1.0 iff all checks pass) | same — but a check requires the op ran on `cuda` |
| Where the policy's GPUs are | slime's training cluster (not here) | slime's training cluster (still not here) |
| When to use | **almost always** — tool actions are shell/IO/CPU tests | only when the *task itself* needs a GPU (CUDA/kernel/torch) |

Profiles are defined in [`../code/agentcore_slime_demo/profiles.py`](../code/agentcore_slime_demo/profiles.py)
(`CPU_SANDBOX`, `GPU_SANDBOX`); the driver is
[`../code/scripts/demo_cases.py`](../code/scripts/demo_cases.py).

---

## The key distinction (don't conflate "sandbox GPU" with "policy GPU")

In agentic RL there are two very different places a GPU can live:

- **The policy's GPUs (the agent's *brain*)** — always on slime's SGLang/Megatron cluster. This is
  where training compute goes, and it is *not* in the AgentCore sandbox in either case.
- **The sandbox's GPU (the agent's *hands*)** — only relevant when the *task's reward* depends on
  running GPU code. That is Case 2, and only Case 2.

So **Case 1 is the default** for essentially all coding-agent RL: the sandbox just runs shells and
tests, which are CPU work, and a GPU there would be wasted spend. Reach for **Case 2** only when the
task is itself a GPU task — e.g. "write/repair a CUDA kernel", "make this torch code run on the GPU",
"optimize a fused op" — where the checker can only pass on a GPU.

---

## Case 1 — CPU sandbox (the RL default)

Tool execution runs on a serverless AgentCore **Code Interpreter** microVM — the `AgentCoreSandbox`
in [`sandbox.py`](../code/agentcore_slime_demo/sandbox.py) talks to it directly
(`CodeInterpreter(region).start(...)` → `executeCommand`/`writeFiles`/`readFiles`). No capacity
provider, no GPU, fastest boot, highest fan-out. This is the path the rest of the demo docs assume.

```bash
python3 code/scripts/demo_cases.py --case cpu          # offline
SANDBOX=agentcore python3 code/scripts/demo_cases.py --case cpu   # real AWS microVM
```

Expected: `two_sum` and `factorial` each fail turn 0 (buggy attempt), pass turn 1 — `solved 2/2`.

---

## Case 2 — GPU-supported AgentCore Runtime

Backed by an AgentCore **Runtime instance** on a **GPU capacity provider** (your EC2, GPU-accelerated
family). Grounded fact: "AgentCore Runtime instances" reached GA on **2026-08-06** (AWS "What's New");
they run agents on customer EC2 via a capacity provider, support GPU families, and allow sessions up
to 14 days (vs the ~8 h serverless microVM cap). Provisioning:

```bash
agentcore add capacity-provider --instance-types g6.xlarge,g6.2xlarge --name gpu-pool
agentcore add agent --capacity-provider gpu-pool
# The rollout targets this agent/runtime; nvidia-smi and torch.cuda now see a GPU in-session.
```
(Confirm exact flags against your installed `agentcore` CLI — these are the shapes reported at GA.)

The task `gpu_vector_add` ([`data/gpu_tasks.jsonl`](../code/agentcore_slime_demo/data/gpu_tasks.jsonl))
is a normal code-repair task whose checks are GPU-gated:

1. `torch.cuda.is_available()` — the sandbox must have a CUDA GPU at all,
2. the elementwise-add result is numerically correct,
3. `last_device().startswith('cuda')` — the op **actually ran on the GPU**, not silently on CPU.

The offline MockPolicy replays a buggy CPU-only attempt (fails check 3) then a fixed `.cuda()`
attempt (passes) — so the multi-turn write→run→read-failure→fix loop is genuinely exercised, and on
a GPU host it runs on the real device.

```bash
python3 code/scripts/demo_cases.py --case gpu          # offline; uses this host's GPU if present
SANDBOX=agentcore python3 code/scripts/demo_cases.py --case gpu   # real GPU Runtime instance
```

On a CPU-only host the GPU case reports "no GPU detected" and the task cannot pass — which is the
correct, honest signal that this case *requires* GPU compute the CPU profile does not provision.

---

## Run both

```bash
python3 code/scripts/demo_cases.py        # both cases, prints each profile + per-turn transcript
```

`test_smoke.py` covers both: `test_profiles_describe_two_demo_cases` (the provisioning distinction)
and `test_gpu_task_needs_real_cuda_inside_sandbox` (skips cleanly when no CUDA is present).

---

## Honesty / limits

- The GPU demo proves the *sandbox can host a GPU workload*; it does not claim the policy trains in
  the sandbox. Training stays on slime's cluster in both cases (see [`demo-plan.md`](./demo-plan.md) §1).
- The real Claude Code harness (multi-user CLI) needs a **custom AgentCore Runtime container**, which
  is orthogonal to CPU-vs-GPU: you can build that container on either a CPU or a GPU capacity provider.
- AgentCore CLI/SDK field and flag shapes are pinned to what was reported at GA, not a frozen schema —
  version-pin and keep the Phase-0 smoke test. See [`agentcore-slime-integration.md`](./agentcore-slime-integration.md).
