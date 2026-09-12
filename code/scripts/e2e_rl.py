#!/usr/bin/env python3
"""End-to-end agentic-RL rollout driver for both demos — LIVE, with a REAL policy.

This is the "run it for real" entry point behind the two demo cases (see docs/demos-guide.md):

  --profile cpu   Demo A: MBPP coding tasks in an AgentCore Code Interpreter microVM (CPU).
                  The RL default — tool execution is cheap shell/IO; the policy GPUs stay on
                  slime's cluster. Backend from $SANDBOX (agentcore = real AWS, local = offline).
  --profile gpu   Demo B: KernelBench-style CUDA tasks. The checker only passes on a real GPU;
                  it runs on the remote GPU sandbox Runtime (an AgentCore Runtime instance +
                  GPU capacity provider), reached via $SANDBOX=agentcore-runtime.

The policy is REAL by default: LocalHFPolicy runs Qwen3-4B on the GPU via transformers and
returns text + token_ids + logprobs — the exact Completion contract slime training consumes.
(SGLangPolicy is the production path; LocalHFPolicy avoids depending on a running router and
sidesteps this box's sgl_kernel/torch ABI break — a documented pitfall.) Use --policy mock for
a model-free dry run that still exercises the whole write->run->observe->fix loop.

For every solved episode we ALSO build the loss-masked token stream (generate.build_token_stream)
and print its shape — proving each rollout yields real, trainable RL data end-to-end.

Usage:
    python3 code/scripts/e2e_rl.py --profile cpu --limit 3                 # MBPP, real policy, offline
    SANDBOX=agentcore python3 code/scripts/e2e_rl.py --profile cpu --limit 3   # ... on real AWS microVM
    SANDBOX=agentcore-runtime python3 code/scripts/e2e_rl.py --profile gpu  # KernelBench on remote GPU Runtime
    python3 code/scripts/e2e_rl.py --profile cpu --policy mock             # model-free dry run
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentcore_slime_demo.harness import run_episode
from agentcore_slime_demo.sandbox import make_sandbox
from agentcore_slime_demo.task import CodingTask

DEFAULT_MODEL = os.environ.get("POLICY_MODEL_PATH", "Qwen/Qwen3-4B")


def load_profile_tasks(profile: str, limit: int | None) -> list[CodingTask]:
    if profile == "cpu":
        from agentcore_slime_demo.mbpp import load_mbpp
        return load_mbpp("train", limit=limit)
    from agentcore_slime_demo.kernelbench import load_kernelbench_tasks
    tasks = load_kernelbench_tasks()
    return tasks[:limit] if limit else tasks


def build_policy(kind: str, model: str, device: str):
    """A shared policy instance (loads the model ONCE, serves every episode)."""
    if kind == "mock":
        return None  # per-task MockPolicy is built inside the loop from task.mock_attempts
    from agentcore_slime_demo.policy import LocalHFPolicy
    # Qwen3 emits <think> by default; suppress it for tight codegen rollouts.
    return LocalHFPolicy(model, device=device, enable_thinking=False)


async def run_one(task: CodingTask, shared_policy, kind: str, sandbox_kind: str) -> dict:
    from agentcore_slime_demo.policy import MockPolicy

    policy = shared_policy if shared_policy is not None else MockPolicy(task.mock_attempts)
    sampling = {"max_new_tokens": 1024, "temperature": 0.0 if kind == "real" else 1.0}
    t0 = time.perf_counter()
    async with make_sandbox(sandbox_kind) as sb:
        traj = await run_episode(policy, sb, task, sampling_params=sampling)
        sid = sb.sandbox_id
    dt = time.perf_counter() - t0

    # Prove the rollout is trainable: build the loss-masked token stream (real policy only —
    # MockPolicy has no token_ids). Uses the policy's own tokenizer, no retokenization drift.
    n_tok = n_trainable = 0
    if shared_policy is not None and traj.turns and traj.turns[0].completion.token_ids:
        from agentcore_slime_demo.generate import build_token_stream
        tokens, response_length, mask, _lp = build_token_stream(traj, shared_policy.tokenizer)
        n_tok, n_trainable = len(tokens), sum(mask)

    print(f"\n  task {task.task_id}  (sandbox {sid}, {dt:.1f}s)")
    for i, turn in enumerate(traj.turns):
        head = turn.observation.splitlines()[0] if turn.observation else ""
        print(f"    turn {i}: submitted={turn.submitted_code} exit={turn.exit_code} "
              f"gen_tokens={len(turn.completion.token_ids)} | {head[:80]}")
    print(f"    --> solved={traj.solved} reward={traj.reward} turns={traj.num_turns}"
          + (f" | training tokens={n_tok} (trainable={n_trainable})" if n_tok else ""))
    return {"solved": traj.solved, "reward": traj.reward, "tokens": n_tok, "trainable": n_trainable}


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=["cpu", "gpu"], required=True)
    ap.add_argument("--policy", choices=["real", "mock"], default="real")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=3)
    args = ap.parse_args()

    sandbox_kind = os.environ.get("SANDBOX", "local").lower()
    print("=" * 78)
    print(f"E2E agentic RL  |  profile={args.profile}  policy={args.policy}  "
          f"sandbox={sandbox_kind}  model={args.model if args.policy == 'real' else '—'}")
    print("=" * 78)

    tasks = load_profile_tasks(args.profile, args.limit)
    print(f"loaded {len(tasks)} task(s)")

    shared_policy = build_policy(args.policy, args.model, args.device)
    if shared_policy is not None:
        print(f"policy ready: {type(shared_policy).__name__} on {args.device}")

    results = []
    for task in tasks:
        results.append(await run_one(task, shared_policy, args.policy, sandbox_kind))

    solved = sum(r["solved"] for r in results)
    tok = sum(r["tokens"] for r in results)
    trainable = sum(r["trainable"] for r in results)
    print("\n" + "=" * 78)
    print(f"SUMMARY  solved {solved}/{len(results)}  |  avg reward "
          f"{sum(r['reward'] for r in results) / max(len(results), 1):.2f}"
          + (f"  |  training tokens {tok} (trainable {trainable})" if tok else ""))
    print("=" * 78)
    return 0 if solved == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
