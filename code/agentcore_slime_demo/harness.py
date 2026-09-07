"""The harness: the agent loop that ties policy + sandbox + task together.

This plays the role of Agent Lightning's *Rollout Controller* + the agent harness: it runs
the agent, keeps tools/environment in the loop, and records every interaction as training
data. Crucially, the harness is decoupled from training — it only produces a Trajectory of
transitions; converting those into slime training samples happens in generate.py.

Multi-turn loop (one episode = one task):
    turn 0: policy sees system+task  -> emits solution.py -> sandbox runs checker -> observation
    turn 1: policy sees observation  -> fixes solution.py -> sandbox runs checker -> observation
    ... until checker passes or max_turns reached.

Each turn is one MDP transition: (prompt so far) --action(assistant tokens)--> (observation).
Only the assistant tokens are trainable; observation/tool tokens are masked downstream. This
is the "unified data interface" the AL report builds credit assignment on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .policy import Completion, Message, Policy
from .sandbox import Sandbox
from .task import CodingTask

if TYPE_CHECKING:
    from .gpu_pool import GpuPool


@dataclass
class Turn:
    """One MDP transition."""
    prompt_messages: list[Message]      # full context the policy saw (state)
    completion: Completion              # the action (assistant tokens — trainable)
    observation: str                    # environment feedback (masked downstream)
    exit_code: int                      # checker exit code this turn
    submitted_code: bool                # did the agent produce a code block?


@dataclass
class Trajectory:
    task_id: str
    turns: list[Turn] = field(default_factory=list)
    reward: float = 0.0
    solved: bool = False
    # Recovery bookkeeping (see recovery.py). status ∈ {"completed", "aborted"}.
    status: str = "completed"
    abort_reason: str = ""
    boot_attempts: int = 1   # how many sandbox boots this episode consumed

    @property
    def num_turns(self) -> int:
        return len(self.turns)


async def run_episode(
    policy: Policy,
    sandbox: Sandbox,
    task: CodingTask,
    sampling_params: dict | None = None,
    gpu_pool: "GpuPool | None" = None,
) -> Trajectory:
    """Run one multi-turn episode inside an already-open sandbox.

    If `gpu_pool` is given, the checker exec (which for a GPU task runs torch on CUDA) is
    routed through the pool for admission control + a device/memory lease — this is how a
    dense GPU instance holds many concurrent episodes (see gpu_pool.py, docs/gpu-density.md).
    Backend-agnostic: we just wrap the sandbox, so the loop below is unchanged.
    """
    sampling_params = sampling_params or {"max_new_tokens": 1024, "temperature": 1.0}
    if gpu_pool is not None:
        from .gpu_pool import GpuGatedSandbox
        # The checker command is `python3 run_checks.py`; the heuristic can't see the torch
        # import inside it, so gate every exec for a GPU episode (one checker exec per turn).
        sandbox = GpuGatedSandbox(sandbox, gpu_pool, gate_all=True)
    traj = Trajectory(task_id=task.task_id)

    # Write the checker once; it's stable across turns.
    await sandbox.write_file("run_checks.py", task.runner_source())

    messages: list[Message] = [
        {"role": "system", "content": task.system_prompt()},
        {"role": "user", "content": task.initial_user_prompt()},
    ]

    for _ in range(task.max_turns):
        completion = await policy.complete(messages, sampling_params)
        messages.append({"role": "assistant", "content": completion.text})

        code = CodingTask.extract_code(completion.text)
        if code is None:
            # No code this turn — nudge and record a "no submission" transition.
            observation = "No ```python code block found. Reply with the full solution.py in one block."
            exit_code, submitted = 1, False
        else:
            await sandbox.write_file("solution.py", code)
            exit_code, stdout, stderr = await sandbox.exec(
                "python3 run_checks.py", timeout=60
            )
            observation = task.observation(exit_code, stdout, stderr)
            submitted = True

        traj.turns.append(Turn(
            prompt_messages=list(messages[:-1]),   # state before this action
            completion=completion,
            observation=observation,
            exit_code=exit_code,
            submitted_code=submitted,
        ))

        if submitted and exit_code == 0:
            traj.solved = True
            break

        messages.append({"role": "user", "content": observation})

    traj.reward = CodingTask.reward(0 if traj.solved else 1)
    return traj
