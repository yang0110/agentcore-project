"""agentcore_slime_demo — multi-turn agentic-RL demo.

slime (trainer) + a pluggable harness + an AgentCore Runtime sandbox, wired per the
Agent Lightning v1.0 design principles (decoupled execution, gateway-captured tokens,
unified per-turn trajectory). Runs fully offline via MockPolicy + LocalSandbox.
"""

from .policy import Completion, MockPolicy, Policy, SGLangPolicy
from .sandbox import AgentCoreSandbox, LocalSandbox, Sandbox, make_sandbox
from .task import CodingTask, load_tasks
from .harness import Trajectory, Turn, run_episode
from .recovery import (
    FatalSandboxError, FlakySandbox, SandboxError, TransientSandboxError,
    is_recoverable, resilient_episode,
)

__all__ = [
    "Completion", "Policy", "MockPolicy", "SGLangPolicy",
    "Sandbox", "LocalSandbox", "AgentCoreSandbox", "make_sandbox",
    "CodingTask", "load_tasks",
    "Trajectory", "Turn", "run_episode",
    "SandboxError", "TransientSandboxError", "FatalSandboxError",
    "is_recoverable", "resilient_episode", "FlakySandbox",
]
