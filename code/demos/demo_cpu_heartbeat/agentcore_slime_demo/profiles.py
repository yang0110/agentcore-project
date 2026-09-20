"""Sandbox provisioning profiles — the two demo cases.

A `SandboxProfile` captures *how a rollout's sandbox is provisioned on AWS AgentCore* and
*why you'd pick it*. It is deliberately declarative: the harness/rollout code is backend-
agnostic (see sandbox.py), so the profile only carries the AgentCore knobs plus the
human-facing rationale and the exact CLI to stand it up. The offline demo attaches a profile
to each run so `scripts/demo_cases.py` can show the two cases side by side without AWS.

Two cases (see docs/demo-cases.md):

  * CPU_SANDBOX — the RL default. Tool actions (write file, run checker/tests) execute on a
    cheap CPU microVM. The policy's GPUs live on slime's training cluster, NOT here. This is
    what you want for almost all agentic-RL coding tasks: the sandbox is I/O + shell, not math.

  * GPU_SANDBOX — an AgentCore Runtime instance backed by a GPU capacity provider (customer
    EC2, e.g. g6.xlarge). Use it only when the *task itself* needs a GPU inside the sandbox —
    e.g. writing/repairing a CUDA kernel or torch code whose reward is "does it run correctly
    on the GPU". The policy can still be served from slime's cluster; this GPU is the agent's
    hands, not its brain.

Grounded AgentCore facts (see memory / AWS "What's New", 2026-08-06 GA of "AgentCore Runtime
instances"): serverless Code Interpreter microVMs are CPU and capped at 8 h sessions; Runtime
instances run agents on customer EC2 via a **capacity provider**, support GPU-accelerated
instance families, and allow sessions up to 14 days. Provisioning CLI:
    agentcore add capacity-provider --instance-types g6.xlarge,g6.2xlarge --name gpu-pool
    agentcore add agent --capacity-provider gpu-pool
Pin your SDK/CLI version and confirm exact flags against your installed `agentcore` — these
are the shapes reported at GA, not a frozen schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SandboxProfile:
    name: str
    compute: str                       # "cpu" | "gpu"
    agentcore_mode: str                # "code_interpreter" | "runtime_instance"
    instance_types: tuple[str, ...] = ()   # empty for serverless CPU
    capacity_provider: str = ""        # required for GPU runtime instances
    max_session: str = ""              # human-readable session cap
    rationale: str = ""                # when to pick this profile
    provision_cli: tuple[str, ...] = field(default_factory=tuple)

    @property
    def needs_gpu(self) -> bool:
        return self.compute == "gpu"

    def summary(self) -> str:
        lines = [
            f"profile        : {self.name}",
            f"compute        : {self.compute}",
            f"agentcore mode : {self.agentcore_mode}",
            f"instance types : {', '.join(self.instance_types) or '(serverless microVM)'}",
            f"max session    : {self.max_session}",
            f"use when       : {self.rationale}",
        ]
        if self.capacity_provider:
            lines.insert(4, f"capacity prov. : {self.capacity_provider}")
        return "\n".join("    " + ln for ln in lines)


# ---- Case 1: CPU sandbox (the RL default) --------------------------------------
CPU_SANDBOX = SandboxProfile(
    name="cpu-sandbox",
    compute="cpu",
    agentcore_mode="code_interpreter",
    instance_types=(),
    max_session="8 h (serverless microVM)",
    rationale=(
        "Default for agentic-RL coding tasks. Tool actions are shell/file I/O and CPU tests; "
        "the policy's GPUs stay on slime's cluster. Cheapest, fastest to boot, highest fan-out."
    ),
    provision_cli=(
        "# Serverless Code Interpreter — no capacity provider, no GPU.",
        "# The AgentCoreSandbox in sandbox.py talks to this directly:",
        "#   CodeInterpreter(region).start(...)  ->  executeCommand / writeFiles / readFiles",
    ),
)

# ---- Case 2: GPU-supported AgentCore Runtime -----------------------------------
GPU_SANDBOX = SandboxProfile(
    name="gpu-sandbox",
    compute="gpu",
    agentcore_mode="runtime_instance",
    instance_types=("g6.xlarge", "g6.2xlarge"),
    capacity_provider="gpu-pool",
    max_session="up to 14 d (Runtime instance on customer EC2)",
    rationale=(
        "Only when the task itself needs a GPU inside the sandbox (CUDA/kernel/torch code whose "
        "reward is 'runs correctly on the GPU'). Backed by a GPU capacity provider on your EC2."
    ),
    provision_cli=(
        "agentcore add capacity-provider --instance-types g6.xlarge,g6.2xlarge --name gpu-pool",
        "agentcore add agent --capacity-provider gpu-pool",
        "# The rollout then targets this agent/runtime; a GPU is visible to `nvidia-smi`",
        "# and torch.cuda inside the session.",
    ),
)

PROFILES = {p.name: p for p in (CPU_SANDBOX, GPU_SANDBOX)}


def get_profile(name: str) -> SandboxProfile:
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(f"unknown profile {name!r}; choose from {sorted(PROFILES)}") from None
