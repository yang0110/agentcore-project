"""Dense GPU packing: hold many envs on ONE (or few) GPU EC2 instance.

The problem this solves
-----------------------
Agentic RL scales by fanning out environments. On a GPU box the naive "one resident
CUDA context per env" model does NOT scale: an L40S has ~46 GB, so 100 resident envs =
~460 MB each — you run out of *memory* long before you run out of SM compute. But most
RL env GPU use is **bursty**: the env does CPU/IO work, waits on the policy, and only
briefly touches the GPU (run a kernel, score a reward). So we don't need 100 resident
contexts; we need a queue in front of the GPU.

This module is that queue. It is deliberately backend-agnostic (Agent Lightning
decoupling, arXiv 2608.17528): it never runs CUDA itself. It hands each env a *lease*
carrying the environment variables that pin the env's GPU work to one device with a
memory cap. The env's Sandbox.exec(..., env=lease.env) then runs the actual workload.
So it works identically whether the env is a LocalSandbox (offline, this box) or an
AgentCoreSandbox on a Runtime instance.

Why this is Path A ("DIY dense packing"), not AgentCore's managed model
-----------------------------------------------------------------------
AgentCore maps 1 session -> 1 isolated microVM and does NOT expose GPU-sharing config
(time-slicing / MPS / MIG) across sessions on one instance (see docs/, integration.md).
To pack 100 envs onto one GPU instance with a sharing policy YOU control, you run your
own containers on a GPU EC2 box and multiplex the GPU yourself. This scheduler is the
software half of that; the ops half is the NVIDIA device-plugin time-slicing / MPS
config on the host (see docs/gpu-density.md).

Sharing mechanics on an L40S (Ada Lovelace)
-------------------------------------------
L40S has **no MIG** (that's Ampere/Hopper only), so hard partitioning is off the table.
The two available knobs are:
  * time-slicing  — the GPU round-robins between contexts; oversubscribe freely, but
                    there is NO memory isolation, so a runaway env can OOM neighbours.
  * CUDA MPS      — processes share one context concurrently (better small-kernel util).
Neither isolates memory, so we enforce a per-env cap in software (see `mem_fraction`
below and `set_per_process_memory_fraction` in the task workload).

Designed for MIXED / unprofiled workloads: gate concurrent GPU work with a small
semaphore, cap per-env memory, and MEASURE (peak concurrency, admission waits) so you
can tune `max_concurrent_gpu` up or down once you see real usage.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Callable

from .sandbox import ExecResult, FileContent, Sandbox


def probe_gpus() -> list[int]:
    """Total memory (MiB) per visible GPU via nvidia-smi; [] if none/no driver."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    mib: list[int] = []
    for ln in out.stdout.splitlines():
        ln = ln.strip()
        if ln:
            with contextlib.suppress(ValueError):
                mib.append(int(float(ln)))
    return mib


@dataclass
class GpuLease:
    """One env's grant of GPU time on a specific device, with a memory cap.

    `env` is meant to be passed straight to `Sandbox.exec(cmd, env=lease.env)`:
      * CUDA_VISIBLE_DEVICES  pins the workload to the assigned physical GPU.
      * ENV_GPU_MEM_FRACTION  the fraction of that GPU the env may allocate; the task
                              workload calls torch.cuda.set_per_process_memory_fraction
                              with it (software cap — see module docstring on why).
    """

    device: int
    mem_fraction: float
    waited_s: float

    @property
    def env(self) -> dict[str, str]:
        return {
            "CUDA_VISIBLE_DEVICES": str(self.device),
            "ENV_GPU_MEM_FRACTION": f"{self.mem_fraction:.4f}",
        }


@dataclass
class PoolStats:
    admitted: int = 0
    peak_concurrent: int = 0
    total_wait_s: float = 0.0
    max_wait_s: float = 0.0
    per_device_admits: dict[int, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "admitted": self.admitted,
            "peak_concurrent_gpu": self.peak_concurrent,
            "avg_admission_wait_s": round(self.total_wait_s / self.admitted, 4) if self.admitted else 0.0,
            "max_admission_wait_s": round(self.max_wait_s, 4),
            "per_device_admits": dict(sorted(self.per_device_admits.items())),
        }


class GpuPool:
    """Admission control + device assignment so N envs share M GPUs on one instance.

    - `max_concurrent_gpu` bounds how many envs run GPU work AT ONCE (the key knob that
      makes 100 bursty envs fit on a few GPUs). Envs beyond it queue on the semaphore.
    - devices are assigned to minimise in-flight leases per device (least-loaded), so
      concurrent work spreads across all GPUs on the box.
    - `per_env_mem_mib` -> a memory fraction per device, enforced by the workload.

    Nothing here imports torch; it only schedules. Acquire a slot with `async with
    pool.slot() as lease:` and hand `lease.env` to the sandbox exec that touches the GPU.
    """

    def __init__(
        self,
        *,
        num_gpus: int | None = None,
        devices: list[int] | None = None,
        gpu_mem_mib: int | None = None,
        max_concurrent_gpu: int = 8,
        per_env_mem_mib: int = 1024,
    ) -> None:
        probed = probe_gpus()
        # `devices` = the PHYSICAL GPU ids this pool may lease (e.g. [6, 7] when the trainer
        # and SGLang engines own 0-5 in the disaggregated/async layout). Leases pin
        # CUDA_VISIBLE_DEVICES to these absolute ids, so envs land only on the GPUs reserved
        # for them. If omitted, the pool owns the first `num_gpus` devices (0..n-1).
        if devices is not None:
            if not devices:
                raise ValueError("devices must be a non-empty list of physical GPU ids")
            self._devices = list(devices)
            self.num_gpus = len(self._devices)
        else:
            self.num_gpus = num_gpus if num_gpus is not None else (len(probed) or 1)
            self._devices = list(range(self.num_gpus))
        self.gpu_mem_mib = gpu_mem_mib if gpu_mem_mib is not None else (probed[0] if probed else 46068)
        if max_concurrent_gpu < 1:
            raise ValueError("max_concurrent_gpu must be >= 1")
        self.max_concurrent_gpu = max_concurrent_gpu
        self.per_env_mem_mib = per_env_mem_mib
        # Fraction of ONE device each env may use. Bounded to (0, 1]. The safety invariant
        # is: at most `ceil(max_concurrent_gpu / num_gpus)` leases can co-reside on a
        # device, so keep per-env fraction * that <= 1 to avoid OOM under time-slicing.
        self.mem_fraction = min(1.0, per_env_mem_mib / self.gpu_mem_mib)
        self._sem = asyncio.Semaphore(max_concurrent_gpu)
        self._inflight = [0] * self.num_gpus   # active leases per device
        self._live = 0                         # currently-admitted GPU workers
        self._lock = asyncio.Lock()
        self.stats = PoolStats()

    def max_leases_per_device(self) -> int:
        return -(-self.max_concurrent_gpu // self.num_gpus)  # ceil

    def memory_safety(self) -> dict:
        """Static check: can the worst-case co-residency fit in one device's memory?"""
        worst = self.max_leases_per_device()
        needed = worst * self.per_env_mem_mib
        return {
            "worst_case_coresident_per_gpu": worst,
            "worst_case_mib_per_gpu": needed,
            "gpu_mem_mib": self.gpu_mem_mib,
            "fits": needed <= self.gpu_mem_mib,
            "headroom_mib": self.gpu_mem_mib - needed,
        }

    @contextlib.asynccontextmanager
    async def slot(self):
        """Acquire a GPU lease (blocks if `max_concurrent_gpu` are already active)."""
        t0 = time.perf_counter()
        await self._sem.acquire()
        waited = time.perf_counter() - t0
        async with self._lock:
            pos = min(range(self.num_gpus), key=lambda d: self._inflight[d])
            physical = self._devices[pos]
            self._inflight[pos] += 1
            self._live += 1
            self.stats.admitted += 1
            self.stats.peak_concurrent = max(self.stats.peak_concurrent, self._live)
            self.stats.total_wait_s += waited
            self.stats.max_wait_s = max(self.stats.max_wait_s, waited)
            self.stats.per_device_admits[physical] = self.stats.per_device_admits.get(physical, 0) + 1
        try:
            yield GpuLease(device=physical, mem_fraction=self.mem_fraction, waited_s=waited)
        finally:
            async with self._lock:
                self._inflight[pos] -= 1
                self._live -= 1
            self._sem.release()

    def describe(self) -> str:
        safety = self.memory_safety()
        return (
            f"GpuPool: {self.num_gpus} GPU(s) x {self.gpu_mem_mib} MiB | "
            f"max_concurrent_gpu={self.max_concurrent_gpu} "
            f"(<= {self.max_leases_per_device()}/device) | "
            f"per_env={self.per_env_mem_mib} MiB (frac={self.mem_fraction:.3f}) | "
            f"worst-case {safety['worst_case_mib_per_gpu']} MiB/GPU -> "
            f"{'FITS' if safety['fits'] else 'OVERCOMMIT'} "
            f"(headroom {safety['headroom_mib']} MiB)"
        )

    @classmethod
    def from_env(cls) -> "GpuPool":
        """Build from env vars so the rollout path can opt in without code changes:
        GPU_POOL_CONCURRENCY (default 8), GPU_POOL_PER_ENV_MIB (default 1024),
        GPU_POOL_DEVICES (comma-separated physical GPU ids the envs may use, e.g. "6,7";
        unset => the pool owns every probed GPU)."""
        raw = os.environ.get("GPU_POOL_DEVICES", "").strip()
        devices = [int(x) for x in raw.split(",") if x.strip()] or None
        return cls(
            devices=devices,
            max_concurrent_gpu=int(os.environ.get("GPU_POOL_CONCURRENCY", "8")),
            per_env_mem_mib=int(os.environ.get("GPU_POOL_PER_ENV_MIB", "1024")),
        )


# Tokens that mark a command as GPU-touching. Deliberately conservative: we would rather
# gate a CPU command occasionally than let an ungated CUDA command race for memory.
_GPU_TOKENS = re.compile(
    r"\b(nvidia-smi|nvcc|cuda|\.cuda\(|device\s*=\s*['\"]cuda|torch\.cuda|cupy|"
    r"tensorflow|jax\.|--gpus?\b|CUDA_VISIBLE_DEVICES)\b",
    re.IGNORECASE,
)


def gpu_command_heuristic(cmd: str, env: dict[str, str] | None) -> bool:
    """Best-effort: does this exec touch the GPU? Matches common CUDA/torch/tooling
    tokens, or an env that already requests a specific device. For workloads the heuristic
    can't see (e.g. `python3 run_checks.py` that imports torch internally), use
    `GpuGatedSandbox(..., gate_all=True)` or pass `always_gate` substrings instead."""
    if env and env.get("CUDA_VISIBLE_DEVICES"):
        return True
    return bool(_GPU_TOKENS.search(cmd or ""))


class GpuGatedSandbox:
    """Transparent Sandbox decorator: routes GPU-touching `exec` calls through a GpuPool.

    Wrap ANY Sandbox (LocalSandbox / AgentCoreSandbox) with this and hand the wrapper to
    the harness. Every `exec` the harness (or slime's ClaudeCodeHarness) makes is inspected;
    if it looks GPU-bound, the call blocks on `pool.slot()`, merges the lease's env
    (CUDA_VISIBLE_DEVICES + ENV_GPU_MEM_FRACTION) into the exec env, and only then runs.
    Non-GPU calls pass straight through with no gating overhead. `write_file`/`read_file`/
    lifecycle all delegate unchanged, so the wrapper is a drop-in `Sandbox`.

    Modes:
      * default            — gate by `gpu_command_heuristic`.
      * gate_all=True      — gate EVERY exec (use for a known-GPU code-repair task whose
                             single checker exec runs torch internally).
      * always_gate=(...)  — extra substrings that force gating (e.g. "run_checks.py").
      * predicate=fn       — supply your own (cmd, env) -> bool.
    """

    def __init__(
        self,
        inner: Sandbox,
        pool: GpuPool,
        *,
        predicate: Callable[[str, dict[str, str] | None], bool] | None = None,
        gate_all: bool = False,
        always_gate: tuple[str, ...] = (),
    ) -> None:
        self._inner = inner
        self._pool = pool
        self._predicate = predicate or gpu_command_heuristic
        self._gate_all = gate_all
        self._always_gate = always_gate

    # ---- Sandbox protocol: identity + lifecycle delegate to the wrapped sandbox --------
    @property
    def sandbox_id(self) -> str:
        return self._inner.sandbox_id

    async def __aenter__(self) -> "GpuGatedSandbox":
        await self._inner.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self._inner.__aexit__(exc_type, exc, tb)

    def _is_gpu(self, cmd: str, env: dict[str, str] | None) -> bool:
        if self._gate_all:
            return True
        if any(s in (cmd or "") for s in self._always_gate):
            return True
        return self._predicate(cmd, env)

    async def exec(
        self, cmd: str, *, user: str = "root", env: dict[str, str] | None = None,
        timeout: int = 120, check: bool = False, idempotent: bool = True,
    ) -> ExecResult:
        if not self._is_gpu(cmd, env):
            return await self._inner.exec(
                cmd, user=user, env=env, timeout=timeout, check=check, idempotent=idempotent
            )
        async with self._pool.slot() as lease:
            merged = {**(env or {}), **lease.env}   # lease pins device + memory cap
            return await self._inner.exec(
                cmd, user=user, env=merged, timeout=timeout, check=check, idempotent=idempotent
            )

    async def write_file(self, sandbox_path: str, content: FileContent, *, user: str = "root") -> None:
        await self._inner.write_file(sandbox_path, content, user=user)

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str:
        return await self._inner.read_file(sandbox_path, user=user)
