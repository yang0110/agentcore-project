#!/usr/bin/env python3
"""Pack N envs onto one (or few) GPU instance and prove the GPU sharing holds.

This is the runnable half of Path A ("DIY dense packing"): it launches N concurrent
envs — each its own Sandbox — but funnels their GPU work through a single `GpuPool`
that (a) bounds how many envs touch the GPU at once, (b) round-robins them across the
box's GPUs, and (c) caps each env's memory. It models the MIXED/bursty case: every env
does a CPU phase (prep / "waiting on the policy") and then a short GPU phase.

On this box (8x L40S) the GPU phase runs REAL CUDA via torch. With no GPU it degrades
to a CPU matmul and says so. Either way it prints the packing stats you tune against:
peak concurrent GPU workers, per-device spread, admission waits, and the static memory
safety check.

    python3 code/scripts/gpu_density.py                        # 100 envs, gate=8
    python3 code/scripts/gpu_density.py --envs 100 --gpu-concurrency 16
    python3 code/scripts/gpu_density.py --envs 100 --per-env-mem-mib 2048

Key knobs:
  --envs             how many envs to hold at once (the fan-out you asked about: 100)
  --gpu-concurrency  max envs running GPU work simultaneously (bursty -> keep this small)
  --per-env-mem-mib  per-env GPU memory cap; worst-case co-residency must fit one GPU
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentcore_slime_demo.gpu_pool import GpuPool
from agentcore_slime_demo.sandbox import make_sandbox

# Runs INSIDE each env's sandbox during its GPU phase. Honours the lease: it is pinned to
# one device by CUDA_VISIBLE_DEVICES and self-limits memory to ENV_GPU_MEM_FRACTION. This
# is the software memory cap the module docstring calls for (L40S has no MIG isolation).
WORKLOAD = r"""
import json, os
frac = float(os.environ.get("ENV_GPU_MEM_FRACTION", "0.05"))
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(min(frac, 1.0), 0)
        dev = torch.device("cuda:0")   # cuda:0 == the ONE device the lease exposed
        a = torch.randn(512, 512, device=dev)
        b = torch.randn(512, 512, device=dev)
        for _ in range(3):
            a = (a @ b).relu()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated(0) // (1024 * 1024)
        print(json.dumps({"ok": True, "device": torch.cuda.get_device_name(0),
                          "visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
                          "peak_mib": peak, "backend": "cuda"}))
    else:
        import torch
        a = torch.randn(512, 512); b = torch.randn(512, 512)
        _ = a @ b
        print(json.dumps({"ok": True, "device": "cpu", "visible": None,
                          "peak_mib": 0, "backend": "cpu-fallback"}))
except ImportError:
    print(json.dumps({"ok": True, "device": "none", "visible": None,
                      "peak_mib": 0, "backend": "no-torch"}))
"""


async def run_env(idx: int, pool: GpuPool) -> dict:
    async with make_sandbox("local") as sb:
        # CPU/prep phase — in real RL this is workspace setup + waiting on the policy.
        await sb.exec("true")
        # GPU phase — admission-controlled by the pool.
        async with pool.slot() as lease:
            await sb.write_file("work.py", WORKLOAD)
            code, out, err = await sb.exec("python3 work.py", env=lease.env, timeout=180)
        try:
            res = json.loads(out.strip().splitlines()[-1]) if out.strip() else {"ok": False}
        except (ValueError, IndexError):
            res = {"ok": False, "raw": out[-200:], "err": err[-200:]}
        res.update(env_idx=idx, exit=code, waited_s=round(lease.waited_s, 4),
                   device_assigned=lease.device)
        return res


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=100)
    ap.add_argument("--gpu-concurrency", type=int, default=8)
    ap.add_argument("--per-env-mem-mib", type=int, default=1024)
    ap.add_argument("--num-gpus", type=int, default=None, help="override probed GPU count")
    args = ap.parse_args()

    pool = GpuPool(
        num_gpus=args.num_gpus,
        max_concurrent_gpu=args.gpu_concurrency,
        per_env_mem_mib=args.per_env_mem_mib,
    )
    print("=" * 78)
    print(f"Dense GPU packing: {args.envs} envs on one instance")
    print("=" * 78)
    print("  " + pool.describe())
    safety = pool.memory_safety()
    if not safety["fits"]:
        print(f"  WARNING: worst-case {safety['worst_case_mib_per_gpu']} MiB/GPU exceeds "
              f"{safety['gpu_mem_mib']} MiB — lower --gpu-concurrency or --per-env-mem-mib, "
              f"or add GPUs. (Relying on bursty timing to avoid OOM is unsafe.)")

    t0 = time.perf_counter()
    results = await asyncio.gather(*(run_env(i, pool) for i in range(args.envs)))
    elapsed = time.perf_counter() - t0

    ok = [r for r in results if r.get("ok")]
    backends = {}
    peak_mib = 0
    for r in ok:
        backends[r.get("backend", "?")] = backends.get(r.get("backend", "?"), 0) + 1
        peak_mib = max(peak_mib, r.get("peak_mib", 0))

    print(f"\n  ran {len(ok)}/{args.envs} envs OK in {elapsed:.1f}s")
    print(f"  backends: {backends}")
    print(f"  observed peak GPU memory allocated by a single env: {peak_mib} MiB "
          f"(cap was {args.per_env_mem_mib} MiB)")
    print(f"  packing stats: {json.dumps(pool.stats.as_dict())}")
    print(f"\n  interpretation:")
    print(f"    - {args.envs} envs coexisted; at most {pool.stats.peak_concurrent} touched a "
          f"GPU at once (gate={args.gpu_concurrency}).")
    print(f"    - the other {args.envs - pool.stats.peak_concurrent} were in their CPU phase "
          f"or queued — this is why 100 bursty envs fit on few GPUs.")
    print(f"    - GPU work spread across devices: {pool.stats.as_dict()['per_device_admits']}")

    failed = [r for r in results if not r.get("ok")]
    if failed:
        print(f"\n  {len(failed)} env(s) failed; first: {failed[0]}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
