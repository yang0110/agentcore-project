#!/usr/bin/env python3
"""Prove Demo B4's premise offline: N concurrent sandboxes SHARE one GPU via Ray fractions.

This is the runnable, slime-free AND AWS-free half of Demo B4: it exercises the SAME server-side
farm (``sandbox_ray_farm.RayGpuFarm``) that runs INSIDE the g6 sandbox Runtime #2 in the real demo,
just on a local Ray cluster instead of the g6. It starts a small Ray cluster, builds the farm (each
sandbox = a Ray actor holding ``--gpu-fraction`` of a GPU), then fans out ``--episodes`` concurrent
episodes through it. Each episode opens a sub-session, writes a tiny torch workload, and runs it on
the shared GPU — so you can watch several sandboxes coexist on ONE physical GPU, each self-capped to
its memory fraction, exactly as the g6 does when several B4 rollouts share its GPU.

On a GPU box it runs REAL CUDA; with no GPU (or no torch) it degrades and says so. Either way it
prints the packing you tune against: how many sandboxes fit per GPU, checkout waits, peak memory.

    # 8 episodes through 4 sandboxes sharing 1 GPU (frac=0.25):
    python3 code/scripts/ray_farm_smoke.py --episodes 8 --gpu-fraction 0.25 --farm-gpus 1
    # 12 episodes through a 2-GPU farm at frac=0.5 (=> 4 sandboxes):
    python3 code/scripts/ray_farm_smoke.py --episodes 12 --gpu-fraction 0.5 --farm-gpus 2

Knobs mirror the B4 g6-baked env vars (B4_GPU_FRACTION / B4_FARM_GPUS / B4_MAX_SANDBOXES).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentcore_slime_demo.sandbox_ray_farm import RayGpuFarm

# Runs inside each leased sandbox. sitecustomize (injected by the actor) has already applied the
# ENV_GPU_MEM_FRACTION cap, so this just does real GPU work and reports what it saw.
WORKLOAD = r"""
import json, os
try:
    import torch
    if torch.cuda.is_available():
        dev = torch.device("cuda:0")   # the ONE device Ray exposed to this sandbox's actor
        a = torch.randn(1024, 1024, device=dev)
        b = torch.randn(1024, 1024, device=dev)
        for _ in range(5):
            a = (a @ b).relu()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated(0) // (1024 * 1024)
        print(json.dumps({"ok": True, "device": torch.cuda.get_device_name(0),
                          "visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
                          "mem_frac": os.environ.get("ENV_GPU_MEM_FRACTION"),
                          "peak_mib": peak, "backend": "cuda"}))
    else:
        a = torch.randn(256, 256); _ = a @ a
        print(json.dumps({"ok": True, "device": "cpu", "backend": "cpu-fallback"}))
except ImportError:
    print(json.dumps({"ok": True, "device": "none", "backend": "no-torch"}))
"""


def run_episode(idx: int, farm: RayGpuFarm) -> dict:
    """One episode: open a sub-session (lease a fractional-GPU actor), run the workload, close it.

    Sync on purpose — RayGpuFarm is the sync, thread-safe server-side API, and episodes here run
    concurrently on a ThreadPoolExecutor (the g6's AgentCore entrypoint is likewise multi-threaded)."""
    sub = f"smoke-{idx}"
    farm.open(sub)
    try:
        farm.write_file(sub, "work.py", WORKLOAD)
        res = farm.exec(sub, "python3 work.py", {}, 180)
    finally:
        farm.close(sub)
    code, out, err = res.get("exit_code", 1), res.get("stdout", ""), res.get("stderr", "")
    try:
        parsed = json.loads(out.strip().splitlines()[-1]) if out.strip() else {"ok": False}
    except (ValueError, IndexError):
        parsed = {"ok": False, "raw": out[-200:], "err": err[-200:]}
    parsed.update(episode=idx, exit=code)
    return parsed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--gpu-fraction", type=float, default=0.25)
    ap.add_argument("--farm-gpus", type=float, default=1.0)
    ap.add_argument("--max-sandboxes", type=int, default=None)
    ap.add_argument("--num-gpus", type=int, default=None,
                    help="GPUs to give the local Ray cluster (default: let Ray autodetect)")
    args = ap.parse_args()

    import ray

    ray.init(num_gpus=args.num_gpus, namespace="b4_farm", ignore_reinit_error=True)

    farm = RayGpuFarm(
        gpu_fraction=args.gpu_fraction,
        farm_gpus=args.farm_gpus,
        max_sandboxes=args.max_sandboxes,
        workspace_root="/tmp/b4-smoke",
    ).start()

    print("=" * 78)
    print(f"Ray fractional-GPU sandbox farm: {args.episodes} episodes")
    print("=" * 78)
    print("  " + farm.describe())

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.episodes) as pool:
        results = list(pool.map(lambda i: run_episode(i, farm), range(args.episodes)))
    elapsed = time.perf_counter() - t0

    ok = [r for r in results if r.get("ok")]
    backends: dict[str, int] = {}
    peak = 0
    for r in ok:
        backends[r.get("backend", "?")] = backends.get(r.get("backend", "?"), 0) + 1
        peak = max(peak, r.get("peak_mib", 0))

    st = farm.stats
    print(f"\n  ran {len(ok)}/{args.episodes} episodes OK in {elapsed:.1f}s")
    print(f"  backends: {backends}")
    print(f"  observed peak GPU memory by a single sandbox: {peak} MiB")
    print(f"  farm stats: checkouts={st.checkouts} max_wait_s={st.max_wait_s:.3f} "
          f"avg_wait_s={(st.total_wait_s / st.checkouts if st.checkouts else 0):.3f}")
    print("\n  interpretation:")
    print(f"    - {farm.max_sandboxes} sandbox(es) shared {farm.farm_gpus:g} GPU(s) at "
          f"frac={args.gpu_fraction:g} (Ray packed floor(1/frac) per GPU).")
    print(f"    - {args.episodes} episodes ran through those {farm.max_sandboxes} slots; extras "
          f"queued on the farm gate (see max_wait_s).")

    failed = [r for r in results if not r.get("ok")]
    if failed:
        print(f"\n  {len(failed)} episode(s) failed; first: {failed[0]}")
    ray.shutdown()
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
