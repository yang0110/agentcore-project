#!/usr/bin/env python3
"""2-node NCCL bandwidth probe, one process per GPU (srun -N2 --ntasks-per-node=8, nccl_bw.sbatch).

Two collectives, each timed over --iters after warm-up, at a few sizes:
  * all_reduce over all 16 GPUs: the nccl-tests "bus bandwidth" (algbw x 2(n-1)/n), comparable to nccl-tests;
  * broadcast from rank 0 (the trainer node) to the 8 GPUs of the other node only: the shape of slime's
    weight push (Megatron -> SGLang engines) over the cross-node link.
Prints one `[nccl_bw] k=v ...` line per (op, size); the transport (EFA / sockets) comes from net_env.sh.
"""
from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes-mb", default="64,256,1024")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--label", default=os.environ.get("NET_DESC", "?"))
    a = ap.parse_args()
    rank, world = int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_NTASKS"])
    local = int(os.environ["SLURM_LOCALID"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    per_node = world // 2
    # rank 0 (trainer node, GPU 0) + every rank of node 2: the cross-node broadcast group.
    push = dist.new_group([0] + list(range(per_node, world)))
    for mb in (int(x) for x in a.sizes_mb.split(",")):
        n = mb * 2**20 // 2                                   # bf16 elements
        t = torch.ones(n, dtype=torch.bfloat16, device="cuda")
        for op in ("all_reduce", "push"):
            if op == "push" and 0 < rank < per_node:
                dist.barrier()
                continue
            fn = (lambda: dist.all_reduce(t)) if op == "all_reduce" else (lambda: dist.broadcast(t, 0, group=push))
            for _ in range(2):
                fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(a.iters):
                fn()
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / a.iters
            dist.barrier()
            if rank == 0:
                algbw = mb / 1024 / dt                          # GB/s of payload
                bus = algbw * 2 * (world - 1) / world if op == "all_reduce" else algbw
                print(f"[nccl_bw] net={a.label!r} op={op} size_mb={mb} ms={dt * 1e3:.1f} algbw_GBps={algbw:.2f} "
                      f"busbw_GBps={bus:.2f} ranks={world if op == 'all_reduce' else per_node + 1}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
