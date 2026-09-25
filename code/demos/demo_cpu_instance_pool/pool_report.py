#!/usr/bin/env python3
"""Summarise a demo_cpu_instance_pool run log: how the warm pool was used.

Reads the per-episode ``[demo]`` lines (pool_slot / instance / open_ms / cold / reward, emitted by the
harness from RayFarmRuntimeSandbox.demo_fields) and the per-op ``[sbx-op]`` lines, and prints:

  * episodes per pool member and per instance (boot_id) — the spread across the pool,
  * cold boots (opens >= COLD_BOOT_MS, default 5000 ms) — ideally one per pool member,
  * warm-op latency p50 / p95 (every op under COLD_BOOT_MS) and the cold-open latencies,
  * mean reward per pool member (isolation check: no member should look systematically different).

    python3 pool_report.py logs/<tag>.log
"""
from __future__ import annotations

import os
import re
import statistics
import sys
from collections import Counter, defaultdict

COLD_BOOT_MS = float(os.environ.get("COLD_BOOT_MS", "5000"))
KV = re.compile(r"(\w+)=(\S+)")


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else float("nan")


def main(path: str) -> int:
    eps, ops = [], []
    with open(path, errors="replace") as f:
        for line in f:
            i = line.find("[demo] ")
            if i >= 0 and "pool_slot=" in line:
                eps.append(dict(KV.findall(line[i + 7:])))
                continue
            i = line.find("[sbx-op] ")
            if i >= 0:
                kv = dict(KV.findall(line[i + 9:]))
                if "ms" in kv:
                    ops.append((kv.get("op", "?"), float(kv["ms"]), kv.get("status", "?")))
    if not eps:
        print(f"no pool [demo] lines in {path} (was SANDBOX_POOL set?)")
        return 1

    by_slot = Counter(e["pool_slot"] for e in eps)
    by_inst = Counter(e.get("instance", "?") for e in eps)
    slot_inst = defaultdict(set)
    rew = defaultdict(list)
    for e in eps:
        slot_inst[e["pool_slot"]].add(e.get("instance", "?"))
        rew[e["pool_slot"]].append(float(e.get("reward", 0)))
    cold = [float(e["open_ms"]) for e in eps if e.get("cold") == "1"]
    warm = [ms for _, ms, st in ops if ms < COLD_BOOT_MS and st == "ok"]

    print(f"episodes: {len(eps)}   pool_size: {eps[0].get('pool_size', '?')}")
    for s in sorted(by_slot):
        r = rew[s]
        print(f"  {s}: {by_slot[s]:4d} episodes  instances={sorted(slot_inst[s])}  "
              f"mean_reward={statistics.mean(r):.3f}")
    print(f"distinct instances (boot_id): {len(by_inst)}  {dict(by_inst)}")
    print(f"cold boots (open >= {COLD_BOOT_MS:.0f} ms): {len(cold)}  "
          f"{', '.join(f'{c / 1000:.1f}s' for c in cold)}")
    print(f"warm ops: {len(warm)}  p50={pct(warm, 0.5):.0f} ms  p95={pct(warm, 0.95):.0f} ms")
    errs = Counter(st for _, _, st in ops if st != "ok")
    if errs:
        print(f"op errors: {dict(errs)}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
