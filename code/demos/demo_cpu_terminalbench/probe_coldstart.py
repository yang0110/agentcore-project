#!/usr/bin/env python3
"""Measure the cold start of EVERY Runtime in the Terminal-Bench pool — and decompose it.

Run this MANUALLY (no training needed) once the pool is deployed. For each task Runtime
(`tasks/arn_by_image.json`) it opens fresh sessions exactly as a training episode does and times:

    open_s        client: first invoke sent -> reply        (the whole cold start an episode pays)
    pull_boot_s   sent -> server process started in the microVM (allocate + image pull + container start)
    ready_s       server started -> first invoke delivered     (platform health check + routing)
    warm_hit      1 if the server was already running BEFORE the request (AgentCore pre-provisioned
                  a microVM for this Runtime), so the session skipped the pull + boot entirely

The server timestamps come from the task image's own acs_http_server (baked by deploy_terminalbench.py);
both clocks are EC2 time-sync'd (ms-level skew). Two rounds:

  round=first  one session per Runtime                 -> "the next episode's" cold start
  round=burst  PROBE_BURST sessions per Runtime at once -> what GRPO does (NS siblings of one task
               start together): shows how many find a pre-provisioned microVM and how many pay a
               full pull + boot

Every session is stopped right after its first ping. Joins tasks/images.csv (compressed image size)
and writes tasks/coldstart.csv plus a summary (overall, warm vs cold, by image-size bucket).

    export AGENTCORE_REGION=us-west-2
    python3 probe_coldstart.py                    # all Runtimes, burst of 4
    PROBE_BURST=8 PROBE_CONC=32 python3 probe_coldstart.py
    TB_TASKS=a,b python3 probe_coldstart.py        # subset;  TB_LIMIT=10 for the first 10
    PROBE_ROUNDS=first python3 probe_coldstart.py  # skip the burst round
"""
from __future__ import annotations

import asyncio
import csv
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # vendored agentcore_slime_demo (this demo's own copy)

from agentcore_slime_demo.agentcore_runtime_sandbox import AgentCoreRuntimeSandbox  # noqa: E402

TASKS_DIR = HERE / "tasks"
POOL_MAP = TASKS_DIR / "arn_by_image.json"
OUT_CSV = TASKS_DIR / "coldstart.csv"
CONC = int(os.environ.get("PROBE_CONC", "16"))
BURST = int(os.environ.get("PROBE_BURST", "4"))
ROUNDS = [r for r in os.environ.get("PROBE_ROUNDS", "first,burst").split(",") if r]


def _load_pool() -> dict[str, str]:
    if not POOL_MAP.exists():
        sys.exit(f"ERROR: {POOL_MAP} not found — run deploy/deploy_terminalbench.py first.")
    pool = json.loads(POOL_MAP.read_text())
    only = {t for t in os.environ.get("TB_TASKS", "").split(",") if t}
    if only:
        pool = {k: v for k, v in pool.items() if k in only}
    limit = int(os.environ.get("TB_LIMIT", "0"))
    if limit:
        pool = dict(list(pool.items())[:limit])
    if not pool:
        sys.exit("ERROR: no Runtimes selected (check the pool map / TB_TASKS / TB_LIMIT).")
    return pool


def _sizes() -> dict[str, float]:
    p = TASKS_DIR / "images.csv"
    if not p.exists():
        return {}
    with p.open() as f:
        return {r["task"]: float(r["size_mb"]) for r in csv.DictReader(f) if r.get("size_mb")}


async def _open_one(task: str, arn: str, region: str, rnd: str, idx: int) -> dict:
    row = {"task": task, "round": rnd, "idx": idx, "status": "ok"}
    sb = AgentCoreRuntimeSandbox(runtime_arn=arn, region=region, workdir="")
    t0 = time.time()
    try:
        await sb.__aenter__()
        boot = (sb.open_reply or {}).get("boot") or {}
        sent, recv = sb.open_sent_epoch, sb.open_recv_epoch
        srv, first = boot.get("server_start_epoch"), boot.get("first_invoke_epoch")
        row["open_s"] = round(recv - sent, 3)
        if srv:
            row["pull_boot_s"] = round(max(0.0, srv - sent), 3)
            row["warm_hit"] = int(srv < sent)
            row["server_age_s"] = round(max(0.0, sent - srv), 1)   # how long the spare had been up
        if srv and first:
            row["ready_s"] = round(first - max(srv, sent), 3)
    except Exception as e:  # noqa: BLE001 — record, keep probing the rest
        row.update(status=f"{type(e).__name__}: {e}"[:160], open_s=round(time.time() - t0, 3))
    finally:
        try:
            await sb.__aexit__(None, None, None)   # stop_runtime_session
        except Exception:  # noqa: BLE001
            pass
    return row


def _pct(xs: list[float], p: float) -> float:
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else float("nan")


def _summ(label: str, rows: list[dict], key: str = "open_s") -> None:
    xs = sorted(r[key] for r in rows if r.get("status") == "ok" and key in r)
    if xs:
        print(f"  {label:<34} n={len(xs):<4} p50={_pct(xs, .5):6.2f}s  p90={_pct(xs, .9):6.2f}s  "
              f"max={xs[-1]:6.2f}s")


async def _main() -> int:
    region = os.environ.get("AGENTCORE_REGION", "us-west-2")
    pool = _load_pool()
    sizes = _sizes()
    sem = asyncio.Semaphore(CONC)
    rows: list[dict] = []

    async def guarded(*a):
        async with sem:
            return await _open_one(*a)

    for rnd in ROUNDS:
        n = 1 if rnd == "first" else BURST
        print(f"round={rnd}: {len(pool)} Runtimes x {n} session(s) (conc={CONC})", flush=True)
        t0 = time.perf_counter()
        if rnd == "first":
            got = await asyncio.gather(*[guarded(t, a, region, rnd, 0) for t, a in pool.items()])
        else:
            # A burst must really be concurrent PER RUNTIME, so each Runtime's n opens share one slot.
            async def burst(t, a):
                async with sem:
                    return await asyncio.gather(*[_open_one(t, a, region, rnd, i) for i in range(n)])
            got = [r for grp in await asyncio.gather(*[burst(t, a) for t, a in pool.items()]) for r in grp]
        rows += got
        print(f"  done in {time.perf_counter() - t0:.0f}s", flush=True)
        await asyncio.sleep(5)

    for r in rows:
        r["size_mb"] = sizes.get(r["task"], "")
    fields = ["task", "round", "idx", "size_mb", "status", "open_s", "pull_boot_s", "ready_s",
              "warm_hit", "server_age_s"]
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r["status"] == "ok"]
    print("\n=== cold-start summary ===")
    print(f"  sessions ok: {len(ok)}/{len(rows)}   failed: {len(rows) - len(ok)}")
    for rnd in ROUNDS:
        rr = [r for r in ok if r["round"] == rnd]
        hits = sum(r.get("warm_hit", 0) for r in rr)
        print(f" round={rnd}: warm_hit {hits}/{len(rr)}")
        _summ("open (all)", rr)
        _summ("open (warm hit)", [r for r in rr if r.get("warm_hit") == 1])
        _summ("open (cold)", [r for r in rr if r.get("warm_hit") == 0])
        _summ("pull+boot (cold)", [r for r in rr if r.get("warm_hit") == 0], "pull_boot_s")
        _summ("ready (server->first invoke)", rr, "ready_s")
    cold = [r for r in ok if r.get("warm_hit") == 0 and isinstance(r["size_mb"], float)]
    if cold:
        print(" cold open by compressed image size:")
        for lo, hi in ((0, 250), (250, 500), (500, 1000), (1000, 2048)):
            _summ(f"{lo}-{hi} MB", [r for r in cold if lo <= r["size_mb"] < hi])
    fails = [r for r in rows if r["status"] != "ok"]
    if fails:
        print(f"  failures: {[(r['task'], r['status'][:60]) for r in fails][:10]}")
    print(f"  per-session rows -> {OUT_CSV}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
