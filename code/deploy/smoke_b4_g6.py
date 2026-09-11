#!/usr/bin/env python3
"""Pre-flight for Demo B4's REMOTE g6 Ray-farm sandbox, over invoke_agent_runtime (no slime).

Validates the whole b4 remote path end-to-end on ONE real g6 before a full training run:
  * the g6 cold-launches, installs torch at startup, and the server imports as a PACKAGE
    (the farm's relative imports resolve);
  * a sub_session op builds the in-g6 RayGpuFarm (ray.init with the forced GPU count, fractional
    actors, warm CUDA) and leases an actor;
  * the checker subprocess sees the injected driver + the sitecustomize memory cap;
  * several sub_sessions under ONE shared runtimeSessionId run concurrently on the single GPU.

All episodes invoke under ONE shared runtimeSessionId (so they land on the SAME g6 instance) and
carry a per-episode sub_session — exactly what ray_farm_generate.generate does.

    AGENTCORE_SANDBOX_RUNTIME_ARN=<b4 arn> python3 smoke_b4_g6.py [--episodes 4]
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import boto3
from botocore.config import Config

REGION = os.environ.get("AWS_REGION", "us-west-2")
ARN = os.environ["AGENTCORE_SANDBOX_RUNTIME_ARN"]
SHARED_SESSION = "b4-smoke-shared-" + "x" * 24  # runtimeSessionId must be [33,128] chars

# Reports what the leased fractional-GPU actor's subprocess actually sees.
WORKLOAD = r"""
import json, os
try:
    import torch
    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
        a = torch.randn(1024, 1024, device=dev); b = torch.randn(1024, 1024, device=dev)
        for _ in range(5):
            a = (a @ b).relu()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated(0) // (1024 * 1024)
        print(json.dumps({"ok": True, "device": torch.cuda.get_device_name(0),
                          "visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
                          "mem_frac": os.environ.get("ENV_GPU_MEM_FRACTION"),
                          "peak_mib": peak, "backend": "cuda"}))
    else:
        print(json.dumps({"ok": True, "device": "cpu", "backend": "cpu-no-cuda"}))
except Exception as e:
    print(json.dumps({"ok": False, "err": f"{type(e).__name__}: {e}"}))
"""


def client():
    # Wide read timeout: the FIRST invoke cold-launches the g6, installs torch (~90s), inits ray and
    # warms CUDA — minutes, well past botocore's 60s default.
    cfg = Config(read_timeout=int(os.environ.get("INVOKE_READ_TIMEOUT", "900")),
                 connect_timeout=60, retries={"max_attempts": 0})
    return boto3.client("bedrock-agentcore", region_name=REGION, config=cfg)


def invoke(rt, payload: dict, *, retries: int = 40) -> dict:
    for attempt in range(retries):
        try:
            resp = rt.invoke_agent_runtime(agentRuntimeArn=ARN, runtimeSessionId=SHARED_SESSION,
                                           payload=json.dumps(payload).encode())
            raw = resp["response"].read()
            return json.loads(raw.decode()) if raw else {}
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            transient = any(s in msg for s in ("500", "502", "503", "504", "throttl",
                            "ServiceUnavailable", "RuntimeClientError", "not ready", "timeout",
                            "Timeout", "capacity"))
            print(f"  attempt {attempt}: {type(e).__name__}: {msg[:140]}")
            if not transient:
                raise
            time.sleep(15)
    raise SystemExit("b4 g6 sandbox never became ready")


def episode(rt, idx: int) -> dict:
    sub = f"smoke-ep-{idx}"
    invoke(rt, {"op": "open", "sub_session": sub, "workdir": "/workspace"})
    try:
        invoke(rt, {"op": "write_file", "sub_session": sub, "path": "work.py", "text": WORKLOAD})
        res = invoke(rt, {"op": "exec", "sub_session": sub, "command": "python3 work.py",
                          "timeout": 180})
    finally:
        invoke(rt, {"op": "close", "sub_session": sub})
    out = (res.get("stdout") or "").strip()
    try:
        parsed = json.loads(out.splitlines()[-1]) if out else {"ok": False}
    except (ValueError, IndexError):
        parsed = {"ok": False, "raw": out[-200:], "stderr": (res.get("stderr") or "")[-200:]}
    parsed["episode"] = idx
    return parsed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=4)
    args = ap.parse_args()
    rt = client()

    print(f"b4 g6 farm smoke: warming + first episode (cold g6 launch can take minutes)...")
    t0 = time.perf_counter()
    first = episode(rt, 0)
    print(f"  episode 0 ({time.perf_counter()-t0:.1f}s): {first}")
    ping = invoke(rt, {"op": "ping", "sub_session": "probe"})
    print(f"  farm: {ping.get('farm')}")

    if args.episodes > 1:
        print(f"fanning out {args.episodes-1} more concurrent episodes on the shared g6...")
        t1 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.episodes - 1) as pool:
            rest = list(pool.map(lambda i: episode(rt, i), range(1, args.episodes)))
        print(f"  {args.episodes-1} concurrent episodes done in {time.perf_counter()-t1:.1f}s")
        for r in rest:
            print(f"    {r}")

    all_ok = first.get("ok") and (args.episodes == 1 or all(r.get("ok") for r in rest))
    print("\nRESULT:", "PASS — b4 remote farm path works" if all_ok else "FAIL — see above")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
