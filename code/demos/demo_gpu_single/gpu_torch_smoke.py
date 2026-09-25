#!/usr/bin/env python3
"""demo_gpu_single (custom-Runtime angle) — prove a CUSTOM AgentCore Runtime gives you a GPU the managed sandbox can't.

The GPU teaching point is a CONTRAST the CPU demo can't show: the managed Code Interpreter
(SANDBOX=agentcore) runs on a CPU serverless microVM and has NO GPU, so `torch.cuda.is_available()`
is False there (torch usually isn't even installed). Only YOUR OWN custom Runtime, deployed on an
AgentCore capacity-provider *Instance* (a g6 GPU box), gives the sandbox a real CUDA device — while
exposing the exact SAME task-agnostic contract (`write_file` / `exec` / `read_file`).

This driver exercises that contract directly (no training, no model):
    1. write_file()  a tiny torch-on-CUDA probe script into the sandbox
    2. exec()        `python3 <probe>` and capture (exit_code, stdout, stderr)
    3. read_file()   a result file the probe wrote — proving round-trip file I/O
and prints whether CUDA was visible. Same three calls work on every backend; only the answer to
"is there a GPU?" differs.

Usage:
    # offline smoke (no AWS) — proves the write/exec/read contract with LocalSandbox. CUDA on the
    # host may be True or False; the offline path only asserts the contract round-trips, not CUDA:
    python3 gpu_torch_smoke.py

    # live: run the probe inside YOUR custom GPU Runtime on g6 (expect CUDA=True):
    AWS_REGION=us-west-2 AGENTCORE_SANDBOX_RUNTIME_ARN=<gpu-arn> \
        python3 gpu_torch_smoke.py --backend agentcore-runtime

    # live: show the managed Code Interpreter has NO GPU (expect CUDA=False / no torch):
    AWS_REGION=us-west-2 python3 gpu_torch_smoke.py --backend agentcore
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agentcore_slime_demo.sandbox import make_sandbox

# Written into the sandbox and executed there. Guarded so it degrades cleanly on a backend with no
# torch (e.g. the managed Code Interpreter): it still writes a result file and exits 0, so the
# read_file round-trip is exercised on every backend — only the CUDA answer differs.
_PROBE = r'''
import json
info = {"torch": None, "cuda_available": False, "device": None}
try:
    import torch
    info["torch"] = torch.__version__
    info["cuda_available"] = bool(torch.cuda.is_available())
    if info["cuda_available"]:
        info["device"] = torch.cuda.get_device_name(0)
except Exception as e:
    info["error"] = f"{type(e).__name__}: {e}"
# Relative path: the sandbox contract writes/reads/execs against the sandbox working dir on every
# backend (LocalSandbox remaps under its tmp root; the custom Runtime uses SANDBOX_WORKDIR), so a
# relative path resolves consistently where an absolute one would not.
with open("gpu_probe_result.json", "w") as f:
    json.dump(info, f)
print("PROBE", json.dumps(info))
'''

# The reason to reach for each backend for GPU work — printed after the run.
_OPS = {
    "agentcore": "Managed Code Interpreter: CPU microVM only — NO GPU (expect cuda_available=False / no torch)",
    "agentcore-runtime": "Custom Runtime on a g6 capacity-provider Instance: a REAL CUDA device (expect cuda_available=True)",
    "local": "LocalSandbox (offline subprocess): whatever CUDA the host has; used only to prove the contract round-trips",
}


async def _run(backend: str) -> int:
    async with make_sandbox(backend) as sb:
        print(f"[{backend}] sandbox_id={sb.sandbox_id}")
        # 1. write the probe   2. exec it   3. read back the result file — the whole contract.
        # Relative paths resolve against the sandbox working dir on every backend (see the probe).
        await sb.write_file("gpu_probe.py", _PROBE)
        exit_code, stdout, stderr = await sb.exec("python3 gpu_probe.py", timeout=300, check=False)
        print(f"[{backend}] exec exit_code={exit_code}")
        if stdout.strip():
            print(f"[{backend}] stdout: {stdout.strip()}")
        if stderr.strip():
            print(f"[{backend}] stderr: {stderr.strip()[:400]}")
        try:
            result = await sb.read_file("gpu_probe_result.json")
            print(f"[{backend}] read_file(result.json): {result.strip()}")
        except Exception as e:  # noqa: BLE001
            print(f"[{backend}] read_file FAILED: {type(e).__name__}: {e}")
            return 1
    return exit_code


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="local",
                    help="sandbox backend (default: local offline smoke). "
                         "agentcore-runtime = your custom GPU Runtime; agentcore = managed Code Interpreter (no GPU)")
    args = ap.parse_args(argv)

    print(f"Backend: {args.backend}  ({_OPS.get(args.backend, 'unknown backend')})\n")
    rc = asyncio.run(_run(args.backend))
    print(f"\nWhat this shows: the SAME write_file/exec/read_file contract ran on '{args.backend}'.")
    print("A GPU appears ONLY under a custom Runtime on a g6 Instance — the managed Code Interpreter has none.")
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
