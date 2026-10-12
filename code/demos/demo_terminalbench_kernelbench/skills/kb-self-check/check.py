#!/usr/bin/env python3
"""Self-check a KernelBench solution the way the hidden grader does (same seeds, checks and timing).

    python3 check.py [solution.py]

reference.py (the task's Model + get_inputs + get_init_inputs) sits next to this script; the skill
installer writes it per episode from the task prompt. This file never sees the grader itself.
"""
import importlib.util
import os
import sys
import time
import traceback

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
_SEED_W, _SEED_F = 1, 2


def _fail(check, why):
    print(f"FAIL {check}: {why}")
    sys.exit(1)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    sol_path = sys.argv[1] if len(sys.argv) > 1 else "solution.py"
    if not torch.cuda.is_available():
        _fail("cuda", "no CUDA GPU visible")
    dev = torch.device("cuda")
    torch.backends.cudnn.deterministic = True
    ref = _load(os.path.join(HERE, "reference.py"), "kb_reference")
    try:
        sol = _load(os.path.abspath(sol_path), "kb_solution")
        ModelNew = sol.ModelNew
    except Exception:
        traceback.print_exc(limit=3)
        _fail("build", f"cannot import ModelNew from {sol_path}")

    torch.manual_seed(0)
    init = ref.get_init_inputs() if hasattr(ref, "get_init_inputs") else []
    fwd = [x.to(dev) if torch.is_tensor(x) else x for x in ref.get_inputs()]

    def build(cls):
        torch.manual_seed(_SEED_W)
        m = cls(*init)
        if hasattr(m, "to"):
            m = m.to(dev)
        if hasattr(m, "eval"):
            m.eval()
        return m

    def run(m):
        torch.manual_seed(_SEED_F)
        with torch.no_grad():
            return m(*fwd)

    expect = run(build(ref.Model))
    try:
        new = build(ModelNew)
        out = run(new)
        torch.cuda.synchronize()
    except Exception:
        traceback.print_exc(limit=4)
        _fail("build", "ModelNew failed to construct or run")
    print("ok   build")
    if not torch.is_tensor(out):
        _fail("shape", f"forward returned {type(out).__name__}, not a tensor")
    if not str(out.device).startswith("cuda"):
        _fail("device", f"output is on {out.device}, must be cuda")
    print("ok   device")
    if out.shape != expect.shape:
        _fail("shape", f"{tuple(out.shape)} vs reference {tuple(expect.shape)}")
    print("ok   shape", tuple(out.shape))
    if not torch.allclose(out.float(), expect.float(), atol=1e-2, rtol=1e-2):
        diff = (out.float() - expect.float()).abs()
        _fail("allclose", f"max abs diff {diff.max().item():.4g} (atol/rtol 1e-2); "
                          "same weights? copy Model.__init__ verbatim")
    print("ok   allclose")

    def timed(m):
        for _ in range(3):
            run(m)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            run(m)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / 20

    t_new = timed(new)
    t_ref = timed(build(ref.Model))
    sp = t_ref / max(t_new, 1e-9)
    verdict = "full reward" if sp >= 1.10 else "0.5 reward: needs >= 1.10x for full"
    print(f"ALL OK  speedup={sp:.2f}x (custom {t_new * 1e3:.3f}ms, ref {t_ref * 1e3:.3f}ms) -> {verdict}")


if __name__ == "__main__":
    main()
