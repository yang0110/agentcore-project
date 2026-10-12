#!/usr/bin/env python3
"""Build the KernelBench rows (ScalingIntelligence/KernelBench, all 270 problems) -> kernelbench/*.jsonl.

Run INSIDE slime_trainer (it has `datasets` + the HF cache):
    python3 kernelbench/build_kernelbench.py              # all 4 levels -> kernelbench_{train,eval}.jsonl

Levels: level_1 = 100 single ops, level_2 = 100 fusions, level_3 = 50 architectures, level_4 = 20 HF models.
Held-out eval (27): the same 20 level_2 problems as before (ids 1, 10-27, 100: eval/kernelbench stays
comparable with earlier runs) + level_3 ids 10/20/30/40/50 + level_4 ids 10/20. Train = the other 243.

Each row names an image PER LEVEL so image_routing.py can send levels to different substrates:
    kernelbench-gpu     level_1  (the AgentCore g6 Runtime by default; the key earlier runs used)
    kernelbench-gpu-l2  level_2  \
    kernelbench-gpu-l3  level_3   > TB_KB_EC2_LEVELS (default 2,3,4) -> the plain-EC2 GPU pool
    kernelbench-gpu-l4  level_4  /  level_4 ONLY there: its HF models are baked into the EC2 image
TB_KB_LEVELS=1,2,3 leaves levels out. kernelbench/oracle.csv (validate_kernelbench.py: the reference
itself submitted as ModelNew) drops the problems whose oracle fails, each listed with its reason.

The prompt, checks and test_setup are byte-identical to the rows the demo used before (verified against
the earlier in-tree level_1 + level_2 files): the generator is the old agentcore_slime_demo.kernelbench.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL_IDS = {"level_2": {1, *range(10, 28), 100}, "level_3": {10, 20, 30, 40, 50}, "level_4": {10, 20}}
IMAGE = {"level_1": "kernelbench-gpu", "level_2": "kernelbench-gpu-l2", "level_3": "kernelbench-gpu-l3",
         "level_4": "kernelbench-gpu-l4"}
MAX_TURNS = 3

_KB_SETUP_HEAD = "import torch\ntorch.backends.cudnn.deterministic = True\n"

# Appended AFTER the reference `code` (which defines Model + get_inputs + get_init_inputs).
# Seeds are split so weight-init (_SEED_W) and any forward-time randomness (_SEED_F) are
# reproduced identically for the reference and the candidate ModelNew.
_KB_SETUP_TAIL = '''
_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_SEED_W, _SEED_F = 1, 2

torch.manual_seed(0)
_kb_init = get_init_inputs() if 'get_init_inputs' in globals() else []
_kb_fwd = [x.to(_dev) if torch.is_tensor(x) else x for x in get_inputs()]

def _kb_build(cls):
    torch.manual_seed(_SEED_W)
    m = cls(*_kb_init)
    if hasattr(m, 'to'):
        m = m.to(_dev)
    if hasattr(m, 'eval'):
        m.eval()
    return m

def _kb_run(m):
    torch.manual_seed(_SEED_F)
    with torch.no_grad():
        return m(*_kb_fwd)

_ref = _kb_run(_kb_build(Model))

def _report_speedup(new_model):
    if not torch.cuda.is_available():
        return True
    import time
    for _ in range(3):
        _kb_run(new_model)          # warmup
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        _kb_run(new_model)
    torch.cuda.synchronize()
    _cust = (time.perf_counter() - t0) / 20
    _base_model = _kb_build(Model)
    for _ in range(3):
        _kb_run(_base_model)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        _kb_run(_base_model)
    torch.cuda.synchronize()
    _base = (time.perf_counter() - t0) / 20
    print(f"SPEEDUP vs reference: {_base / max(_cust, 1e-9):.2f}x "
          f"(custom {_cust*1e3:.3f}ms, ref {_base*1e3:.3f}ms)")
    return True
'''

# Bare-name asserts (assert mode seeds the ns with the solution globals + the setup above).
_KB_CHECKS = [
    "assert torch.cuda.is_available(), 'no CUDA GPU in sandbox'",
    "_out = _kb_run(_kb_build(ModelNew))",
    "assert _out.shape == _ref.shape, f'wrong output shape {_out.shape} vs {_ref.shape}'",
    "assert torch.allclose(_out.float(), _ref.float(), atol=1e-2, rtol=1e-2), 'values differ from reference'",
    "assert str(_out.device).startswith('cuda'), 'ModelNew did not run on the GPU'",
    "_report_speedup(_kb_build(ModelNew))",
]

_KB_PROMPT_HEAD = (
    "You are given a reference PyTorch architecture `Model`. Write an OPTIMIZED implementation "
    "named `ModelNew` (a torch.nn.Module) that:\n"
    "  1. produces NUMERICALLY EQUIVALENT output to `Model` for the same inputs (atol/rtol 1e-2), and\n"
    "  2. runs on the CUDA GPU (its output tensor must be on a 'cuda' device).\n"
    "`ModelNew.__init__` must accept the same arguments as `Model.__init__` (see get_init_inputs) "
    "and its `forward` the same inputs (see get_inputs). You may use custom CUDA kernels "
    "(e.g. torch.utils.cpp_extension.load_inline), operator fusion, or optimized torch ops. A "
    "speedup vs the reference is measured, but correctness on the GPU is what is required.\n\n"
)


def _kb_prompt(problem_id, name, code: str) -> str:
    # Concatenate (NOT .format) — reference code may contain literal braces.
    return (_KB_PROMPT_HEAD + f"Reference architecture (problem {problem_id}: {name}):\n```python\n"
            + code.strip() + "\n```\n\nReturn a complete solution that defines `ModelNew`.")


def rows(level: str, ds) -> list[dict]:
    out = []
    for r in ds[level]:
        safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in str(r["name"]))
        tid = f"kb_{level}_{r['problem_id']}_{safe}"
        prompt = _kb_prompt(r["problem_id"], r["name"], r["code"])
        task = {"task_id": tid, "prompt": prompt, "checks": list(_KB_CHECKS), "check_mode": "assert",
                "test_setup": _KB_SETUP_HEAD + r["code"] + _KB_SETUP_TAIL, "max_turns": MAX_TURNS,
                "image": IMAGE[level]}
        out.append({"prompt": prompt, "label": tid, "metadata": {"task": task},
                    "_eval": int(r["problem_id"]) in EVAL_IDS.get(level, set())})
    return out


def _oracle_failures() -> dict:
    p = HERE / "oracle.csv"
    if not p.exists():
        return {}
    with p.open() as f:
        return {r["task"]: r.get("reason", "") for r in csv.DictReader(f) if r.get("status") in ("fail", "timeout")}


def main() -> int:
    from datasets import load_dataset

    ds = load_dataset("ScalingIntelligence/KernelBench")
    levels = [f"level_{x.strip()}" for x in os.environ.get("TB_KB_LEVELS", "1,2,3,4").split(",") if x.strip()]
    bad = _oracle_failures()
    train, held, dropped = [], [], []
    for lvl in levels:
        for r in rows(lvl, ds):
            if r["label"] in bad:
                dropped.append((r["label"], bad[r["label"]]))
                continue
            (held if r.pop("_eval") else train).append(r)
    for name, rs in (("kernelbench_train.jsonl", train), ("kernelbench_eval.jsonl", held)):
        with (HERE / name).open("w") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")
    print(f"# kernelbench: {len(train)} train + {len(held)} eval rows from {', '.join(levels)}"
          + (f"; {len(dropped)} dropped by oracle.csv:" if dropped else ""))
    for tid, why in dropped:
        print(f"#   - {tid}: {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
