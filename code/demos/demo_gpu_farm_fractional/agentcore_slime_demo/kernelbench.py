"""KernelBench-style GPU tasks for the Demo B (GPU instance) RL path.

KernelBench (Ouyang et al., 2025) frames GPU-kernel authoring as: given a reference
PyTorch operator, write a custom implementation that is (1) numerically CORRECT vs the
reference and (2) actually runs on the GPU — with speedup as the headline metric. We adopt
that shape here, deliberately self-contained (no external KernelBench repo dependency) so
it runs today on this box's L40S and inside an AgentCore GPU Runtime instance.

Each task reuses the proven assert-mode runner (task.py): the reference output and random
CUDA inputs are built in `test_setup`, and the checks assert the agent's `custom_op`
matches the reference within tolerance AND ran on a 'cuda' device. Because it is just a
CodingTask, it flows through the SAME multi-turn harness as the CPU tasks; the checker's
CUDA exec runs on the remote GPU sandbox Runtime.

Reward is binary (correct + on GPU), matching the exit-code reward primitive used
everywhere else. Speedup vs the reference is MEASURED and printed by the checker as a
metric (KernelBench's headline number) but does not gate the reward — a correctness-first
RL signal, with speedup available for a shaped reward later.

mock_attempts (offline MockPolicy only): attempt 0 computes on CPU -> fails the 'cuda'
check; attempt 1 moves to the GPU -> passes. Live RL ignores them (the policy writes code).
"""

from __future__ import annotations

from .task import CodingTask

# Built once in the checker namespace before the asserts run. Fixes a seed, puts random
# inputs on CUDA, computes the reference, and defines a helper that times custom vs
# reference and prints the speedup (metric only).
_SETUP_TMPL = """\
import torch, time
torch.manual_seed(0)
_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
{inputs}
_ref = {reference}

def _report_speedup(fn):
    if not torch.cuda.is_available():
        return
    for _ in range(3):
        fn()                      # warmup
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    _cust = (time.perf_counter() - t0) / 20
    t0 = time.perf_counter()
    for _ in range(20):
        {ref_call}
    torch.cuda.synchronize()
    _base = (time.perf_counter() - t0) / 20
    print(f"SPEEDUP vs reference: {{_base / max(_cust, 1e-9):.2f}}x "
          f"(custom {{_cust*1e3:.3f}}ms, ref {{_base*1e3:.3f}}ms)")
"""

# The common check set: CUDA present, correct shape, correct values within tol, ran on GPU,
# and a speedup report. Uses bare names (assert mode seeds the namespace with the solution's
# globals + this setup), so `custom_op` resolves directly.
def _checks(call: str, tol: str = "atol=1e-3, rtol=1e-3") -> list[str]:
    return [
        "assert torch.cuda.is_available(), 'no CUDA GPU in sandbox'",
        f"assert {call}.shape == _ref.shape, 'wrong output shape'",
        f"assert torch.allclose({call}.float(), _ref.float(), {tol}), 'values differ from reference'",
        f"assert str({call}.device).startswith('cuda'), 'op did not run on the GPU'",
        f"_report_speedup(lambda: {call}) or True",
    ]


# ---- offline mock attempts: a CPU (buggy) version and a CUDA (fixed) version -----------
def _mock(cpu_body: str, gpu_body: str) -> list[str]:
    return [cpu_body, gpu_body]


_TASKS: list[CodingTask] = [
    CodingTask(
        task_id="kb_matmul_relu",
        prompt=(
            "Implement `custom_op(A, B)` computing relu(A @ B) for two 2-D float tensors, "
            "running on the CUDA GPU and returning a CUDA tensor. Match the reference "
            "torch result. Signature: def custom_op(A, B)."
        ),
        checks=_checks("custom_op(A, B)"),
        check_mode="assert",
        test_setup=_SETUP_TMPL.format(
            inputs="A = torch.randn(256, 256, device=_dev)\nB = torch.randn(256, 256, device=_dev)",
            reference="(A @ B).relu()",
            ref_call="_ = (A @ B).relu()",
        ),
        max_turns=3,
        mock_attempts=_mock(
            "import torch\n\ndef custom_op(A, B):\n    # buggy: forces CPU\n    return (A.cpu() @ B.cpu()).relu()\n",
            "import torch\n\ndef custom_op(A, B):\n    return (A.cuda() @ B.cuda()).relu()\n",
        ),
    ),
    CodingTask(
        task_id="kb_row_softmax",
        prompt=(
            "Implement `custom_op(X)` computing a numerically-stable row-wise softmax of a "
            "2-D float tensor on the CUDA GPU, returning a CUDA tensor. Match torch.softmax "
            "along dim=1. Signature: def custom_op(X)."
        ),
        checks=_checks("custom_op(X)"),
        check_mode="assert",
        test_setup=_SETUP_TMPL.format(
            inputs="X = torch.randn(512, 512, device=_dev)",
            reference="torch.softmax(X, dim=1)",
            ref_call="_ = torch.softmax(X, dim=1)",
        ),
        max_turns=3,
        mock_attempts=_mock(
            "import torch\n\ndef custom_op(X):\n    # buggy: CPU\n    Xc = X.cpu()\n    return torch.softmax(Xc, dim=1)\n",
            "import torch\n\ndef custom_op(X):\n    Xg = X.cuda()\n    m = Xg.max(dim=1, keepdim=True).values\n    e = (Xg - m).exp()\n    return e / e.sum(dim=1, keepdim=True)\n",
        ),
    ),
    CodingTask(
        task_id="kb_fused_bias_gelu",
        prompt=(
            "Implement `custom_op(X, b)` computing gelu(X + b) where X is 2-D and b is a 1-D "
            "bias broadcast over rows, on the CUDA GPU, returning a CUDA tensor. Match "
            "torch.nn.functional.gelu(X + b). Signature: def custom_op(X, b)."
        ),
        checks=_checks("custom_op(X, b)"),
        check_mode="assert",
        test_setup=_SETUP_TMPL.format(
            inputs="X = torch.randn(384, 384, device=_dev)\nb = torch.randn(384, device=_dev)",
            reference="torch.nn.functional.gelu(X + b)",
            ref_call="_ = torch.nn.functional.gelu(X + b)",
        ),
        max_turns=3,
        mock_attempts=_mock(
            "import torch\n\ndef custom_op(X, b):\n    # buggy: CPU\n    return torch.nn.functional.gelu(X.cpu() + b.cpu())\n",
            "import torch\n\ndef custom_op(X, b):\n    return torch.nn.functional.gelu(X.cuda() + b.cuda())\n",
        ),
    ),
]


def load_kernelbench_tasks() -> list[CodingTask]:
    """The hand-written 3-op smoke set (offline MockPolicy dry-runs use its mock_attempts)."""
    return list(_TASKS)


# ---- REAL KernelBench (ScalingIntelligence/KernelBench) — 270 CUDA problems -------------
# The 3-op set above is a self-contained smoke stub. For real training/eval we pull the
# actual KernelBench benchmark from HuggingFace: 270 problems across 4 levels
# (level_1=100 single ops, level_2=100 operator fusions, level_3=50 full architectures,
# level_4=20). Each HF row gives a reference `Model` (nn.Module) plus module-level
# get_inputs()/get_init_inputs(). The standard KernelBench task: write an equivalent
# `ModelNew` that runs on CUDA and matches the reference output (speedup is measured).
#
# We reuse the SAME assert-mode checker as everything else: test_setup rebuilds the
# reference on the GPU with a fixed seed and computes `_ref`; the checks build the agent's
# `ModelNew` with the SAME seed (so any trainable params match exactly), run it on the same
# inputs, and assert numeric equivalence + ran-on-GPU + a speedup report. No mock_attempts
# (real-RL only — the offline mock smoke path uses load_kernelbench_tasks()).

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
    return (
        _KB_PROMPT_HEAD
        + f"Reference architecture (problem {problem_id}: {name}):\n```python\n"
        + code.strip()
        + "\n```\n\nReturn a complete solution that defines `ModelNew`."
    )


def load_kernelbench_hf(level: str = "level_1", limit: int | None = None,
                        offset: int = 0, max_turns: int = 3) -> list[CodingTask]:
    """Load real KernelBench problems from HuggingFace as GPU CodingTasks.

    ``level`` is one of ``level_1`` (100 single ops), ``level_2`` (100 fusions),
    ``level_3`` (50 architectures), ``level_4`` (20). ``offset``/``limit`` slice the level —
    use disjoint slices (or different levels) for train vs held-out eval to avoid leakage.
    Requires the ``datasets`` package + network/HF cache (available in the training container).
    """
    from datasets import load_dataset

    rows = load_dataset("ScalingIntelligence/KernelBench")[level]
    n = len(rows)
    end = n if limit is None else min(offset + limit, n)
    tasks: list[CodingTask] = []
    for r in rows.select(range(min(offset, n), end)):
        code = r["code"]
        # sanitize name for a filesystem/id-safe task_id
        safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in str(r["name"]))
        tasks.append(CodingTask(
            task_id=f"kb_{level}_{r['problem_id']}_{safe_name}",
            prompt=_kb_prompt(r["problem_id"], r["name"], code),
            checks=list(_KB_CHECKS),
            check_mode="assert",
            test_setup=_KB_SETUP_HEAD + code + _KB_SETUP_TAIL,
            max_turns=max_turns,
        ))
    return tasks
