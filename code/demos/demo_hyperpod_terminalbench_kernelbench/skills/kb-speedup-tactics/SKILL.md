---
name: kb-speedup-tactics
description: Which optimizations actually beat the PyTorch reference by >=1.10x on this GPU, and which waste turns.
families: [kernelbench]
---
# Getting from 1.00x to >= 1.10x

A plain copy of `Model` measures about 1.00x (0.5 reward). The wins come from **doing fewer passes over
memory**:

| Pattern in `Model.forward` | Do this |
|---|---|
| a chain of elementwise ops (`relu(x)*2+1`, `clamp`, `sigmoid*x`…) | fuse the chain into ONE Triton kernel (`kb-triton-kernels`) |
| matmul/conv followed by elementwise ops | keep the torch matmul/conv, fuse everything after it into one kernel |
| `x @ W.T + b` | `torch.addmm(b, x, W.t())` (one cuBLAS call) |
| softmax / layernorm / loss over the last dim | one Triton program per row (load, reduce, write once) |
| attention-like `softmax(q@k.T/s)@v` | `torch.nn.functional.scaled_dot_product_attention` |
| a reduction (`mean`/`sum`) of an elementwise expression | compute the expression inside the reduction kernel; never materialize it |
| several small ops on a tiny tensor | `torch.compile(model)` once in `__init__`, or leave it: launch overhead dominates |

Waste of turns: rewriting cuBLAS matmul or cuDNN conv by hand (you will be slower); `load_inline`
CUDA (no nvcc here); `torch.compile(mode="max-autotune")` (too slow to compile).

Allowed but risky: `torch.backends.cuda.matmul.allow_tf32 = True` speeds up big fp32 matmuls, but
the error can exceed atol 1e-2 when K is large. Self-check it.

Always keep the last version that passed `kb-self-check`. A correct 1.0x beats a broken 2x.
