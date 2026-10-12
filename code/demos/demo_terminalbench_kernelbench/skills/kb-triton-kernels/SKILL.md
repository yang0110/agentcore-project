---
name: kb-triton-kernels
description: How to write a custom GPU kernel here (no nvcc, so Triton instead of load_inline), with a ready fused-elementwise template.
families: [kernelbench]
---
# Custom kernels without nvcc: Triton

This sandbox has PyTorch + Triton but **no CUDA toolkit**: `torch.utils.cpp_extension.load_inline`
with `cuda_sources` fails to build. Triton compiles to the GPU directly. Copy the template and edit
the marked line:

```bash
cp SKILL_DIR/template_elementwise.py solution.py
```

The template computes `y = f(x)` in a single pass over memory. Edit the one line marked `EDIT`.
Rules that keep it correct:
- Inputs must be contiguous (`x = x.contiguous()`). Allocate the output with `torch.empty_like(x)`.
- Compute in fp32 (`.to(tl.float32)`) and store back in the output dtype.
- Always mask the tail: `mask = offs < n`, and pass `mask=mask` to every `tl.load`/`tl.store`.
- `BLOCK` must be a power of 2 (1024 is a good default).
- Reductions over the last dim (softmax, norms, losses): one program per row,
  `tl.arange(0, BLOCK)` with `BLOCK = triton.next_power_of_2(ncols)`, then `tl.max` / `tl.sum` along axis 0.
- For matmul/conv, use the torch op (cuBLAS/cuDNN) and fuse only the elementwise ops *around* it.

The first call JIT-compiles (a few seconds). The grader warms up before timing, so that is fine.
