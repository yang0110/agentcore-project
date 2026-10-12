---
name: kb-modelnew-contract
description: The exact rules the hidden KernelBench grader checks, and the 3 mistakes that fail most ModelNew solutions. Read first.
families: [kernelbench]
---
# ModelNew contract (what the grader checks, in order)

1. `ModelNew(*get_init_inputs())` builds, `.to('cuda')`, `.eval()`, then `forward(*get_inputs())`
   runs under `torch.no_grad()` with the inputs already on CUDA.
2. Output shape == reference shape.
3. `torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=1e-2)`.
4. The output tensor is on a `cuda` device.
5. Speedup vs `Model`: reward 1.0 needs **>= 1.10x**; correct but slower = 0.5; any failure = 0.

## The 3 mistakes behind most failures

- **Different weights.** The grader seeds the RNG, then builds `Model` and `ModelNew` the same way.
  If `Model` has parameters (`nn.Linear`, `nn.Conv2d`, `nn.Parameter(torch.randn…)`), copy its
  `__init__` **verbatim**: same submodules, same order, same shapes. Then use `self.<layer>.weight` in
  your fast path. Never create extra random tensors in `__init__` before them.
- **Wrong shape, or output on CPU.** Return exactly the reference's shape (keep reduced dims,
  scalars stay 0-d tensors). New tensors default to CPU: write `torch.empty_like(x)`, or pass
  `device=x.device, dtype=x.dtype`. Never call `.cpu()`, `.numpy()` or `.item()` on the result.
- **Doesn't build.** There is **no nvcc** in this sandbox, so `load_inline(cuda_sources=...)`
  fails. Use Triton (see skill `kb-triton-kernels`) or fused torch ops.

## Turn budget
You have few turns. Write `solution.py` early, check it once with `kb-self-check`, fix, stop.
A correct copy of `Model` named `ModelNew` already earns 0.5; ship that first if you are unsure.
