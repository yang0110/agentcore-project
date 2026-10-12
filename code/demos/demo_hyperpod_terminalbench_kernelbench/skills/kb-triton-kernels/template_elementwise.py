import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.maximum(x, 0.0)          # EDIT: the elementwise math of Model.forward (e.g. relu shown)
    tl.store(y_ptr + offs, y.to(y_ptr.dtype.element_ty), mask=mask)


def fused(x: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    y = torch.empty_like(x)
    n = x.numel()
    BLOCK = 1024
    _fused_kernel[(triton.cdiv(n, BLOCK),)](x, y, n, BLOCK=BLOCK)
    return y


class ModelNew(nn.Module):
    def __init__(self, *args):      # EDIT: same signature + same submodules, same order, as Model.__init__
        super().__init__()

    def forward(self, x):
        return fused(x)
