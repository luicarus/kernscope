"""SwiGLU activation API and PyTorch reference."""

from typing import Literal

import torch.nn.functional as F
from torch import Tensor

from kernscope.ops._validation import _SUPPORTED_DTYPES


def silu_and_mul(x: Tensor, *, backend: Literal["torch", "triton"] = "torch") -> Tensor:
    """Apply SiLU to the first half, multiply by the second, using FP32 intermediates."""
    if backend not in ("torch", "triton"):
        raise ValueError("backend must be 'torch' or 'triton'")
    if not isinstance(x, Tensor):
        raise TypeError("x must be a torch.Tensor")
    if x.ndim == 0 or x.shape[-1] == 0 or x.shape[-1] % 2:
        raise ValueError("x must have a nonempty, even final dimension")
    if x.dtype not in _SUPPORTED_DTYPES:
        raise TypeError("x must use float16, bfloat16, or float32")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")

    if backend == "torch":
        gate, up = x.float().chunk(2, dim=-1)
        return (F.silu(gate) * up).to(x.dtype)

    if x.device.type != "cuda":
        raise ValueError("the triton backend requires CUDA tensors")
    if x.requires_grad:
        raise ValueError("the triton backend does not support autograd")
    from kernscope.backends.triton import silu_and_mul_triton

    return silu_and_mul_triton(x)
