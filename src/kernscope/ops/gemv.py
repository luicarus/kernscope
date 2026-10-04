"""GEMV API and PyTorch reference."""

from typing import Literal

import torch
from torch import Tensor

from kernscope.ops._validation import _SUPPORTED_DTYPES


def gemv(x: Tensor, weight: Tensor, *, backend: Literal["torch", "triton"] = "torch") -> Tensor:
    """Multiply weight (N, H) by x (H,), accumulating in FP32."""
    if backend not in ("torch", "triton"):
        raise ValueError("backend must be 'torch' or 'triton'")
    if not isinstance(x, Tensor) or not isinstance(weight, Tensor):
        raise TypeError("x and weight must be torch.Tensor instances")
    if x.ndim != 1 or x.numel() == 0:
        raise ValueError("x must be a nonempty vector")
    if weight.ndim != 2 or weight.shape[1] != x.shape[0]:
        raise ValueError("weight must have shape (N, x.shape[0])")
    if x.dtype not in _SUPPORTED_DTYPES:
        raise TypeError("x must use float16, bfloat16, or float32")
    if weight.dtype != x.dtype or weight.device != x.device:
        raise ValueError("weight must match x's dtype and device")
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("x and weight must be contiguous")

    if backend == "triton":
        if x.device.type != "cuda":
            raise ValueError("the triton backend requires CUDA tensors")
        if x.requires_grad or weight.requires_grad:
            raise ValueError("the triton backend does not support autograd")
        from kernscope.backends.triton_gemv import gemv_triton

        return gemv_triton(x, weight)

    with torch.autocast(device_type=x.device.type, enabled=False):
        return torch.mv(weight.float(), x.float()).to(x.dtype)
