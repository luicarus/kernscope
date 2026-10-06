"""Softmax API and PyTorch reference."""

from typing import Literal

import torch
from torch import Tensor

from kernscope.ops._validation import _SUPPORTED_DTYPES


def softmax(x: Tensor, *, backend: Literal["torch", "triton"] = "torch") -> Tensor:
    """Normalize the last dimension using FP32 intermediates."""
    if backend not in ("torch", "triton"):
        raise ValueError("backend must be 'torch' or 'triton'")
    if not isinstance(x, Tensor):
        raise TypeError("x must be a torch.Tensor")
    if x.ndim == 0 or x.shape[-1] == 0:
        raise ValueError("x must have a nonempty final dimension")
    if x.dtype not in _SUPPORTED_DTYPES:
        raise TypeError("x must use float16, bfloat16, or float32")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")

    if backend == "triton":
        if x.device.type != "cuda":
            raise ValueError("the triton backend requires CUDA tensors")
        if x.requires_grad:
            raise ValueError("the triton backend does not support autograd")
        from kernscope.backends.triton_softmax import softmax_triton

        return softmax_triton(x)

    with torch.autocast(device_type=x.device.type, enabled=False):
        return torch.softmax(x.float(), dim=-1).to(x.dtype)
