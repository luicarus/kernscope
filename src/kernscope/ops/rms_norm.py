"""PyTorch reference implementation of RMSNorm."""

import math
from typing import Literal

import torch
from torch import Tensor

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def rms_norm(
    x: Tensor,
    weight: Tensor,
    eps: float = 1e-6,
    *,
    backend: Literal["torch", "triton"] = "torch",
) -> Tensor:
    """Apply RMSNorm over the last dimension, accumulating in FP32."""
    if backend not in ("torch", "triton"):
        raise ValueError("backend must be 'torch' or 'triton'")
    if not isinstance(x, Tensor) or not isinstance(weight, Tensor):
        raise TypeError("x and weight must be torch.Tensor instances")
    if x.ndim == 0 or x.shape[-1] == 0:
        raise ValueError("x must have a nonempty final dimension")
    if weight.shape != (x.shape[-1],):
        raise ValueError("weight must have shape (x.shape[-1],)")
    if x.dtype not in _SUPPORTED_DTYPES:
        raise TypeError("x must use float16, bfloat16, or float32")
    if weight.dtype != x.dtype or weight.device != x.device:
        raise ValueError("weight must match x's dtype and device")
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("x and weight must be contiguous")
    if isinstance(eps, bool) or not isinstance(eps, (int, float)):
        raise TypeError("eps must be numeric")
    eps = float(eps)
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")

    if backend == "torch":
        x_fp32 = x.float()
        scale = torch.rsqrt(x_fp32.square().mean(dim=-1, keepdim=True) + eps)
        return (x_fp32 * scale * weight.float()).to(x.dtype)

    if x.device.type != "cuda":
        raise ValueError("the triton backend requires CUDA tensors")
    if x.requires_grad or weight.requires_grad:
        raise ValueError("the triton backend does not support autograd")
    try:
        from kernscope.backends.triton import rms_norm_triton
    except ModuleNotFoundError as error:
        if error.name != "triton":
            raise
        raise ImportError("install kernscope[gpu] to use the triton backend") from error
    return rms_norm_triton(x, weight, eps)
