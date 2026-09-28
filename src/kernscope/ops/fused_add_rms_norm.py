from typing import Literal

import torch
from torch import Tensor

from kernscope.ops._validation import _validate_rms_inputs


def _overlaps(a: Tensor, b: Tensor) -> bool:
    if not a.numel() or not b.numel():
        return False
    a_start, b_start = a.data_ptr(), b.data_ptr()
    a_end = a_start + a.numel() * a.element_size()
    b_end = b_start + b.numel() * b.element_size()
    return a_start < b_end and b_start < a_end


def fused_add_rms_norm(
    x: Tensor,
    residual: Tensor,
    weight: Tensor,
    eps: float = 1e-6,
    *,
    backend: Literal["torch", "triton"] = "torch",
) -> None:
    """Update residual with the sum and x with its RMSNorm, in place."""
    if backend not in ("torch", "triton"):
        raise ValueError("backend must be 'torch' or 'triton'")
    eps = _validate_rms_inputs(x, weight, eps)
    if not isinstance(residual, Tensor):
        raise TypeError("residual must be a torch.Tensor")
    if residual.shape != x.shape:
        raise ValueError("residual and x must have the same shape")
    if residual.dtype != x.dtype or residual.device != x.device:
        raise ValueError("residual must match x's dtype and device")
    if not residual.is_contiguous():
        raise ValueError("residual must be contiguous")
    if any(_overlaps(x, other) for other in (residual, weight)) or _overlaps(residual, weight):
        raise ValueError("x, residual, and weight must not overlap")
    if x.requires_grad or residual.requires_grad or weight.requires_grad:
        raise ValueError("fused_add_rms_norm is inference-only")
    if backend == "triton":
        if x.device.type != "cuda":
            raise ValueError("the triton backend requires CUDA tensors")
        try:
            from kernscope.backends.triton import fused_add_rms_norm_triton
        except ModuleNotFoundError as error:
            if error.name != "triton":
                raise
            raise ImportError("install kernscope[gpu] to use the triton backend") from error
        return fused_add_rms_norm_triton(x, residual, weight, eps)

    summed = x.float() + residual.float()
    normalized = summed * torch.rsqrt(summed.square().mean(dim=-1, keepdim=True) + eps)
    normalized = normalized * weight.float()
    residual.copy_(summed.to(residual.dtype))
    x.copy_(normalized.to(x.dtype))
