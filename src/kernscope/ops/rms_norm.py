"""PyTorch reference implementation of RMSNorm."""

from typing import Literal

import torch
from torch import Tensor

from kernscope.ops._validation import _validate_rms_inputs


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
    eps = _validate_rms_inputs(x, weight, eps)

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
