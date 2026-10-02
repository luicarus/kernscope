"""RMSNorm API and PyTorch reference."""

from typing import Literal

import torch
from torch import Tensor

from kernscope.ops._validation import _validate_rms_inputs


def rms_norm(
    x: Tensor,
    weight: Tensor,
    eps: float = 1e-6,
    *,
    backend: Literal["torch", "triton", "tilelang_ascend"] = "torch",
) -> Tensor:
    """Apply RMSNorm over the last dimension, accumulating in FP32."""
    if backend not in ("torch", "triton", "tilelang_ascend"):
        raise ValueError("backend must be 'torch', 'triton', or 'tilelang_ascend'")
    eps = _validate_rms_inputs(x, weight, eps)

    if backend == "torch":
        x_fp32 = x.float()
        scale = torch.rsqrt(x_fp32.square().mean(dim=-1, keepdim=True) + eps)
        return (x_fp32 * scale * weight.float()).to(x.dtype)

    device_type = "cuda" if backend == "triton" else "npu"
    if x.device.type != device_type:
        raise ValueError(f"the {backend} backend requires {device_type.upper()} tensors")
    if x.requires_grad or weight.requires_grad:
        raise ValueError(f"the {backend} backend does not support autograd")
    if backend == "tilelang_ascend":
        from kernscope.backends.tilelang_ascend import rms_norm_tilelang_ascend

        return rms_norm_tilelang_ascend(x, weight, eps)

    try:
        from kernscope.backends.triton import rms_norm_triton
    except ModuleNotFoundError as error:
        if error.name != "triton":
            raise
        raise ImportError("install kernscope[gpu] to use the triton backend") from error
    return rms_norm_triton(x, weight, eps)
