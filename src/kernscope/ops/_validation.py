import math

import torch
from torch import Tensor

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _validate_rms_inputs(x: Tensor, weight: Tensor, eps: float) -> float:
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
    try:
        eps = float(eps)
    except OverflowError as error:
        raise ValueError("eps must be finite and positive") from error
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    return eps
