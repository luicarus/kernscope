"""Triton RMSNorm kernel."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    hidden_size: tl.constexpr,
    eps: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, block_size)
    mask = cols < hidden_size
    x = tl.load(x_ptr + row * hidden_size + cols, mask, other=0).to(tl.float32)
    weight = tl.load(weight_ptr + cols, mask, other=0).to(tl.float32)
    mean_square = tl.sum(x * x, axis=0) / hidden_size
    output = x * tl.rsqrt(mean_square + eps) * weight
    tl.store(output_ptr + row * hidden_size + cols, output, mask)


def rms_norm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Run RMSNorm with one Triton program per input row."""
    hidden_size = x.shape[-1]
    rows = x.numel() // hidden_size
    output = torch.empty_like(x)
    if rows == 0:
        return output

    block_size = triton.next_power_of_2(hidden_size)
    num_warps = 4 if block_size >= 128 else 1
    _rms_norm_kernel[(rows,)](
        x, weight, output, hidden_size, eps, block_size, num_warps=num_warps
    )
    return output
