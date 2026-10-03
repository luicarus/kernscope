"""Triton CUDA operator kernels."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_four_rows_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    rows,
    hidden_size: tl.constexpr,
    eps: tl.constexpr,
    block_size: tl.constexpr,
):
    row_ids = tl.program_id(0) * 4 + tl.arange(0, 4)
    cols = tl.arange(0, block_size)
    mask = (row_ids[:, None] < rows) & (cols[None, :] < hidden_size)
    x = tl.load(
        x_ptr + row_ids[:, None] * hidden_size + cols[None, :], mask, other=0
    ).to(tl.float32)
    weight = tl.load(weight_ptr + cols, cols < hidden_size, other=0).to(tl.float32)
    mean_square = tl.sum(x * x, axis=1) / hidden_size
    output = x * tl.rsqrt(mean_square[:, None] + eps) * weight[None, :]
    tl.store(output_ptr + row_ids[:, None] * hidden_size + cols[None, :], output, mask)


def rms_norm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Run RMSNorm with four rows per program and eight warps."""
    hidden_size = x.shape[-1]
    rows = x.numel() // hidden_size
    output = torch.empty_like(x)
    if rows == 0:
        return output

    block_size = triton.next_power_of_2(hidden_size)
    _rms_norm_four_rows_kernel[(triton.cdiv(rows, 4),)](
        x, weight, output, rows, hidden_size, eps, block_size,
        num_warps=8,
    )
    return output


@triton.jit
def _fused_add_rms_norm_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    hidden_size: tl.constexpr,
    eps: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, block_size)
    mask = cols < hidden_size
    x = tl.load(x_ptr + row * hidden_size + cols, mask, other=0).to(tl.float32)
    residual = tl.load(residual_ptr + row * hidden_size + cols, mask, other=0).to(tl.float32)
    weight = tl.load(weight_ptr + cols, mask, other=0).to(tl.float32)
    summed = x + residual
    mean_square = tl.sum(summed * summed, axis=0) / hidden_size
    normalized = summed * tl.rsqrt(mean_square + eps) * weight
    tl.store(residual_ptr + row * hidden_size + cols, summed, mask)
    tl.store(x_ptr + row * hidden_size + cols, normalized, mask)


def fused_add_rms_norm_triton(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> None:
    """Fuse residual addition and RMSNorm in place, one warp per program."""
    hidden_size = x.shape[-1]
    rows = x.numel() // hidden_size
    if rows == 0:
        return None

    block_size = triton.next_power_of_2(hidden_size)
    _fused_add_rms_norm_kernel[(rows,)](
        x, residual, weight, hidden_size, eps, block_size, num_warps=1
    )


@triton.jit
def _silu_and_mul_kernel(
    x_ptr,
    output_ptr,
    elements,
    hidden_size: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < elements
    input_offsets = (offsets // hidden_size) * (2 * hidden_size) + offsets % hidden_size
    gate = tl.load(x_ptr + input_offsets, mask, other=0).to(tl.float32)
    up = tl.load(x_ptr + input_offsets + hidden_size, mask, other=0).to(tl.float32)
    output = gate * tl.sigmoid(gate) * up
    tl.store(output_ptr + offsets, output, mask)


def silu_and_mul_triton(x: torch.Tensor) -> torch.Tensor:
    """Fuse SiLU and multiplication in blocks of 1024 output elements."""
    hidden_size = x.shape[-1] // 2
    output = torch.empty((*x.shape[:-1], hidden_size), dtype=x.dtype, device=x.device)
    elements = output.numel()
    if elements == 0:
        return output

    _silu_and_mul_kernel[(triton.cdiv(elements, 1024),)](
        x, output, elements, hidden_size, 1024, num_warps=4
    )
    return output
