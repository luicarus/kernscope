"""Triton GEMV kernel."""

import torch
import triton
import triton.language as tl


@triton.jit
def _gemv_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    hidden_size: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, block_size)
    accumulator = tl.full((block_size,), 0, tl.float32)
    for start in range(0, hidden_size, block_size):
        offsets = start + cols
        mask = offsets < hidden_size
        x = tl.load(x_ptr + offsets, mask, other=0).to(tl.float32)
        weight = tl.load(weight_ptr + row * hidden_size + offsets, mask, other=0).to(tl.float32)
        accumulator += weight * x
    tl.store(output_ptr + row, tl.sum(accumulator, axis=0))


def gemv_triton(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compute one output row per program with four warps."""
    rows, hidden_size = weight.shape
    output = torch.empty(rows, dtype=x.dtype, device=x.device)
    if rows:
        block_size = 1024 if rows >= 512 and hidden_size in (1024, 2048) else 256
        _gemv_kernel[(rows,)](x, weight, output, hidden_size, block_size, num_warps=4)
    return output
