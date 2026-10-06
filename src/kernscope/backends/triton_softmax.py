"""Triton row-wise Softmax kernel."""

import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(x_ptr, output_ptr, rows, hidden_size: tl.constexpr,
                    block_size: tl.constexpr, rows_per_program: tl.constexpr):
    row = tl.program_id(0).to(tl.int64) * rows_per_program + tl.arange(0, rows_per_program)
    cols = tl.arange(0, block_size)
    offsets = row[:, None] * hidden_size + cols[None, :]
    mask = cols[None, :] < hidden_size
    if rows_per_program > 1:
        mask = mask & (row[:, None] < rows)
    x = tl.load(x_ptr + offsets, mask, other=-float("inf")).to(tl.float32)
    numerator = tl.exp(x - tl.max(x, axis=1)[:, None])
    denominator = tl.sum(numerator, axis=1)[:, None]
    tl.store(output_ptr + offsets, numerator / denominator, mask)


def softmax_triton(x: torch.Tensor) -> torch.Tensor:
    """Group large short-H batches, except H=1023; use one warp for H <= 256."""
    hidden_size = x.shape[-1]
    rows = x.numel() // hidden_size
    output = torch.empty_like(x)
    if rows:
        warps = 1 if hidden_size <= 256 else 4
        rows_per_program = 2 if rows >= 512 and hidden_size <= 1024 and hidden_size != 1023 else 1
        _softmax_kernel[(triton.cdiv(rows, rows_per_program),)](
            x, output, rows, hidden_size, triton.next_power_of_2(hidden_size), rows_per_program, num_warps=warps)
    return output
