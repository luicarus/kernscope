"""TileLang-Ascend RMSNorm backend."""

import torch
import torch_npu  # Registers the NPU device with PyTorch.
import tilelang
from tilelang import language as T
from torch import Tensor

_DTYPES = {torch.float16: "float16", torch.bfloat16: "bfloat16", torch.float32: "float32"}
_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
}


@tilelang.jit(out_idx=[2], target="ascendc", pass_configs=_PASS_CONFIGS)
def _rms_norm_kernel(rows: int, hidden_size: int, dtype: str, eps: float):
    """Compile one Vector block per row with FP32 intermediates."""
    block_size = ((hidden_size + 63) // 64) * 64

    @T.prim_func
    def rms_norm(
        x: T.Tensor((rows, hidden_size), dtype),
        weight: T.Tensor((hidden_size,), dtype),
        output: T.Tensor((rows, hidden_size), dtype),
    ):
        with T.Kernel(rows, threads=1, is_npu=True) as row:
            x_ub = T.alloc_ub((1, block_size), dtype)
            weight_ub = T.alloc_ub((block_size,), dtype)
            x_fp32 = T.alloc_ub((1, block_size), "float32")
            weight_fp32 = T.alloc_ub((block_size,), "float32")
            work = T.alloc_ub((1, block_size), "float32")
            mean_square = T.alloc_ub((1, 1), "float32")
            scale = T.alloc_ub((1, 1), "float32")

            # Zero FP32 padding and copy/cast only valid elements.
            T.tile.fill(x_fp32, 0.0)
            T.tile.fill(weight_fp32, 0.0)
            if dtype == "float32":
                T.copy(x[row : row + 1, 0:hidden_size], x_fp32[:, 0:hidden_size])
                T.copy(weight[0:hidden_size], weight_fp32[0:hidden_size])
            else:
                T.copy(x[row : row + 1, 0:hidden_size], x_ub[:, 0:hidden_size])
                T.copy(weight[0:hidden_size], weight_ub[0:hidden_size])
                T.tile.cast(x_fp32, x_ub, "CAST_NONE", hidden_size)
                T.tile.cast(weight_fp32, weight_ub, "CAST_NONE", hidden_size)

            T.tile.mul(work, x_fp32, x_fp32)
            T.reduce_sum(work, mean_square, dim=-1, real_shape=[1, hidden_size])
            T.tile.mul(mean_square, mean_square, 1.0 / hidden_size)
            T.tile.add(mean_square, mean_square, eps)
            T.tile.rsqrt(scale, mean_square)
            T.tile.broadcast(work, scale)
            T.tile.mul(x_fp32, x_fp32, work)
            T.tile.mul(x_fp32, x_fp32, weight_fp32)

            if dtype == "float32":
                T.copy(x_fp32[:, 0:hidden_size], output[row : row + 1, 0:hidden_size])
            else:
                T.tile.cast(x_ub, x_fp32, "CAST_RINT", hidden_size)
                T.copy(x_ub[:, 0:hidden_size], output[row : row + 1, 0:hidden_size])

    return rms_norm


def rms_norm_tilelang_ascend(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    """Flatten contiguous input rows and restore the kernel output shape."""
    hidden_size = x.shape[-1]
    rows = x.numel() // hidden_size
    if rows == 0:
        return torch.empty_like(x)

    kernel = _rms_norm_kernel(rows, hidden_size, _DTYPES[x.dtype], eps)
    return kernel(x.view(rows, hidden_size), weight).reshape_as(x)
