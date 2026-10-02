# Kernscope

LLM inference operators with PyTorch references and Triton CUDA kernels, developed on a 4 GB NVIDIA GPU.

Kernscope currently provides **RMSNorm** and **fused residual-add RMSNorm** as directly callable Python operators. Each has an explicit numerical contract, correctness checks, microbenchmarks, and Nsight Compute reports. The package depends on PyTorch, with Triton as an optional GPU dependency.

## Install from source

The recorded GPU environment is Ubuntu 24.04 under WSL2 with CUDA-enabled PyTorch. Install a CUDA-enabled PyTorch build before using the Triton backend.

```bash
git clone https://github.com/luicarus/kernscope.git
cd kernscope
python -m pip install -e ".[gpu,test]"
```

For CPU reference implementations and tests, install `python -m pip install -e ".[test]"`. The package requires Python 3.10 or newer.

## Example: fused residual-add RMSNorm

```python
import torch
from kernscope import fused_add_rms_norm

x = torch.randn(512, 1024, device="cuda", dtype=torch.bfloat16)
residual = torch.randn_like(x)
weight = torch.ones(1024, device="cuda", dtype=torch.bfloat16)

fused_add_rms_norm(x, residual, weight, eps=1e-6, backend="triton")
# residual now contains the residual sum; x contains its normalized output.
```

The call updates **both `x` and `residual` in place** and returns `None`. `weight` is read-only. For the PyTorch reference, use `backend="torch"`; this also supports CPU tensors.

For input snapshots `x_in` and `residual_in`, the fused operator computes:

```text
s        = fp32(x_in) + fp32(residual_in)
residual = cast_to_input_dtype(s)
x        = cast_to_input_dtype(s * rsqrt(mean(s², axis=-1) + eps) * fp32(weight))
```

Normalization uses the FP32 sum before it is rounded for storage in `residual`.

## Operator contract

Both functions accept `eps=1e-6` and keyword-only `backend="torch"` by default. Select the GPU implementation explicitly with `backend="triton"`.

The plain RMSNorm API also recognizes experimental `backend="tilelang_ascend"` for NPU inference. It uses a per-row Vector kernel with FP32 intermediates and full-row UB buffers. The kernel has not yet been compiled or validated on Ascend hardware; very large hidden sizes may exceed UB capacity. This backend requires `torch_npu` and the Ascend build of TileLang, imported only when selected. NPU correctness cases skip when NPU dependencies or hardware are unavailable.

| Call | Result | PyTorch backend | Triton backend |
|---|---|---|---|
| `rms_norm(x, weight)` | New tensor; preserves input shape and dtype | CPU/CUDA, autograd | CUDA, inference only |
| `fused_add_rms_norm(x, residual, weight)` | Updates `x` and `residual`; returns `None` | CPU/CUDA, inference only | CUDA, inference only |

- Inputs use FP16, BF16, or FP32. Addition, reduction, and scaling use FP32 intermediates.
- `x` has shape `(..., hidden_size)` with a nonempty final dimension; `weight` has shape `(hidden_size,)`. Tensors must be contiguous and share dtype and device. `eps` must be finite and positive.
- The fused operator requires `residual` to match `x`. The storage regions of `x`, `residual`, and `weight` must not overlap. Neither the fused operator nor the Triton backend accepts tensors requiring gradients.

## Inside the fused kernel

One Triton program handles one row, using one warp. It loads `x`, `residual`, and `weight`, keeps the FP32 sum through the reduction, then writes both outputs. Combining these operations in one kernel avoids a separate addition launch and a reload of the sum for normalization.

<details>
<summary>Show the Triton implementation</summary>

From [src/kernscope/backends/triton.py](src/kernscope/backends/triton.py):

```python
import triton
import triton.language as tl


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
```

The launcher uses a grid of `x.numel() // x.shape[-1]` programs, a power-of-two column block, and `num_warps=1`. Masks handle the padded columns.

</details>

## Measured kernel optimization

The following compares earlier and current **Triton implementations** on an RTX 3050 Ti Laptop GPU (4 GB), at **512×1024 BF16**. Values are medians of three kernel launches captured by Nsight Compute.

| Operator | Before (µs) | After (µs) | Latency reduction | Current launch configuration |
|---|---:|---:|---:|---|
| RMSNorm | 12.544 | 10.176 | 18.9% | 4 rows/program, 8 warps |
| Fused add RMSNorm | 20.064 | 19.136 | 4.6% | 1 row/program, 1 warp |

The recorded stack uses Python 3.12, PyTorch 2.8.0+cu128, CUDA 12.8, Triton 3.4.0, and host driver 610.47. These gains apply to the measured workload. Launch settings are fixed; other shapes and dtypes can have different tradeoffs, including regressions relative to earlier configurations.

The fused configuration increases registers per thread from 29 to 84 and lowers achieved occupancy from 84.93% to 25.95%, while reducing profiled latency. The [full metrics table](docs/rms_norm_nsight_results.md) records these tradeoffs. CUDA Graph timings are separate measurements; profiling and benchmark latencies should be compared within their respective methods.

## Reproduce the results

```bash
python -m pytest -q
python benchmarks/bench_rms_norm.py --operator rms_norm --rows 512 --runs 5
python benchmarks/bench_rms_norm.py --operator fused_add_rms_norm --rows 512 --runs 5
```

Tests compare CPU outputs with FP64 math and GPU outputs with the PyTorch references. The `rtol` and `atol` values are both `1e-3` for FP16, `1e-2` for BF16, and `1e-6` for FP32. GPU cases skip when CUDA is unavailable.

Benchmarks use CUDA Graph replay, sweep FP16/BF16/FP32, and write timestamped CSVs with device, software, shape, dtype, backend, and GPU state metadata. Fused timing uses zero inputs so repeated in-place calls are idempotent; correctness checks use nonzero inputs.

See the [profiling guide](docs/profiling.md) for Nsight Compute commands, archived reports, and the source revisions behind the comparison.

## Repository layout

```text
src/kernscope/
  ops/                 Public APIs, validation, and PyTorch references
  backends/triton.py   Triton CUDA kernels
benchmarks/           CUDA Graph benchmarks and Nsight Compute entry points
docs/                 Measurement results and profiling instructions
tests/ops/            CPU and GPU correctness checks
```

## Integration and scope

Serving adapters belong in consumer projects such as [QuantAssay](https://github.com/luicarus/quantassay). Kernscope can be imported and called without installing SGLang or QuantAssay.

An [earlier SGLang serving comparison](docs/rms_norm_before_kernel_optimization.md) predates the kernel optimizations above and did not establish an end-to-end speedup. The optimized kernels have not yet been re-evaluated in that serving workload.

SwiGLU and GEMV are candidates for future operators, subject to a measured workload and an agreed scope. Contribution and agent conventions are documented in [AGENTS.md](AGENTS.md).
