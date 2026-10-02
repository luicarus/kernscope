# Kernscope

A small LLM inference operator library with PyTorch, Triton CUDA, and TileLang-Ascend backends.

Kernscope provides directly callable inference operators behind a small Python API. Each operator has an explicit numerical contract, reference implementation, correctness tests, and backend-specific kernels.

Current operators:

- **RMSNorm**
- **fused residual-add RMSNorm**

Current backends:

- **PyTorch** — portable reference implementation
- **Triton** — CUDA kernels
- **TileLang-Ascend** — Ascend NPU kernels

## Backend support

| Operator | PyTorch | Triton CUDA | TileLang-Ascend |
|---|---|---|---|
| `rms_norm` | CPU / CUDA | CUDA | Ascend NPU |
| `fused_add_rms_norm` | CPU / CUDA | CUDA | — |

Backend selection is explicit:

```python
from kernscope import rms_norm

y = rms_norm(x, weight, backend="torch")
y = rms_norm(x, weight, backend="triton")
y = rms_norm(x, weight, backend="tilelang_ascend")
```

The public API does not require callers to interact with Triton or TileLang directly.

## Install

Kernscope requires Python 3.10 or newer.

Clone the repository:

```bash
git clone https://github.com/luicarus/kernscope.git
cd kernscope
```

For the PyTorch reference implementations and tests:

```bash
python -m pip install -e ".[test]"
```

For the Triton CUDA backend:

```bash
python -m pip install -e ".[gpu,test]"
```

A CUDA-enabled PyTorch installation is required before using `backend="triton"`.

The TileLang-Ascend backend requires an Ascend PyTorch environment with `torch_npu` and a compatible TileLang-Ascend installation. These dependencies are imported only when `backend="tilelang_ascend"` is selected, so CPU and CUDA users do not need the Ascend stack installed.

## Quick start

### RMSNorm

```python
import torch

from kernscope import rms_norm

x = torch.randn(
    512,
    1024,
    device="cuda",
    dtype=torch.bfloat16,
)
weight = torch.ones(
    1024,
    device="cuda",
    dtype=torch.bfloat16,
)

y = rms_norm(
    x,
    weight,
    eps=1e-6,
    backend="triton",
)
```

For the PyTorch reference:

```python
y = rms_norm(
    x,
    weight,
    eps=1e-6,
    backend="torch",
)
```

On Ascend:

```python
import torch
import torch_npu

from kernscope import rms_norm

x = torch.randn(
    512,
    1024,
    device="npu",
    dtype=torch.bfloat16,
)
weight = torch.ones(
    1024,
    device="npu",
    dtype=torch.bfloat16,
)

y = rms_norm(
    x,
    weight,
    eps=1e-6,
    backend="tilelang_ascend",
)
```

### Fused residual-add RMSNorm

```python
import torch

from kernscope import fused_add_rms_norm

x = torch.randn(
    512,
    1024,
    device="cuda",
    dtype=torch.bfloat16,
)
residual = torch.randn_like(x)
weight = torch.ones(
    1024,
    device="cuda",
    dtype=torch.bfloat16,
)

fused_add_rms_norm(
    x,
    residual,
    weight,
    eps=1e-6,
    backend="triton",
)
```

The fused operator updates **both `x` and `residual` in place** and returns `None`.

`weight` is read-only.

## Numerical contract

### RMSNorm

For input `x` and weight `weight`, RMSNorm computes:

```text
x_fp32 = fp32(x)

scale =
    rsqrt(
        mean(x_fp32², axis=-1)
        + eps
    )

output =
    cast_to_input_dtype(
        x_fp32
        * scale
        * fp32(weight)
    )
```

Reduction and scaling use FP32 intermediates.

### Fused residual-add RMSNorm

For input snapshots `x_in` and `residual_in`:

```text
s =
    fp32(x_in)
    + fp32(residual_in)

residual =
    cast_to_input_dtype(s)

x =
    cast_to_input_dtype(
        s
        * rsqrt(
            mean(s², axis=-1)
            + eps
        )
        * fp32(weight)
    )
```

Normalization uses the FP32 residual sum before it is rounded for storage in `residual`.

## API contract

Both operators accept `eps=1e-6`.

The PyTorch implementation is the default backend:

```python
rms_norm(
    x,
    weight,
    eps=1e-6,
    backend="torch",
)

fused_add_rms_norm(
    x,
    residual,
    weight,
    eps=1e-6,
    backend="torch",
)
```

`rms_norm` currently supports:

```text
torch
triton
tilelang_ascend
```

`fused_add_rms_norm` currently supports:

```text
torch
triton
```

### Inputs

Supported dtypes:

```text
FP16
BF16
FP32
```

`x` has shape:

```text
(..., hidden_size)
```

and `weight` has shape:

```text
(hidden_size,)
```

Requirements:

- the final dimension must be nonempty;
- tensors must be contiguous;
- participating tensors must share dtype and device;
- `eps` must be finite and positive.

For `fused_add_rms_norm`:

- `residual` must have the same shape as `x`;
- `x`, `residual`, and `weight` storage regions must not overlap;
- `x` and `residual` are modified in place.

Triton and TileLang-Ascend kernels are inference-only and reject tensors requiring gradients.

## Triton CUDA backend

The CUDA kernels are implemented in:

```text
src/kernscope/backends/triton.py
```

### RMSNorm

The current RMSNorm implementation processes multiple rows per Triton program.

The kernel:

1. loads the input row into FP32;
2. computes the sum of squares;
3. evaluates the RMS normalization scale;
4. applies the learned weight;
5. writes the result in the original input dtype.

The launch configuration used by the current measured implementation is:

```text
4 rows / program
8 warps / program
```

### Fused residual-add RMSNorm

One Triton program handles one row.

The kernel loads:

```text
x
residual
weight
```

and keeps the residual sum in FP32 through normalization before writing both outputs.

Conceptually:

```text
x ─────────┐
           ├─ FP32 add ───────────────┐
residual ──┘                          │
                                      ├─ RMSNorm ─→ x
                                      │
                                      └────────────→ residual
weight ───────────────────────────────┘
```

Fusing residual addition with RMSNorm removes a separate addition kernel launch and avoids reloading the residual sum from global memory.

The current launch configuration is:

```text
1 row / program
1 warp / program
```

## TileLang-Ascend backend

The Ascend RMSNorm implementation is in:

```text
src/kernscope/backends/tilelang_ascend.py
```

It is compiled through TileLang with:

```python
@tilelang.jit(
    out_idx=[2],
    target="ascendc",
    pass_configs={
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
    },
)
```

The input is flattened into rows:

```text
(..., hidden_size)
        │
        ▼
(rows, hidden_size)
```

and the kernel maps one NPU Vector block to each row.

The main data path is:

```text
Global Memory
     │
     ▼
     UB
     │
     ▼
    FP32
     │
     ├─ square
     ├─ reduce_sum
     ├─ / hidden_size
     ├─ + eps
     └─ rsqrt
     │
     ▼
normalize
     │
     ├─ multiply weight
     ├─ cast to input dtype
     │
     ▼
Global Memory
```

FP16 and BF16 inputs are staged through UB buffers and converted to FP32 before reduction.

The working width is padded to a multiple of 64:

```python
block_size = (
    (hidden_size + 63) // 64
) * 64
```

Padding is zero-filled before reduction, while the reduction itself uses the real hidden size.

The current implementation keeps full-row intermediate buffers in UB. This avoids an additional hidden-dimension tiling path but also makes UB capacity a constraint for sufficiently large hidden sizes.

## CUDA performance

The following compares earlier and current Triton implementations on an **RTX 3050 Ti Laptop GPU (4 GB)** at **512 × 1024 BF16**.

Values are medians of three kernel launches captured by Nsight Compute.

| Operator | Before | Current | Latency reduction | Current configuration |
|---|---:|---:|---:|---|
| RMSNorm | 12.544 µs | 10.176 µs | 18.9% | 4 rows/program, 8 warps |
| Fused add RMSNorm | 20.064 µs | 19.136 µs | 4.6% | 1 row/program, 1 warp |

Recorded environment:

```text
Python        3.12
PyTorch       2.8.0+cu128
CUDA          12.8
Triton        3.4.0
GPU           RTX 3050 Ti Laptop GPU
VRAM          4 GB
Driver        610.47
```

These measurements apply to the recorded workload. Launch settings are currently fixed, so other shapes and dtypes may produce different tradeoffs.

For fused residual-add RMSNorm, the current configuration increases registers per thread from:

```text
29 → 84
```

and lowers achieved occupancy from:

```text
84.93% → 25.95%
```

while still reducing measured kernel latency.

This is also a useful example of why higher occupancy does not necessarily imply lower kernel latency.

The complete profiler data is available in:

[`docs/rms_norm_nsight_results.md`](docs/rms_norm_nsight_results.md)

CUDA Graph benchmark results and Nsight Compute kernel measurements use different timing methods and should only be compared within the same measurement method.

## Correctness

Run the complete test suite with:

```bash
python -m pytest -q
```

CUDA cases automatically skip when CUDA is unavailable.

Ascend hardware-dependent cases automatically skip when `torch_npu`, TileLang, or an available NPU runtime is missing.

### Numerical tolerances

| dtype | `rtol` | `atol` |
|---|---:|---:|
| FP16 | `1e-3` | `1e-3` |
| BF16 | `1e-2` | `1e-2` |
| FP32 | `1e-6` | `1e-6` |

CPU tests are checked against high-precision reference computation.

CUDA and Ascend implementations are checked against the same public RMSNorm numerical contract.

Ascend RMSNorm tests currently cover shapes including:

```text
(5,)
(2, 3, 5)
(3, 1000)
(512, 1024)
```

as well as zero-valued and empty inputs.

Backend dispatch, device validation, autograd rejection, and dependency isolation are tested separately.

## Benchmarks

Run the CUDA RMSNorm benchmark:

```bash
python benchmarks/bench_rms_norm.py \
    --operator rms_norm \
    --rows 512 \
    --runs 5
```

Run fused residual-add RMSNorm:

```bash
python benchmarks/bench_rms_norm.py \
    --operator fused_add_rms_norm \
    --rows 512 \
    --runs 5
```

Benchmarks use CUDA Graph replay and sweep:

```text
FP16
BF16
FP32
```

Timestamped CSV files record metadata including:

```text
device
software versions
shape
dtype
backend
GPU state
```

Fused timing uses zero inputs so repeated in-place calls remain idempotent. Correctness tests use nonzero inputs.

See [`docs/profiling.md`](docs/profiling.md) for Nsight Compute commands, archived reports, and the source revisions corresponding to recorded measurements.

## Repository layout

```text
src/kernscope/
├── ops/
│   ├── rms_norm.py
│   ├── fused_add_rms_norm.py
│   └── ...
│
└── backends/
    ├── triton.py
    └── tilelang_ascend.py

benchmarks/
└── CUDA Graph benchmarks
    and profiling entry points

docs/
├── profiling.md
├── rms_norm_nsight_results.md
└── ...

tests/ops/
├── PyTorch reference tests
├── Triton CUDA tests
├── TileLang-Ascend API tests
└── TileLang-Ascend NPU tests
```

The split between `ops` and `backends` is intentional:

```text
ops/
    public API
    validation
    numerical contract
    PyTorch reference

backends/
    hardware-specific kernel implementations
```

This keeps backend dependencies outside the public operator definitions until the corresponding backend is selected.

## Scope

Kernscope is a small operator library rather than a serving framework.

Serving-specific integration belongs in consumer projects such as [QuantAssay](https://github.com/luicarus/quantassay). Kernscope can be installed and imported without SGLang or QuantAssay.

Current coverage is:

```text
RMSNorm
├── PyTorch
├── Triton CUDA
└── TileLang-Ascend

Fused residual-add RMSNorm
├── PyTorch
└── Triton CUDA
```

An [earlier SGLang serving comparison](docs/rms_norm_before_kernel_optimization.md) predates the current Triton optimizations and did not establish an end-to-end serving speedup. The optimized kernels have not yet been re-evaluated in that workload.

Additional operators and backend implementations are added around concrete inference workloads rather than maintained as standalone kernel demos.
