# Kernscope

> A reusable LLM GPU operator library for memory-constrained GPUs, with a versioned SGLang integration for QuantAssay.

**Repository:** `luicarus/kernscope`
**Status:** Milestone 1 in progress: package skeleton and PyTorch RMSNorm reference are in place. CPU correctness cases, Triton kernels, and serving integration remain to be done.

Kernscope aims to be a small, installable library of dependable LLM inference operators for constrained consumer GPUs. Its first integration target is the Qwen3-0.6B serving path measured by [QuantAssay](https://github.com/luicarus/quantassay).

This project is a kernel library. Correctness tests, benchmarks, and profiler reports are how each operator earns its place in the library; they are not the product by themselves.

## Why this project

QuantAssay compares quantized and BF16 models through a live SGLang server. Its current end-to-end result reflects the kernels SGLang actually executes. Kernscope adds a reusable low-level operator backend to that stack, so we can evaluate whether a kernel improves real model serving as well as an isolated operation.

## Goals

- Publish a normal Python package with a small, stable operator API.
- Provide explicit numerical behavior, supported input contracts, and backend selection.
- Keep a PyTorch reference backend for correctness checks and CPU development.
- Add Triton CUDA implementations for selected LLM inference operators.
- Route supported operators into a pinned SGLang runtime, then let QuantAssay measure the resulting serving path.
- Target memory-constrained consumer GPUs, initially the RTX 3050 Ti Laptop GPU with 4 GB of VRAM.

## Initial scope

The first operator family is RMSNorm:

1. `rms_norm`: normalize over the final dimension, accumulate in FP32, and return the input dtype.
2. `fused_add_rms_norm`: add the residual and normalize in one operator, matching the fused path used by transformer layers.

Both operators will begin with a PyTorch reference and then receive Triton implementations. Initial input dtypes are FP16, BF16, and FP32. The first public API is `kernscope.rms_norm(x, weight, eps=1e-6, *, backend="torch")`.

For `rms_norm`, `x` has shape `(..., hidden_size)` and `weight` has shape `(hidden_size,)`. Both must be contiguous tensors on the same device with the same supported dtype. The operator computes `x * rsqrt(mean(x ** 2) + eps) * weight` over the last dimension. `eps` must be finite and positive. Squaring, reduction, and scaling use FP32 values; the result is converted back to `x.dtype`. Leading dimensions are preserved. The initial `torch` backend runs on CPU or on a device supported by PyTorch; later backends will be added behind the same API.

As a hand-check, `x = [3, 4]`, `weight = [1, 1]`, and `eps = 1e-6` produce approximately `[0.8485, 1.1314]`.

The package currently includes only the plain RMSNorm reference. The fused operator, CPU correctness suite, and Triton implementation are still pending.

Next candidates are SwiGLU and GEMV. Attention is later work, selected only when profiling shows that it is a meaningful bottleneck. The first release will not try to replace every kernel in SGLang or implement a complete inference engine.

## QuantAssay and SGLang integration

The operator package stays independent of QuantAssay and SGLang. A separate, versioned adapter connects it to the serving runtime.

QuantAssay sends requests to a live SGLang service, so an operator affects its measurements only when SGLang calls that operator while serving the model. For SGLang 0.5.3, Qwen3 uses SGLang's `RMSNorm` class. Its CUDA path calls `sgl_kernel.rmsnorm` when there is no residual and `sgl_kernel.fused_add_rmsnorm` when there is one. Those call sites provide an initial integration point:

- [SGLang 0.5.3 RMSNorm implementation](https://github.com/sgl-project/sglang/blob/v0.5.3/python/sglang/srt/layers/layernorm.py)
- [SGLang 0.5.3 Qwen3 model](https://github.com/sgl-project/sglang/blob/v0.5.3/python/sglang/srt/models/qwen3.py)

QuantAssay's current launch and benchmark path is in [`gating.py`](https://github.com/luicarus/quantassay/blob/main/src/quantassay/gating.py) and [`serving/benchmark.py`](https://github.com/luicarus/quantassay/blob/main/src/quantassay/serving/benchmark.py). The first integration will target the SGLang version already pinned by QuantAssay; compatibility with other SGLang versions is separate work.

QuantAssay's W4A16 serving path currently uses Marlin for quantized linear layers. For that reason, the first shared operators are normalization operators. Replacing the quantized linear path is a different and substantially larger integration.

### Serving comparison

When the adapter is ready, compare the built-in SGLang backend and Kernscope under the same model, request set, and serving settings:

| Model path | Built-in backend | Kernscope backend |
|---|---:|---:|
| BF16 | measure | measure |
| W4A16 | measure | measure |

This separates the effect of the operator backend from the effect of quantization. Kernel latency and end-to-end serving results will be reported separately.

## Proposed package layout

```text
src/kernscope/
  ops/                 Public operator API and validation
  backends/            PyTorch reference and Triton implementations
  dispatch.py          Backend selection and supported-shape rules

integrations/
  sglang/0.5.3/         Version-specific SGLang adapter

benchmarks/             Reproducible kernel benchmarks
tests/                  CPU reference and GPU correctness checks
```

The core package will not import QuantAssay or SGLang. Integration code will be isolated and version-pinned so using the operators outside QuantAssay does not pull in the serving stack.

## Milestones

1. **Define the library contract:** package metadata, RMSNorm API, PyTorch reference, and CPU correctness cases.
2. **Implement the first Triton kernels:** plain RMSNorm and fused residual RMSNorm, then verify them on an NVIDIA GPU.
3. **Measure and profile:** compare supported shapes and dtypes; retain only optimizations supported by repeatable measurements and profiler evidence.
4. **Integrate with SGLang 0.5.3:** route Qwen3 RMSNorm calls through the library while keeping the built-in path selectable.
5. **Add a QuantAssay backend comparison:** record the selected kernel backend in run identity and reports; compare BF16 and W4A16 with matched settings.
6. **Expand based on evidence:** select SwiGLU, GEMV, or Attention only after identifying a serving bottleneck.

## Completion criteria

- The package can be installed and imported without installing SGLang.
- Every operator has a documented input contract and a PyTorch reference.
- Triton outputs match the reference within a documented tolerance on the target GPU and supported shapes.
- Benchmarks record the actual GPU, driver, framework versions, input shapes, dtype, and selected backend.
- The SGLang adapter makes the serving process execute Kernscope, and the selected backend is visible in QuantAssay artifacts.
- No speedup is claimed without measured kernel-level and end-to-end evidence.

## Target integration baseline

QuantAssay's repository currently documents a tested environment of RTX 3050 Ti Laptop GPU (4 GB), Ubuntu 24.04 under WSL2, Python 3.12, and a pinned SGLang/PyTorch stack. Kernscope will initially target that integration environment rather than promise broad compatibility. This is a target, not evidence that Kernscope has already run there.

## Naming

- Project, distribution, and repository name: **Kernscope** / `kernscope`
- Python import name: `kernscope`
- Initial positioning: **LLM inference operators for memory-constrained GPUs**
