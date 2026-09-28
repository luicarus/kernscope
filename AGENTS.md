# Kernscope Agent Guide

## Project goals

- Build a reusable LLM inference operator library for memory-constrained GPUs.
- Treat Kernscope as a long-term learning project and a portfolio project for GPU kernels and operators.
- Make correctness, benchmarks, profiler evidence, and real serving integration support every performance claim.
- Keep the core package independent of SGLang and QuantAssay; put runtime adapters in versioned integration directories.

## Working style

- Do not use Superpowers skills in this project.
- Work in small, understandable steps and explain the relevant CUDA, Triton, numerical, or profiling concept briefly as it comes up.
- Prefer concise code. Keep comments and docstrings to one line when that is clear; avoid boilerplate and premature abstractions.
- Keep validation and error messages clear. Do not shorten code at the cost of readability or correctness.
- Follow the README's scope. Add operators only when profiling or the agreed milestone justifies them.

## Operator contract

- Keep public operators in `src/kernscope/ops/` and backend selection in the core package.
- Start with a PyTorch reference, then add a Triton implementation against the same contract.
- Current `rms_norm` API: `rms_norm(x, weight, eps=1e-6, *, backend="torch")`; use `backend="triton"` for the CUDA kernel.
- RMSNorm inputs have shape `(..., hidden_size)` and `weight` has shape `(hidden_size,)`; inputs are contiguous and share device and dtype.
- Support FP16, BF16, and FP32. Accumulate in FP32 and return the input dtype. Reject invalid shapes, dtypes, devices, and epsilon values clearly.
- Keep the plain RMSNorm PyTorch backend usable on CPU and with autograd; the Triton backend is CUDA-only and inference-only.
- `fused_add_rms_norm(x, residual, weight, eps=1e-6, *, backend="torch")` mutates `residual` to the sum and `x` to its normalized result, then returns `None`; use `backend="triton"` for CUDA. Use FP32 intermediates, preserve input dtypes, require matching contiguous inputs with non-overlapping storage, and keep it inference-only.
- Keep Kernscope independent of serving frameworks; version-specific adapters belong in consumer projects such as QuantAssay.

## Evidence and claims

- Document input contracts and tolerances for every operator.
- Record GPU, driver, framework versions, shape, dtype, and backend in benchmark results.
- Report kernel latency separately from end-to-end serving results.
- Do not claim a speedup without repeatable measurements; include profiler evidence when explaining an optimization.
- Keep examples and reports reproducible so the work can support a credible résumé description.
