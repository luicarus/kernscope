# Kernscope Repository Guide

Read [README.md](README.md) for the supported APIs, environment, and measured results. If `AGENTS.local.md` exists, read it for local workspace preferences; keep that file outside version control.

## Scope and structure

- Keep Kernscope a directly callable LLM inference operator library.
- Public APIs, validation, and PyTorch references live in `src/kernscope/ops/`; accelerator kernels live in `src/kernscope/backends/`.
- Keep serving-framework dependencies and version-specific adapters in consumer projects.
- Add operators when a measured workload or an agreed milestone justifies them.

## Implementation conventions

- Establish a PyTorch reference and input contract before adding a Triton implementation.
- Preserve the public signatures and mutation behavior documented in the README.
- Support FP16, BF16, and FP32 with FP32 intermediates and input-dtype outputs.
- Keep the PyTorch references for `rms_norm`, `silu_and_mul`, and `gemv` usable on CPU and with autograd. Triton and other accelerator backends are inference-only.
- `fused_add_rms_norm` is an in-place, inference-only PyTorch and Triton operator. It updates both `x` and `residual`, returns `None`, and normalizes the FP32 sum before storage rounding. Its input storage regions must not overlap.
- `silu_and_mul` returns a new tensor, leaves `x` unchanged, and computes `silu(gate) * up` in FP32 before casting to the input dtype.
- `gemv` reads `x (H,)` and `weight (N, H)` and returns a new `(N,)` tensor. `H` is nonempty; `N=0` returns an empty output. Read-only inputs may share storage.
- Prefer concise, readable code and short comments. Keep validation errors clear and avoid abstractions without a concrete use.

## Validation

- Install development dependencies with `python -m pip install -e ".[gpu,test]"`; use `.[test]` for CPU development.
- Run relevant correctness checks after numerical, launch, or API changes: `python -m pytest -q`.
- GPU tests skip without CUDA; report skips accurately when describing validation.
- Compare both outputs of `fused_add_rms_norm` and preserve documented tolerances: FP16 `1e-3`, BF16 `1e-2`, FP32 `1e-6` for both `rtol` and `atol`.

## Performance evidence

- Record GPU, driver, framework versions, shape, dtype, backend, and measurement method.
- Distinguish CUDA Graph latency, Nsight Compute profiling, and end-to-end serving results. Scope claims to the measured configuration.
- Keep reproducible before/after reports and document changes in sampling or aggregation.
- Use temporary locations for parameter sweeps. Remove rejected implementations and disposable results; retain the selected runtime configuration and its supporting evidence.
- Consult [docs/profiling.md](docs/profiling.md) for profiling commands and result locations.
