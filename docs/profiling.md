# Benchmarking and profiling

The archived optimization comparison targets 512×1024 BF16 on an NVIDIA GeForce RTX 3050 Ti Laptop GPU with 4 GB of memory. The recorded environment is Ubuntu 24.04 under WSL2, Python 3.12, PyTorch 2.8.0+cu128, CUDA 12.8, Triton 3.4.0, and Windows host driver 610.47.

## CUDA Graph microbenchmarks

After installing `.[gpu,test]`, run from the repository root:

```bash
python benchmarks/bench_rms_norm.py --operator rms_norm --rows 512 --hidden-size 1024 --rep-ms 50 --runs 5
python benchmarks/bench_rms_norm.py --operator fused_add_rms_norm --rows 512 --hidden-size 1024 --rep-ms 50 --runs 5
```

Each command measures the PyTorch and Triton backends in FP16, BF16, and FP32. It warms up each backend, records independent CUDA Graph timing medians, and writes a new timestamped CSV under `benchmarks/results/`. An existing output file is preserved unless `--overwrite` is passed.

Fused timing uses zero-valued `x` and `residual` so repeated in-place calls remain idempotent. Correctness tests use nonzero inputs and compare both outputs. The CSV records device and software versions plus GPU clocks, power, and temperature before and after the run. Check these alongside repeat variability before attributing a difference to a kernel change.

Archived CUDA Graph measurements:

- [RMSNorm baseline](../benchmarks/results/rms_norm-20260928T124252Z.csv)
- [Fused RMSNorm baseline](../benchmarks/results/fused_add_rms_norm-20260928T133037Z.csv)
- [Fused RMSNorm with one warp](../benchmarks/results/fused_add_rms_norm-20260929T150206Z.csv)

## Nsight Compute

Nsight Compute must be installed separately and have access to GPU performance counters. The following Bash commands capture the current kernels into new report paths outside the repository:

```bash
metrics="gpu__time_duration.sum,dram__sectors.sum,launch__registers_per_thread,sm__warps_active.avg.pct_of_peak_sustained_active,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"

ncu --metrics "$metrics" --target-processes all \
  --kernel-name regex:rms_norm_four_rows_kernel --launch-count 3 \
  --export /tmp/kernscope-rmsnorm \
  python benchmarks/profile_rms_norm.py --rows 512 --hidden-size 1024 --dtype bf16 --iterations 3

ncu --metrics "$metrics" --target-processes all \
  --kernel-name regex:fused_add_rms_norm_kernel --launch-count 3 \
  --export /tmp/kernscope-fused-rmsnorm \
  python benchmarks/profile_fused_add_rms_norm.py --rows 512 --hidden-size 1024 --dtype bf16 --iterations 3
```

Choose a new export path for subsequent runs. Both reports contain three profiled launches. Fused profiling invokes the operator three times on the same tensors, so each call consumes the preceding call's updated values. Open `.ncu-rep` files in Nsight Compute or inspect their raw metrics with:

```bash
ncu --import /tmp/kernscope-fused-rmsnorm.ncu-rep --page raw --csv
```

The [results table](rms_norm_nsight_results.md) takes the median of each metric across the three launches. `DRAM Bytes` is `dram__sectors.sum × 32 B`; latency reduction is `(before - after) / before`. These are profiled kernel measurements, separate from CUDA Graph timings and serving latency. Registers, occupancy, and throughput explain tradeoffs; none alone establishes a speedup.

## Archived reports and implementations

The revisions below identify the corresponding kernel configurations. Baseline implementations remain available in Git history; the current package contains the selected kernels.

| Operator | Configuration | Source revision | NCU report |
|---|---|---|---|
| RMSNorm, before | 1 row/program, 4 warps | `8670ce0` | [Baseline](../benchmarks/results/rms_norm-bf16-512x1024.ncu-rep) |
| RMSNorm, after | 4 rows/program, 8 warps | `71498b4` | [Optimized](../benchmarks/results/rms_norm-bf16-512x1024-rows4-warps8.ncu-rep) |
| Fused RMSNorm, before | 1 row/program, 4 warps | `71498b4` | [Baseline](../benchmarks/results/fused_add_rms_norm-bf16-512x1024.ncu-rep) |
| Fused RMSNorm, after | 1 row/program, 1 warp | `d78efde` | [Optimized](../benchmarks/results/fused_add_rms_norm-bf16-512x1024-warp1.ncu-rep) |

Reproduce historical configurations from their listed source revisions. The current profiling scripts run the current implementations and cannot regenerate a historical baseline by switching a runtime parameter.
