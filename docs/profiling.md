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

## SwiGLU BF16 baseline

Reproduce the SwiGLU baseline from the repository root with:

```bash
python benchmarks/bench_silu_and_mul.py --rows 1 16 128 512 --hidden-size 1024 4096 8192 --dtype bf16 --runs 5 --rep-ms 50 --warmup 20
```

Each case uses an input of shape `rows × 2D` and produces an output of shape `rows × D`. The benchmark compares the public PyTorch eager implementation with Triton through CUDA Graph replay, uses seed `0`, warms up both backends for 20 calls, and alternates their order across five repeats. The reported value is the median of the five per-repeat medians; the raw CSV keeps every repeat median. Each shape records GPU state before and after measurement and the source code hash. The performance scope is this FP32-intermediate reference implementation only; it does not represent `torch.compile`, vLLM, or end-to-end serving performance.

The [SwiGLU results table](silu_and_mul_baseline.md) records Nsight metrics at output shape `512×4096`. The [CUDA Graph CSV](../benchmarks/results/silu_and_mul-baseline-20261003T101637Z.csv) retains all 12 shape combinations and 120 repeat measurements; the [NCU report](../benchmarks/results/silu_and_mul-bf16-512x4096-baseline.ncu-rep) contains three kernel launches. The Triton backend file SHA256 in the CSV identifies the implementation measured before it was committed.

## GEMV BF16 CUDA Graph measurements

```bash
python benchmarks/bench_gemv.py --rows 1 16 128 512 4096 --hidden-size 1024 4096 8192 --dtype bf16 --runs 5 --rep-ms 50 --warmup 20
```

The command measures the currently selected kernel. The archived table below describes the original 1024-column baseline.

`rows=N` is the number of output features, not a batch size. Inputs are `x (H,)` and `weight (N,H)`, with output `(N,)`. Seed is `0`; weights are scaled by `1/sqrt(H)` before timing. Each backend is warmed up for 20 calls, with backend order rotated across five repeats. Summaries take the median of the five per-repeat CUDA Graph medians. TF32 and reduced-precision reductions are disabled and recorded in the CSV.

- `torch_reference`: public `gemv(..., backend="torch")`, including FP32 casts of the input and entire weight matrix.
- `torch_native`: `torch.mv(weight, x)` in the original dtype, without those explicit casts.
- `triton`: public Triton API, converting loaded elements to FP32 inside the kernel.

All paths are checked against the reference before timing. The reference latency includes conversion overhead; the native path is the direct framework performance comparison. These measurements exclude Python dispatch and JIT compilation and are separate from Nsight and serving latency.

| Weight shape N×H | FP32 reference (µs) | Native torch.mv (µs) | Triton (µs) |
|---|---:|---:|---:|
| 1×4096 | 7.961 | 2.539 | 1.487 |
| 16×4096 | 9.634 | 3.756 | 1.629 |
| 128×4096 | 28.909 | 7.733 | 2.885 |
| 512×4096 | 119.358 | 26.440 | 23.935 |
| 4096×4096 | 926.073 | 181.499 | 180.332 |
| 4096×8192 | 1846.998 | 360.884 | 389.890 |

The main configuration's Triton repeat medians span 180.298–180.372 µs. Recorded GPU state samples span 59–78°C and 1695–1957 MHz SM clocks, with memory clocks at 6001 MHz. Native and Triton main-configuration latency are close; at `4096×8192`, Triton is about 8.0% slower.

Environment: RTX 3050 Ti Laptop GPU (4 GB), host driver 610.47, Ubuntu 24.04 / WSL2, Python 3.12.3, PyTorch 2.8.0+cu128, CUDA 12.8, Triton 3.4.0, Nsight Compute 2022.4.1. Source hashes identify the uncommitted implementation at measurement time.

- [CUDA Graph CSV: 15 shapes, 225 samples](../benchmarks/results/gemv-baseline-20261004T001855Z.csv)
- [Nsight report: three launches](../benchmarks/results/gemv-bf16-4096x4096-baseline.ncu-rep)
- [GEMV Nsight table](gemv_nsight_results.md)

## GEMV optimization rounds

The first four rounds keep one output row per program and four warps, reducing the fixed column block from 1024 to 256. The public signature and FP32 calculation contract are unchanged.

| Round | Single variable | Decision |
|---|---|---|
| 1 | Runtime loop trip count | Removes spills at H=8192; the simpler static-block change gives similar profiled latency. Runtime-loop CUDA Graph screening was inconclusive under the then-abnormal GPU state. An unroll hint alone did not change compiled resources. |
| 2 | Column block: 256, 512, 1024, 2048, 4096, 8192 | Select 256: no spills at H=8192, balanced results across the profiled shapes. |
| 3 | Warps: 1, 2, 4, 8, with block=256 | Keep 4; main latency is similar, while 1/2 warps use more registers and 8 regresses some smaller cases. |
| 4 | Weight load cache: default vs `.cg` | Keep default; the main-configuration difference is about 0.2%. |

The [paired NCU report](../benchmarks/results/gemv-bf16-block256-comparison.ncu-rep) and [per-launch CSV](../benchmarks/results/gemv-nsight-block256-20261004.csv) contain only the baseline and selected implementation. Six BF16 weight shapes are covered: `4096×4096`, `4096×8192`, `128×8192`, `16×4096`, `16×1024`, and `1×8192`. Both configurations use the same input/output buffers for each shape, seed 0, weights scaled by `1/sqrt(H)`, alternating order, three launches each, six metrics, seven collection passes, `--cache-control all`, and `--clock-control base`. Each table value is the median of its three samples. CSV launch IDs map directly to NCU IDs; it retains configuration, compiler spill count, source hashes, GPU state and software versions.

At `4096×8192`, kernel duration drops **8.1%** and DRAM traffic **5.8%**; registers drop 64→39, spills 2→0, and achieved occupancy rises 65.55%→97.88%. At `4096×4096`, latency remains 181.376 µs despite higher SM throughput. `128×8192` improves 14.848→13.760 µs; `16×4096` regresses 3.488→3.552 µs (1.8%). Claims are limited to these Nsight measurements. With DRAM throughput near 97% and no useful further warp/cache gain, this profiling round stops here.

The original 181.760 µs Nsight baseline and CUDA Graph CSV remain archived; the wide-H results tables use the paired capture. Original backend SHA256 is `f9aad145f61f4b40c364d92a93e2cf3d1d43db836065a2bfa6a71f0734d36248`; block-256 stage SHA256 is `7312c87b1a2c81767e01c1acceddc3b3c7ca64d6174622bbba3455445ee4bb31`. That stage changes only the launch block constant and its docstring. To reproduce the original baseline, use an isolated copy with that constant set to 1024. Run the profiling command below once per configuration and width with separate export paths.

Earlier disposable CUDA Graph captures were excluded because GPU utilization and thermal slowdown remained elevated without a test process. The accepted follow-up began after idle utilization returned to 0% and software thermal slowdown cleared. Nsight GPU-state samples ranged 80–84°C and 1462–1822 MHz SM clocks, with memory clocks at 6001 MHz; timing methods remain separate. Correctness after the launch change: 222 passed, 7 Ascend tests skipped because `torch_npu` is unavailable.

### CUDA Graph follow-up

The [paired CSV](../benchmarks/results/gemv-block256-comparison-20261004T023022Z.csv) covers all 15 baseline shapes, five repeats per configuration (150 samples), seed 0, 20 warmup calls, 50 ms repetition time, and alternating before/after order on the same inputs. Summaries take the median of five per-repeat medians. Source hashes match the baseline and selected kernels above. Sampled GPU temperatures were 60–74°C, SM clocks 1695–1950 MHz, and memory clocks 6001 MHz; every thermal-state sample was `Not Active`.

| Weight shape N×H | Before, block=1024 (µs) | After, block=256 (µs) | Latency change |
|---|---:|---:|---:|
| 128×8192 | 9.768 | 4.404 | −54.9% |
| 512×1024 | 2.815 | 3.186 | +13.2% |
| 512×8192 | 50.126 | 46.480 | −7.3% |
| 4096×4096 | 180.328 | 180.336 | ≈0% |
| 4096×8192 | 390.071 | 359.431 | −7.9% |

At `4096×8192`, selected repeat medians span 359.394–359.638 µs; all paired ratios show a 7.8–7.9% latency reduction. The fixed block improves the wider reduction dimension while `512×1024` regresses; performance claims are scoped to each measured shape.

The [current three-path CSV](../benchmarks/results/gemv-block256-20261004.csv) independently repeats `4096×4096` and `4096×8192` through the existing benchmark script (30 samples). At `4096×8192`, selected Triton and native `torch.mv` medians are close at 359.497 and 360.838 µs; the conversion-inclusive FP32 reference is 1846.341 µs. These measurements use the original baseline environment and benchmark settings, with sampled temperatures 60–72°C and SM clocks 1695–1935 MHz. Reproduce with:

```bash
python benchmarks/bench_gemv.py --rows 4096 --hidden-size 4096 8192 --dtype bf16 --runs 5 --rep-ms 50 --warmup 20 --output benchmarks/results/gemv-block256-new.csv
```

## GEMV short-H selection, round 5

The current launch uses 1024 columns when `N>=512` and `H` is 1024 or 2048, and 256 otherwise. It keeps one kernel and four warps; selection adds one line to the wrapper. The 256/512/1024-column screening identified gains at these larger-N, short-H cases. Broader short-H rules regressed odd widths, so they were rejected. Compiled BF16 PTX at H=1000 uses two `ld.global.v4.b32` instructions with block=1024 versus eight scalar `ld.global.b32` instructions with block=256; H=1023/1025 loses vector loads. The final rule is restricted to the measured widths.

The [paired Nsight report](../benchmarks/results/gemv-bf16-short-h-comparison.ncu-rep) and [per-launch CSV](../benchmarks/results/gemv-nsight-short-h-20261004.csv) cover `512×1024`, `1024×1024`, `512×2048`, `512×1023`, `4096×4096`, and `4096×8192`, with the same seed, input scaling, fixed buffers, alternating order, three launches, six metrics and cache/clock settings as round 2. The before source hash is `7312c87b1a2c81767e01c1acceddc3b3c7ca64d6174622bbba3455445ee4bb31`; current source hash is `6bb0e3e7cfd443a519cfddaed88ae200ec3bce1be3802ee869a335472f6ffa16`.

With Nsight cache flushing, `512×1024` latency is essentially unchanged (8.000→8.064 µs), registers fall 22→21, and DRAM traffic is similar. `512×2048` is also essentially unchanged (13.760→13.696 µs), with a register tradeoff of 30→36. Wide-H controls remain near 181.6/360.8 µs with no spills. These cold-cache results do not establish a warm-latency speedup.

The interrupted CUDA Graph capture was discarded. Final validation resumed after idle GPU utilization returned to 0% and software thermal slowdown cleared. The [paired CSV](../benchmarks/results/gemv-short-h-comparison-20261004T041246Z.csv) covers 15 shapes and 150 samples, using seed 0, 20 warmup calls, five repeats, 50 ms repetition time, alternating order and the source hashes above. Each value below is the median of five per-repeat CUDA Graph medians; latency change is computed from those two medians. All before/after thermal samples were `Not Active`. Recorded temperatures were 61–77°C, SM clocks 1695–1935 MHz and memory clocks 6001 MHz.

| Weight shape N×H | Before, fixed block=256 (µs) | After, short-H selection (µs) | Latency change |
|---|---:|---:|---:|
| 512×1024 | 3.167 | 2.824 | −10.8% |
| 1024×1024 | 5.722 | 4.850 | −15.2% |
| 512×2048 | 4.438 | 4.282 | −3.5% |
| 4096×4096 | 180.383 | 180.399 | ≈0% |
| 4096×8192 | 359.591 | 359.490 | ≈0% |

At `512×1024`, selected repeat medians span 2.808–2.839 µs; every paired repeat improves by 10.2–11.5%. The result returns to the original block-1024 baseline's 2.839 µs and the later control's 2.815 µs range. At `512×2048`, every paired repeat improves by 3.0–3.9%, with the register increase recorded above. The warm CUDA Graph gain accompanies essentially unchanged cold-cache Nsight latency. Odd H=1023/1025/2047/2049 cases and N=511 continue using block=256; small unchanged cases contain occasional timing outliers, so their raw samples are retained and no uniform speedup is claimed.

The existing benchmark script also produced an [independent three-path CSV](../benchmarks/results/gemv-short-h-20261004T041504Z.csv), covering N=512/1024 and H=1024/2048 (60 samples). At `512×1024`, Triton is 2.797 µs, native `torch.mv` 7.531 µs, and the conversion-inclusive FP32 reference 28.020 µs. GPU-state samples were 61–73°C, SM clocks 1695–1950 MHz, with memory clocks at 6001 MHz; thermal slowdown was inactive at run boundaries. Reproduce the current implementation with:

```bash
python benchmarks/bench_gemv.py --rows 512 1024 --hidden-size 1024 2048 --dtype bf16 --runs 5 --rep-ms 50 --warmup 20 --output benchmarks/results/gemv-short-h-new.csv
```

Correctness with the final rule: **240 passed, 7 Ascend skipped**. GPU cases cover all three dtypes, N=511/512, odd H, H=2048/2049, FP64 references, offset views and read-only input aliasing. The profiling command below runs the current rule; use `--rows 512 --hidden-size 1024` or `2048` for the short-H cases. To reproduce the before configuration, use an isolated copy with block fixed at 256.

## GEMV rows/program, round 6

Tested 1/2/4 output rows per program with the round-5 column rule and four warps fixed. Both grouped candidates passed all 99 GEMV correctness cases. The Nsight comparison showed insufficient latency benefit with higher register/shared-memory use and lower occupancy in the short-H cases. The five-repeat CUDA Graph confirmation was interrupted by abnormal GPU/thermal state and discarded.

Keep the round-5 one-row implementation. Rejected code and disposable results were removed; the retained before/after reports and backend source remain unchanged.

## GEMV register lifetime, round 7

Tested delaying FP32 conversion until both inputs were loaded, with launch parameters fixed. The candidate passed all 99 GEMV correctness cases; separate checks found bit-identical outputs across all three dtypes, representative widths and offset views. Normalized FP32 SASS instructions were unchanged. FP16/BF16 scheduling and register assignment changed, but instruction counts were unchanged and registers mostly stayed the same or increased.

The paired BF16 Nsight comparison showed no useful latency gain at the measured short/wide configurations. Keep the existing explicit conversion after each load. Rejected source, cubins and profiling results were removed; selected kernel and retained reports remain unchanged.

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

ncu --metrics "$metrics" --target-processes all \
  --cache-control all --clock-control base \
  --kernel-name regex:silu_and_mul_kernel --launch-count 3 \
  --export /tmp/kernscope-silu-and-mul \
  python benchmarks/profile_silu_and_mul.py --rows 512 --hidden-size 4096 --dtype bf16 --iterations 3

ncu --metrics "$metrics" --target-processes all \
  --cache-control all --clock-control base \
  --kernel-name regex:gemv_kernel --launch-count 3 \
  --export /tmp/kernscope-gemv \
  python benchmarks/profile_gemv.py --rows 4096 --hidden-size 4096 --dtype bf16 --iterations 3
```

Choose a new export path for subsequent runs. Each report contains three profiled launches. `fused_add_rms_norm` profiling invokes the operator three times on the same tensors, so each call consumes the preceding call's updated values. Open `.ncu-rep` files in Nsight Compute or inspect their raw metrics with:

```bash
ncu --import /tmp/kernscope-fused-rmsnorm.ncu-rep --page raw --csv
```

The results tables for [RMSNorm](rms_norm_nsight_results.md), [SwiGLU](silu_and_mul_baseline.md), and [GEMV](gemv_nsight_results.md) take the median of each metric across three launches; `DRAM Bytes` is `dram__sectors.sum × 32 B`. GEMV uses the same six metrics and seven collection passes per launch. These are profiled kernel measurements, separate from CUDA Graph timings and serving latency. Registers, occupancy, and throughput explain tradeoffs; none alone establishes a speedup.

## Archived reports and implementations

The revisions below identify the corresponding kernel configurations. Baseline implementations remain available in Git history; the current package contains the selected kernels.

| Operator | Configuration | Source revision | NCU report |
|---|---|---|---|
| RMSNorm, before | 1 row/program, 4 warps | `8670ce0` | [Baseline](../benchmarks/results/rms_norm-bf16-512x1024.ncu-rep) |
| RMSNorm, after | 4 rows/program, 8 warps | `71498b4` | [Optimized](../benchmarks/results/rms_norm-bf16-512x1024-rows4-warps8.ncu-rep) |
| Fused RMSNorm, before | 1 row/program, 4 warps | `71498b4` | [Baseline](../benchmarks/results/fused_add_rms_norm-bf16-512x1024.ncu-rep) |
| Fused RMSNorm, after | 1 row/program, 1 warp | `d78efde` | [Optimized](../benchmarks/results/fused_add_rms_norm-bf16-512x1024-warp1.ncu-rep) |

Reproduce historical configurations from their listed source revisions. The current profiling scripts run the current implementations and cannot regenerate a historical baseline by switching a runtime parameter.
