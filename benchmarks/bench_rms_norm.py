import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import torch
import triton
import triton.testing

from kernscope import fused_add_rms_norm, rms_norm
from _environment import GPU_FIELDS, gpu_state

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def main():
    parser = argparse.ArgumentParser(description="Benchmark PyTorch and Triton RMSNorm backends")
    parser.add_argument("--operator", choices=("rms_norm", "fused_add_rms_norm"), default="rms_norm")
    parser.add_argument("--rows", nargs="+", type=int, default=[1, 16, 128, 512])
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--rep-ms", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    if (
        args.hidden_size < 1
        or args.rep_ms < 1
        or args.warmup < 0
        or args.runs < 1
        or any(rows < 1 for rows in args.rows)
    ):
        parser.error("rows, hidden size, runs, and repetition time must be positive")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path("benchmarks/results") / f"{args.operator}-{timestamp}.csv"
    if output.exists() and not args.overwrite:
        parser.error(f"{output} exists; choose a new path or pass --overwrite")

    torch.manual_seed(0)
    state_before = gpu_state()
    versions = [torch.__version__, torch.version.cuda or "unknown", triton.__version__]
    results = []
    for rows in args.rows:
        for dtype_name, dtype in DTYPES.items():
            if args.operator == "rms_norm":
                x = torch.randn(rows, args.hidden_size, device="cuda", dtype=dtype)
                residual = None
            else:
                x = torch.zeros(rows, args.hidden_size, device="cuda", dtype=dtype)
                residual = torch.zeros_like(x)
            weight = torch.randn(args.hidden_size, device="cuda", dtype=dtype)
            for backend in ("torch", "triton"):
                if args.operator == "rms_norm":
                    fn = lambda: rms_norm(x, weight, backend=backend)
                else:
                    fn = lambda: fused_add_rms_norm(x, residual, weight, backend=backend)
                for _ in range(args.warmup):
                    fn()
                torch.cuda.synchronize()
                for repeat in range(1, args.runs + 1):
                    latency_us = triton.testing.do_bench_cudagraph(
                        fn, rep=args.rep_ms, return_mode="median"
                    ) * 1000
                    results.append(
                        [timestamp, repeat, *state_before, *versions, args.operator, rows,
                         args.hidden_size, dtype_name, backend, latency_us]
                    )

    state_after = gpu_state()
    header = [
        "timestamp_utc", "repeat", *GPU_FIELDS[:2],
        "pstate_before", "sm_clock_before_mhz", "memory_clock_before_mhz",
        "power_before_w", "temperature_before_c", "torch", "cuda", "triton",
        "operator", "rows", "hidden_size", "dtype", "backend", "median_us",
        "pstate_after", "sm_clock_after_mhz", "memory_clock_after_mhz",
        "power_after_w", "temperature_after_c",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(row + state_after[2:] for row in results)
    print(output)


if __name__ == "__main__":
    main()
