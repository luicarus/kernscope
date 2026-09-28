import argparse
import csv
import subprocess
import sys
from pathlib import Path

import torch
import triton
import triton.testing

from kernscope import rms_norm

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark PyTorch and Triton RMSNorm backends"
    )
    parser.add_argument("--rows", nargs="+", type=int, default=[1, 16, 128, 512])
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--rep-ms", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    if args.hidden_size < 1 or args.rep_ms < 1 or any(rows < 1 for rows in args.rows):
        parser.error("rows, hidden size, and repetition time must be positive")
    torch.manual_seed(0)

    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    metadata = [
        torch.cuda.get_device_name(),
        driver or "unknown",
        torch.__version__,
        torch.version.cuda or "unknown",
        triton.__version__,
    ]
    results = []
    for rows in args.rows:
        for dtype_name, dtype in DTYPES.items():
            x = torch.randn(rows, args.hidden_size, device="cuda", dtype=dtype)
            weight = torch.randn(args.hidden_size, device="cuda", dtype=dtype)
            for backend in ("torch", "triton"):
                fn = lambda: rms_norm(x, weight, backend=backend)
                fn()
                torch.cuda.synchronize()
                latency_us = triton.testing.do_bench_cudagraph(
                    fn, rep=args.rep_ms, return_mode="median"
                ) * 1000
                results.append(
                    metadata + [rows, args.hidden_size, dtype_name, backend, latency_us]
                )

    header = [
        "gpu", "driver", "torch", "cuda", "triton", "rows",
        "hidden_size", "dtype", "backend", "median_us",
    ]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(results)
    else:
        writer = csv.writer(sys.stdout)
        writer.writerow(header)
        writer.writerows(results)


if __name__ == "__main__":
    main()
