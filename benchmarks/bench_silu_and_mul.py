import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import torch
import triton
import triton.testing

from _environment import GPU_FIELDS, gpu_state, source_metadata
from kernscope import silu_and_mul


DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
TOLERANCES = {"fp16": 1e-3, "bf16": 1e-2, "fp32": 1e-6}
ROOT = Path(__file__).resolve().parents[1]
KERNEL_SOURCE = ROOT / "src/kernscope/backends/triton.py"


def main():
    parser = argparse.ArgumentParser(description="Benchmark PyTorch and Triton SwiGLU backends")
    parser.add_argument("--rows", nargs="+", type=int, default=[1, 16, 128, 512])
    parser.add_argument("--hidden-size", nargs="+", type=int, default=[1024, 4096, 8192])
    parser.add_argument("--dtype", choices=DTYPES, default="bf16")
    parser.add_argument("--rep-ms", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    if (
        any(rows < 1 for rows in args.rows)
        or any(hidden_size < 1 for hidden_size in args.hidden_size)
        or args.rep_ms < 1
        or args.warmup < 0
        or args.runs < 1
    ):
        parser.error("rows, hidden size, runs, and repetition time must be positive; warmup must be nonnegative")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path("benchmarks/results") / f"silu_and_mul-baseline-{timestamp}.csv"
    if output.exists() and not args.overwrite:
        parser.error(f"{output} exists; choose a new path or pass --overwrite")

    torch.manual_seed(0)
    kernel_hash, source_revision, dirty = source_metadata(KERNEL_SOURCE)
    versions = [torch.__version__, torch.version.cuda or "unknown", triton.__version__]
    results = []
    with torch.inference_mode():
        for rows in args.rows:
            for hidden_size in args.hidden_size:
                input_width = 2 * hidden_size
                x = torch.randn(rows, input_width, device="cuda", dtype=DTYPES[args.dtype])
                fns = {
                    backend: (lambda backend=backend: silu_and_mul(x, backend=backend))
                    for backend in ("torch", "triton")
                }
                expected = fns["torch"]()
                torch.testing.assert_close(
                    fns["triton"](), expected,
                    rtol=TOLERANCES[args.dtype], atol=TOLERANCES[args.dtype],
                )
                for fn in fns.values():
                    for _ in range(args.warmup):
                        fn()
                torch.cuda.synchronize()
                state_before = gpu_state()
                for repeat in range(1, args.runs + 1):
                    order = ("torch", "triton") if repeat % 2 else ("triton", "torch")
                    for backend in order:
                        median_us = triton.testing.do_bench_cudagraph(
                            fns[backend], rep=args.rep_ms, return_mode="median"
                        ) * 1000
                        results.append(
                            [
                                timestamp, repeat, *state_before, *versions, rows, hidden_size,
                                input_width, args.dtype, backend, "cuda_graph", median_us,
                                args.warmup, args.runs, args.rep_ms,
                                kernel_hash, source_revision, dirty,
                            ]
                        )
                torch.cuda.synchronize()
                state_after = gpu_state()
                for row in results[-2 * args.runs :]:
                    row.extend(state_after)

    header = (
        ["timestamp_utc", "repeat"]
        + [f"{field}_before" for field in GPU_FIELDS]
        + [
            "torch", "cuda", "triton", "rows", "hidden_size", "input_width", "dtype",
            "backend", "measurement", "median_us", "warmup", "runs", "rep_ms",
            "kernel_source_sha256", "source_revision",
            "dirty",
        ]
        + [f"{field}_after" for field in GPU_FIELDS]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(results)
    print(output)


if __name__ == "__main__":
    main()
