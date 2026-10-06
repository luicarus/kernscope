import argparse
import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import torch
import triton
import triton.testing

from _environment import GPU_FIELDS, gpu_state, source_metadata, thermal_state
from kernscope import softmax

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
TOLERANCES = {"fp16": 1e-3, "bf16": 1e-2, "fp32": 1e-6}
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Benchmark Softmax reference, native PyTorch, and Triton")
    parser.add_argument("--rows", nargs="+", type=int, default=[1, 16, 128, 512])
    parser.add_argument("--hidden-size", nargs="+", type=int, default=[128, 1024, 4096, 8192])
    parser.add_argument("--dtype", choices=DTYPES, default="bf16")
    parser.add_argument("--rep-ms", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    if (any(value < 1 for value in args.rows + args.hidden_size)
            or args.rep_ms < 1 or args.runs < 1 or args.warmup < 0):
        parser.error("dimensions, runs, and repetition time must be positive; warmup must be nonnegative")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path("benchmarks/results") / f"softmax-{timestamp}.csv"
    if output.exists() and not args.overwrite:
        parser.error(f"{output} exists; choose a new path or pass --overwrite")
    torch.manual_seed(0)
    kernel_hash, revision, dirty = source_metadata(ROOT / "src/kernscope/backends/triton_softmax.py")
    reference_hash = hashlib.sha256((ROOT / "src/kernscope/ops/softmax.py").read_bytes()).hexdigest()
    versions = [torch.__version__, torch.version.cuda or "unknown", triton.__version__]
    results = []
    with torch.inference_mode(), torch.autocast("cuda", enabled=False):
        for rows in args.rows:
            for width in args.hidden_size:
                x = torch.randn(rows, width, device="cuda", dtype=DTYPES[args.dtype])
                fns = {
                    "torch_reference": lambda: softmax(x),
                    "torch_native": lambda: torch.softmax(x, dim=-1),
                    "triton": lambda: softmax(x, backend="triton"),
                }
                expected = fns["torch_reference"]()
                for fn in fns.values():
                    torch.testing.assert_close(fn(), expected,
                                               rtol=TOLERANCES[args.dtype], atol=TOLERANCES[args.dtype])
                    for _ in range(args.warmup):
                        fn()
                torch.cuda.synchronize()
                before, thermal_before = gpu_state(), thermal_state()
                if thermal_before != "Not Active":
                    parser.error("thermal state is not normal; postpone measurement")
                names = list(fns)
                for repeat in range(1, args.runs + 1):
                    start = (repeat - 1) % len(names)
                    order = names[start:] + names[:start]
                    if repeat % 2 == 0:
                        order.reverse()
                    for backend in order:
                        us = triton.testing.do_bench_cudagraph(
                            fns[backend], rep=args.rep_ms, return_mode="median"
                        ) * 1000
                        results.append([timestamp, repeat, *before, *versions, rows, width,
                                        args.dtype, backend, "cuda_graph", us, args.warmup,
                                        args.runs, args.rep_ms, kernel_hash, reference_hash, revision, dirty])
                torch.cuda.synchronize()
                after, thermal_after = gpu_state(), thermal_state()
                if thermal_after != "Not Active":
                    parser.error("thermal slowdown interrupted measurement; no results written")
                for row in results[-len(names) * args.runs:]:
                    row.extend([thermal_before, thermal_after, *after])

    header = ["timestamp_utc", "repeat", *[f"{name}_before" for name in GPU_FIELDS],
              "torch", "cuda", "triton", "rows", "hidden_size", "dtype", "backend",
              "measurement", "median_us", "warmup", "runs", "rep_ms", "kernel_source_sha256",
              "reference_source_sha256", "source_revision", "dirty", "thermal_before", "thermal_after",
              *[f"{name}_after" for name in GPU_FIELDS]]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(results)
    print(output)


if __name__ == "__main__":
    main()
