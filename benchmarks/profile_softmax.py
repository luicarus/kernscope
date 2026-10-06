import argparse

import torch

from kernscope import softmax

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def main():
    parser = argparse.ArgumentParser(description="Launch Triton Softmax for Nsight Compute")
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--dtype", choices=DTYPES, default="bf16")
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    if args.rows < 1 or args.hidden_size < 1 or args.iterations < 1:
        parser.error("rows, hidden size, and iterations must be positive")

    torch.manual_seed(0)
    x = torch.randn(args.rows, args.hidden_size, device="cuda", dtype=DTYPES[args.dtype])
    with torch.inference_mode():
        for _ in range(args.iterations):
            softmax(x, backend="triton")
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
