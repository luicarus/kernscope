import argparse

import torch

from kernscope import fused_add_rms_norm

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def main():
    parser = argparse.ArgumentParser(description="Launch one fused Triton RMSNorm kernel for Nsight Compute")
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--dtype", choices=DTYPES, default="fp32")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    if args.rows < 1 or args.hidden_size < 1:
        parser.error("rows and hidden size must be positive")

    dtype = DTYPES[args.dtype]
    x = torch.randn(args.rows, args.hidden_size, device="cuda", dtype=dtype)
    residual = torch.randn_like(x)
    weight = torch.randn(args.hidden_size, device="cuda", dtype=dtype)
    fused_add_rms_norm(x, residual, weight, backend="triton")
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
