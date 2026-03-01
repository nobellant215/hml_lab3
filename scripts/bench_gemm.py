from __future__ import annotations

import argparse

import torch

from gemm_lab.ops import GemmConfig, gemm
from gemm_lab.utils.bench import bench_once, tflops


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    a = torch.randn((args.m, args.k), device=args.device, dtype=dtype)
    b = torch.randn((args.k, args.n), device=args.device, dtype=dtype)

    bench_fns = {
        "naive": lambda x, y: gemm(x, y, cfg=GemmConfig(kernel="naive")),
        "tiled": lambda x, y: gemm(x, y, cfg=GemmConfig(kernel="tiled")),
        "torch.matmul": lambda x, y: torch.matmul(x, y),
    }

    print(f"shape={args.m}x{args.n}x{args.k} dtype={args.dtype} device={args.device}")
    for name, fn in bench_fns.items():
        sec = bench_once(fn, a, b, warmup=args.warmup, iters=args.iters)
        print(f"{name:12s} {tflops(args.m, args.n, args.k, sec):8.2f} TFLOP/s  ({sec*1e3:.3f} ms)")


if __name__ == "__main__":
    main()
