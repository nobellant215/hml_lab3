from __future__ import annotations

import argparse
import time

import torch

from triton_gemm_lab.kernels import naive_gemm, tiled_gemm


def tflops(m: int, n: int, k: int, seconds: float) -> float:
    # GEMM does approximately 2*M*N*K floating-point ops.
    return (2.0 * m * n * k) / (seconds * 1e12)


def bench_once(fn, a: torch.Tensor, b: torch.Tensor, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        _ = fn(a, b)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = fn(a, b)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    a = torch.randn((args.m, args.k), device="cuda", dtype=dtype)
    b = torch.randn((args.k, args.n), device="cuda", dtype=dtype)

    sec_naive = bench_once(naive_gemm, a, b, args.warmup, args.iters)
    sec_tiled = bench_once(tiled_gemm, a, b, args.warmup, args.iters)
    sec_torch = bench_once(lambda x, y: torch.matmul(x, y), a, b, args.warmup, args.iters)

    print(f"shape={args.m}x{args.n}x{args.k} dtype={args.dtype}")
    print(f"naive:       {tflops(args.m, args.n, args.k, sec_naive):.2f} TFLOP/s ({sec_naive*1e3:.3f} ms)")
    print(f"tiled:       {tflops(args.m, args.n, args.k, sec_tiled):.2f} TFLOP/s ({sec_tiled*1e3:.3f} ms)")
    print(f"torch.matmul:{tflops(args.m, args.n, args.k, sec_torch):.2f} TFLOP/s ({sec_torch*1e3:.3f} ms)")


if __name__ == "__main__":
    main()
