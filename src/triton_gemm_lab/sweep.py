from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch

from triton_gemm_lab.benchmark import tflops
from triton_gemm_lab.kernels import pipeline_gemm


def bench_pipeline(a: torch.Tensor, b: torch.Tensor, num_stages: int, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        _ = pipeline_gemm(a, b, num_stages=num_stages)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = pipeline_gemm(a, b, num_stages=num_stages)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--stages", type=str, default="1,2,3,4,5")
    parser.add_argument("--out", type=str, default="results/num_stages_sweep.csv")
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    a = torch.randn((args.m, args.k), device="cuda", dtype=dtype)
    b = torch.randn((args.k, args.n), device="cuda", dtype=dtype)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for stage_text in args.stages.split(","):
        s = int(stage_text)
        sec = bench_pipeline(a, b, s, args.warmup, args.iters)
        perf = tflops(args.m, args.n, args.k, sec)
        rows.append((args.m, args.n, args.k, args.dtype, s, sec, perf))
        print(f"num_stages={s} -> {perf:.2f} TFLOP/s ({sec*1e3:.3f} ms)")

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["m", "n", "k", "dtype", "num_stages", "seconds", "tflops"])
        writer.writerows(rows)

    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
