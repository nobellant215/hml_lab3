from __future__ import annotations

import argparse

import torch

from triton_gemm_lab.kernels import naive_gemm, tiled_gemm


def parse_shapes(shape_text: str) -> list[tuple[int, int, int]]:
    shapes = []
    for item in shape_text.split(","):
        m, n, k = item.split("x")
        shapes.append((int(m), int(n), int(k)))
    return shapes


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def run(shape_list: list[tuple[int, int, int]], dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    for m, n, k in shape_list:
        a = torch.randn((m, k), device="cuda", dtype=dtype)
        b = torch.randn((k, n), device="cuda", dtype=dtype)

        ref = torch.matmul(a, b).float()
        out_naive = naive_gemm(a, b)
        out_tiled = tiled_gemm(a, b)

        err_naive = max_abs_diff(out_naive, ref)
        err_tiled = max_abs_diff(out_tiled, ref)
        print(f"shape={m}x{n}x{k} naive_max_abs={err_naive:.4e} tiled_max_abs={err_tiled:.4e}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shapes",
        type=str,
        default="512x512x512,1024x1024x1024,2048x256x1024,256x2048x1024",
        help="Comma-separated list of MxNxK tuples",
    )
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    args = parser.parse_args()

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    run(parse_shapes(args.shapes), dtype_map[args.dtype])


if __name__ == "__main__":
    main()
