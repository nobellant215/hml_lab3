"""Compare equivalent workloads; compilation occurs before repeated timing."""

import argparse
import json
from pathlib import Path
import torch
from gemm_lab.ops import GemmConfig, gemm
from gemm_lab.kernels.gemm_tiled import triton_gemm_tiled
from gemm_lab.kernels.gemm_fused import fused_linear_relu
from gemm_lab.utils.bench import measure, environment, tflops


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=512)
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--k", type=int, default=512)
    p.add_argument("--workload", choices=["gemm", "linear"], default="gemm")
    p.add_argument(
        "--cases", nargs="+", choices=["torch", "baseline", "optimized", "fused"]
    )
    p.add_argument("--transpose-b", action="store_true")
    p.add_argument("--relu", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("results/gemm.jsonl"))
    args = p.parse_args()
    if not torch.cuda.is_available():
        p.error("A CUDA GPU with Triton is required")
    if min(args.m, args.n, args.k) < 1:
        p.error("Matrix dimensions must be positive")
    torch.manual_seed(args.seed)
    # FP32 oracle uses IEEE precision, independent of GEMM dispatch.
    torch.backends.cuda.matmul.allow_tf32 = False
    a = torch.randn(args.m, args.k, device="cuda", dtype=torch.float16) * 0.25
    b = (
        torch.randn(args.n, args.k, device="cuda", dtype=torch.float16).t()
        if args.transpose_b
        else torch.randn(args.k, args.n, device="cuda", dtype=torch.float16)
    ) * 0.25
    bias = torch.randn(args.n, device="cuda", dtype=torch.float16) * 0.25
    cases = args.cases or (
        ["torch", "baseline", "optimized"]
        if args.workload == "gemm"
        else ["torch", "baseline", "fused"]
    )
    if (args.workload == "gemm" and "fused" in cases) or (
        args.workload == "linear" and "optimized" in cases
    ):
        p.error("fused is a linear case; optimized is a GEMM case")

    def epilogue(y):
        y = y + bias.float()
        return (torch.relu(y) if args.relu else y).half()

    reference = a.float() @ b.float()
    reference = reference.half() if args.workload == "gemm" else epilogue(reference)
    if args.workload == "gemm":
        fns = {
            "torch": lambda: a @ b,
            "baseline": lambda: gemm(a, b),
            "optimized": lambda: gemm(a, b, cfg=GemmConfig("optimized")),
        }
    else:
        # FP32 intermediate gives the same epilogue rounding contract as fusion.
        # torch is a practical comparator and can round slightly differently.
        fns = {
            "torch": lambda: (
                torch.relu(torch.nn.functional.linear(a, b.t(), bias))
                if args.relu
                else torch.nn.functional.linear(a, b.t(), bias)
            ),
            "baseline": lambda: epilogue(
                triton_gemm_tiled(a, b, output_dtype=torch.float32)
            ),
            "fused": lambda: fused_linear_relu(a, b, bias, relu=args.relu),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    env = environment(a.device)
    for name in cases:
        row = {
            "implementation": name,
            "workload": args.workload,
            "shape": [args.m, args.n, args.k],
            "dtype": "float16",
            "seed": args.seed,
            "transpose_b": args.transpose_b,
            "relu": args.relu,
            **env,
        }
        with torch.inference_mode():
            got = fns[name]()
            torch.testing.assert_close(got, reference, atol=0.02, rtol=0.02)
            row.update(
                measure(
                    fns[name],
                    device=a.device,
                    warmup=args.warmup,
                    iters=args.iters,
                    repeats=args.repeats,
                )
            )
        row.update(
            status="ok",
            gemm_equivalent_tflops=tflops(
                args.m, args.n, args.k, row["median_ms"] / 1000
            ),
        )
        print(json.dumps(row))
        with args.out.open("a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
