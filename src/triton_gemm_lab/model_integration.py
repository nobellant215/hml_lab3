from __future__ import annotations

import argparse
import time

import torch
from torch import nn

from triton_gemm_lab.mylinear import MyLinear


class TinyMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, device: str = "cuda"):
        super().__init__()
        self.fc1 = MyLinear(in_dim, hidden_dim, device=device)
        self.act = nn.GELU()
        self.fc2 = MyLinear(hidden_dim, out_dim, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def run_mode(model: nn.Module, x: torch.Tensor, iters: int = 100) -> float:
    for _ in range(20):
        _ = model(x)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model(x)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--in-dim", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--out-dim", type=int, default=1024)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    model = TinyMLP(args.in_dim, args.hidden_dim, args.out_dim, device=args.device).to(dtype=dtype)
    x = torch.randn((args.batch, args.in_dim), device=args.device, dtype=dtype)

    eager_time = run_mode(model, x)
    print(f"eager:         {eager_time * 1e3:.3f} ms")

    compiled = torch.compile(model)
    comp_time = run_mode(compiled, x)
    print(f"torch.compile: {comp_time * 1e3:.3f} ms")

    print("\nTODO(student): inspect profiler traces to explain fusion boundaries around custom Triton calls.")


if __name__ == "__main__":
    main()
