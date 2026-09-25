"""Fixed-shape synthetic inference; no download or partial final batch."""

import argparse
import torch
from torch import nn
from gemm_lab.linear import MyLinear
from gemm_lab.kernels.gemm_fused import FusedLinearReLU
from gemm_lab.utils.bench import measure


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--fused", action="store_true", help="requires completed task C")
    args = p.parse_args()
    if not torch.cuda.is_available():
        p.error("CUDA required")
    if args.batch_size < 1:
        p.error("batch size must be positive")
    torch.manual_seed(42)
    dims = [784, 256, 128, 10]
    reference = (
        nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
        .cuda()
        .half()
        .eval()
    )
    layers = []
    for i, (k, n) in enumerate(zip(dims, dims[1:])):
        if args.fused:
            layers.append(FusedLinearReLU(k, n, relu=i < 2).half())
        else:
            layers.append(MyLinear(k, n).half())
            if i < 2:
                layers.append(nn.ReLU())
    model = nn.Sequential(*layers).eval()
    with torch.no_grad():
        for src, dst in zip(
            [m for m in reference if isinstance(m, nn.Linear)],
            [m for m in model if hasattr(m, "weight")],
        ):
            dst.weight.copy_(src.weight)
            dst.bias.copy_(src.bias)
    x = torch.randn(args.batch_size, 784, device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        torch.testing.assert_close(model(x), reference(x), atol=0.01, rtol=0.02)
    for name, m in [("torch", reference), ("custom", model)]:
        print(name, measure(lambda: m(x), device=x.device))


if __name__ == "__main__":
    main()
