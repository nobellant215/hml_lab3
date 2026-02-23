from __future__ import annotations

import argparse
import copy
import time
from typing import Callable

import torch
from torch import nn

from triton_gemm_lab.mylinear import MyLinear


def replace_linear_layers(module: nn.Module, device: str) -> nn.Module:
    """Recursively replace nn.Linear with MyLinear, copying parameters."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            new_layer = MyLinear(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=child.bias is not None,
                device=device,
            )
            with torch.no_grad():
                new_layer.weight.copy_(child.weight.to(device=device))
                if child.bias is not None and new_layer.bias is not None:
                    new_layer.bias.copy_(child.bias.to(device=device))
            setattr(module, name, new_layer)
        else:
            replace_linear_layers(child, device)
    return module


def _build_model(name: str) -> nn.Module:
    try:
        import torchvision.models as models
    except ImportError as exc:
        raise ImportError("torchvision is required for vision integration") from exc

    registry: dict[str, Callable[[], nn.Module]] = {
        "resnet18": lambda: models.resnet18(weights=None),
        "mobilenet_v3_small": lambda: models.mobilenet_v3_small(weights=None),
    }
    if name not in registry:
        raise ValueError(f"unsupported model '{name}'. Choose from {sorted(registry)}")
    return registry[name]()


def _bench(model: nn.Module, x: torch.Tensor, warmup: int, iters: int) -> float:
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(x)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(iters):
            _ = model(x)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
    return (t1 - t0) / iters


def run_model(name: str, batch: int, image_size: int, dtype: torch.dtype, use_compile: bool) -> None:
    base = _build_model(name).cuda().to(dtype=dtype)
    triton_model = replace_linear_layers(copy.deepcopy(base), device="cuda").to(dtype=dtype)

    x = torch.randn((batch, 3, image_size, image_size), device="cuda", dtype=dtype)

    base_eager = _bench(base, x, warmup=10, iters=40)
    triton_eager = _bench(triton_model, x, warmup=10, iters=40)
    print(f"[{name}] eager torch(ms)={base_eager*1e3:.3f} triton-linear(ms)={triton_eager*1e3:.3f}")

    if use_compile:
        base_compiled = torch.compile(base)
        triton_compiled = torch.compile(triton_model)
        base_comp = _bench(base_compiled, x, warmup=10, iters=40)
        triton_comp = _bench(triton_compiled, x, warmup=10, iters=40)
        print(f"[{name}] compile torch(ms)={base_comp*1e3:.3f} triton-linear(ms)={triton_comp*1e3:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default="resnet18,mobilenet_v3_small")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Triton GEMM lab")

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        run_model(name=name, batch=args.batch, image_size=args.image_size, dtype=dtype, use_compile=args.compile)


if __name__ == "__main__":
    main()
