from dataclasses import dataclass
from typing import Literal

from gemm_lab.kernels.gemm_tiled import triton_gemm_tiled
from gemm_lab.kernels.gemm_optimized import triton_gemm_optimized


@dataclass(frozen=True)
class GemmConfig:
    kernel: Literal["baseline", "optimized"] = "baseline"


def gemm(a, b, *, cfg=None):
    """FP16 CUDA (M,K) @ (K,N) -> FP16 (M,N), with FP32 accumulation."""
    cfg = cfg or GemmConfig()
    if cfg.kernel == "baseline":
        return triton_gemm_tiled(a, b)
    if cfg.kernel == "optimized":
        return triton_gemm_optimized(a, b)
    raise ValueError(f"Unknown kernel: {cfg.kernel}")
