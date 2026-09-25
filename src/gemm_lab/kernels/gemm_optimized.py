"""Student task B: tune and change program ordering without changing semantics."""

from gemm_lab.kernels.gemm_tiled import triton_gemm_tiled


def triton_gemm_optimized(a, b):
    # TODO B1: investigate a bounded set (at most 12) of launch configurations.
    # TODO B2: implement grouped program ordering in your own Triton kernel.
    # Keep gemm_tiled.py unchanged as the fixed experimental control.
    # This executable starting point is correct but performs no optimization.
    return triton_gemm_tiled(a, b)
