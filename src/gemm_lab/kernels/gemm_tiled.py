from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


def _check_inputs(a: torch.Tensor, b: torch.Tensor) -> tuple[int, int, int]:
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError("Expected rank-2 tensors.")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Incompatible shapes: {tuple(a.shape)} x {tuple(b.shape)}")
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("Triton GEMM expects CUDA tensors.")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("Starter kernel expects contiguous inputs.")
    if a.dtype != b.dtype:
        raise ValueError("Input dtypes must match.")
    if a.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Supported dtypes: fp16, bf16, fp32")
    return a.shape[0], b.shape[1], a.shape[1]


if triton is not None:

    @triton.jit
    def _gemm_kernel_tiled(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """
        TODO(student): implement optimized tiled GEMM.

        Suggested optimizations:
        - Use larger BLOCK_M/BLOCK_N than naive.
        - Keep BLOCK_* as multiples of 16 for tensor-core-friendly shapes.
        - Tune num_warps and num_stages by shape bucket.
        - Ensure masked loads/stores are correct on boundary tiles.
        """
        # Placeholder so skeleton branch is explicit.
        return


def triton_gemm_tiled(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 32,
    num_warps: int = 4,
    num_stages: int = 1,
) -> torch.Tensor:
    """
    TODO(student): launch tiled kernel with configurable blocking.

    Deliverable intent:
    - This should outperform naive kernel on medium/large shapes.
    - Keep interface stable so scripts/tests can call it directly.
    """
    if triton is None:
        raise RuntimeError("Triton is not installed.")

    _check_inputs(a, b)
    raise NotImplementedError(
        "TODO(student): implement triton_gemm_tiled in src/gemm_lab/kernels/gemm_tiled.py"
    )
