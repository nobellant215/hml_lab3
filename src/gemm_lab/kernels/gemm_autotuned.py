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

    # TODO(student): replace placeholder config list with real search space.
    # Start with 3-6 configs varying BLOCK_M/N/K, num_warps, and num_stages.
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        ],
        key=["M", "N", "K"],
    )
    @triton.jit
    def _gemm_kernel_autotuned(
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
        TODO(student): implement autotuned GEMM kernel body.

        Notes:
        - Reuse tiled-kernel math structure.
        - Keep FP32 accumulation.
        - Ensure output correctness first, then tune configs.
        """
        # Placeholder so skeleton branch is explicit.
        return


def triton_gemm_autotuned(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    TODO(student): launch autotuned kernel using meta-dependent grid.

    Expected shape of launch code:
    - Allocate C (float32 accumulator output).
    - Define grid as lambda meta: (cdiv(M, meta["BLOCK_M"]), cdiv(N, meta["BLOCK_N"]))
    - Launch kernel with strides and M/N/K.
    """
    if triton is None:
        raise RuntimeError("Triton is not installed.")

    _check_inputs(a, b)
    raise NotImplementedError(
        "TODO(student): implement triton_gemm_autotuned in src/gemm_lab/kernels/gemm_autotuned.py"
    )
