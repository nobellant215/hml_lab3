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
    def _gemm_kernel_naive(
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
        TODO(student): implement a baseline tiled GEMM kernel.

        Suggested steps:
        1) Map `program_id(0/1)` to tile coordinates over M and N.
        2) Build `offs_m`, `offs_n`, `offs_k` with `tl.arange`.
        3) Loop over K by `BLOCK_K` and load tiles with masks.
        4) Accumulate with `tl.dot` into FP32 accumulator.
        5) Store C tile with boundary mask.
        """
        # Placeholder so skeleton branch is explicit.
        # Replace this with a full Triton kernel body.
        return


def triton_gemm_naive(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    TODO(student): launch the naive kernel and return output tensor C.

    Requirements:
    - Keep FP32 accumulation in kernel.
    - Use conservative blocks first (e.g., 16x16x16).
    - Validate against torch.matmul.
    """
    if triton is None:
        raise RuntimeError("Triton is not installed.")

    _check_inputs(a, b)
    raise NotImplementedError(
        "TODO(student): implement triton_gemm_naive in src/gemm_lab/kernels/gemm_naive.py"
    )
