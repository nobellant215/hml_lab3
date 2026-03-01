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
    def gemm_point_kernel_kconstexpr(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        K: tl.constexpr,
    ):
        """
        TODO: implement a truly naive pointwise GEMM kernel.

        Suggested steps:
        1) Use one program per output element: `i = program_id(0)`, `j = program_id(1)`.
        2) Allocate a scalar FP32 accumulator with `tl.zeros((), dtype=tl.float32)`.
        3) Write a true scalar loop: `for k in range(0, K)`.
        4) Load `A[i, k]` and `B[k, j]`, cast to FP32, and accumulate `a * b`.
        5) Store a single output value `C[i, j]`.

        Important:
        - `K` is marked `tl.constexpr`, so this baseline only works when Triton
          can specialize the kernel for the concrete K value at launch time.
        - This is intentionally slow!
        """
        # Placeholder so skeleton branch is explicit.
        # Replace this with a full Triton kernel body.
        return


def triton_gemm_naive(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is not installed.")

    """
    TODO: launch the naive kernel and return output tensor C.
        - Allocate / define output matrix
        - Set grid size: Launch one program per output element using `grid = (M, N)`.
        - You may use _check_inputs function to get M, N, K dimensions
        - Call kernel implelemented above with appropriate parameters
    """
    
    raise NotImplementedError(
        "TODO: implement triton_gemm_naive in src/gemm_lab/kernels/gemm_naive.py"
    )
