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
        # Intentionally naive baseline: one program computes one output element.
        i = tl.program_id(0)
        j = tl.program_id(1)

        acc = tl.zeros((), dtype=tl.float32)

        # True scalar loop over K. This is only valid because K is constexpr.
        for k in range(0, K):
            a = tl.load(a_ptr + i * stride_am + k * stride_ak, mask=i < M, other=0.0).to(tl.float32)
            b = tl.load(b_ptr + k * stride_bk + j * stride_bn, mask=j < N, other=0.0).to(tl.float32)
            acc += a * b

        tl.store(c_ptr + i * stride_cm + j * stride_cn, acc, mask=(i < M) & (j < N))


def triton_gemm_naive(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is not installed.")

    M, N, K = _check_inputs(a, b)
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    # One kernel instance per output element.
    grid = (M, N)
    gemm_point_kernel_kconstexpr[grid](
        a,
        b,
        c,
        M,
        N,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        K=K,
        num_warps=1,
        num_stages=1,
    )
    return c
