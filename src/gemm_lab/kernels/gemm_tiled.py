from __future__ import annotations

import torch

from gemm_lab.validation import check_gemm_inputs

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


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
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            a_ptrs = (
                a_ptr
                + offs_m[:, None] * stride_am
                + (k_start + offs_k)[None, :] * stride_ak
            )
            b_ptrs = (
                b_ptr
                + (k_start + offs_k)[:, None] * stride_bk
                + offs_n[None, :] * stride_bn
            )

            a_mask = (offs_m[:, None] < M) & ((k_start + offs_k)[None, :] < K)
            b_mask = ((k_start + offs_k)[:, None] < K) & (offs_n[None, :] < N)

            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            acc += tl.dot(a, b)

        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc, mask=c_mask)


def triton_gemm_tiled(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 32,
    num_warps: int = 4,
    num_stages: int = 1,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is not installed.")

    M, N, K = check_gemm_inputs(a, b)
    if output_dtype not in (None, a.dtype, torch.float32):
        raise ValueError("Output dtype must be the input dtype or FP32.")
    c = torch.empty((M, N), device=a.device, dtype=output_dtype or a.dtype)

    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    _gemm_kernel_tiled[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return c
