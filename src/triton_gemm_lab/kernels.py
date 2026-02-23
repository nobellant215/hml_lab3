from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError as exc:  # pragma: no cover
    raise ImportError("Triton is required. Install with `pip install triton`.") from exc


@triton.jit
def _gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    m,
    n,
    k,
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

    for k_start in range(0, k, BLOCK_K):
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + (k_start + offs_k)[None, :] * stride_ak
        b_ptrs = b_ptr + (k_start + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn

        a_mask = (offs_m[:, None] < m) & ((k_start + offs_k)[None, :] < k)
        b_mask = ((k_start + offs_k)[:, None] < k) & (offs_n[None, :] < n)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < m) & (offs_n[None, :] < n)
    tl.store(c_ptrs, acc, mask=c_mask)


def _check_inputs(a: torch.Tensor, b: torch.Tensor) -> tuple[int, int, int]:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must be rank-2 tensors")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"incompatible matmul shapes: {a.shape} x {b.shape}")
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("a and b must be CUDA tensors")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("a and b must be contiguous")
    if a.dtype != b.dtype:
        raise ValueError("a and b must have the same dtype")
    if a.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("supported dtypes: fp16, bf16, fp32")
    return a.shape[0], b.shape[1], a.shape[1]


def _launch(
    a: torch.Tensor,
    b: torch.Tensor,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    num_stages: int,
) -> torch.Tensor:
    m, n, k = _check_inputs(a, b)
    c = torch.empty((m, n), device=a.device, dtype=torch.float32)

    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _gemm_kernel[grid](
        a,
        b,
        c,
        m,
        n,
        k,
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


def naive_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _launch(a, b, block_m=16, block_n=16, block_k=16, num_warps=2, num_stages=1)


def tiled_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 32,
    num_warps: int = 4,
) -> torch.Tensor:
    return _launch(a, b, block_m=block_m, block_n=block_n, block_k=block_k, num_warps=num_warps, num_stages=1)


def pipeline_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 32,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    return _launch(
        a,
        b,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def triton_linear(x: torch.Tensor, weight_t: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """
    y = x @ weight_t + bias where weight_t is shaped [in_features, out_features].
    """
    y = tiled_gemm(x, weight_t).to(x.dtype)
    if bias is not None:
        y = y + bias
    return y
