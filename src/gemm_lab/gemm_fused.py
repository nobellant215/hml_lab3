from __future__ import annotations

import warnings

import torch
from torch import nn

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


def _fallback_addmm_relu(
    x: torch.Tensor,
    weight_t: torch.Tensor,
    bias: torch.Tensor | None,
    relu: bool,
) -> torch.Tensor:
    if bias is None:
        y = torch.matmul(x, weight_t)
    else:
        y = torch.addmm(bias, x, weight_t)
    return torch.relu(y) if relu else y


def _triton_eligible(x: torch.Tensor, weight_t: torch.Tensor, bias: torch.Tensor | None) -> tuple[bool, str]:
    if triton is None:
        return False, "Triton is not installed"
    if x.dim() != 2 or weight_t.dim() != 2:
        return False, "Expected rank-2 tensors for x and weight_t"
    if x.shape[1] != weight_t.shape[0]:
        return False, f"Incompatible shapes: {tuple(x.shape)} x {tuple(weight_t.shape)}"
    if not x.is_cuda or not weight_t.is_cuda:
        return False, "Expected CUDA tensors"
    if not x.is_contiguous() or not weight_t.is_contiguous():
        return False, "Expected contiguous x and weight_t"
    if x.dtype != weight_t.dtype:
        return False, "x and weight_t dtype must match"
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False, "Supported dtypes are fp16/bf16/fp32"
    if bias is not None:
        if bias.dim() != 1 or bias.shape[0] != weight_t.shape[1]:
            return False, "Bias must be shape [out_features]"
        if not bias.is_cuda or bias.dtype != x.dtype or not bias.is_contiguous():
            return False, "Bias must be contiguous CUDA tensor with same dtype"
    return True, ""


if triton is not None:

    @triton.jit
    def _fused_linear_bias_relu_kernel(
        x_ptr,
        w_ptr,
        bias_ptr,
        y_ptr,
        M,
        N,
        K,
        stride_xm,
        stride_xk,
        stride_wk,
        stride_wn,
        stride_ym,
        stride_yn,
        stride_bias,
        HAS_BIAS: tl.constexpr,
        DO_RELU: tl.constexpr,
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
            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + (k_start + offs_k)[None, :] * stride_xk
            w_ptrs = w_ptr + (k_start + offs_k)[:, None] * stride_wk + offs_n[None, :] * stride_wn

            x_mask = (offs_m[:, None] < M) & ((k_start + offs_k)[None, :] < K)
            w_mask = ((k_start + offs_k)[:, None] < K) & (offs_n[None, :] < N)

            x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
            w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
            acc += tl.dot(x_tile, w_tile)

        if HAS_BIAS:
            b_ptrs = bias_ptr + offs_n * stride_bias
            b = tl.load(b_ptrs, mask=offs_n < N, other=0.0).to(tl.float32)
            acc += b[None, :]

        if DO_RELU:
            acc = tl.maximum(acc, 0.0)

        y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
        y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(y_ptrs, acc, mask=y_mask)


def fused_linear_relu(
    x: torch.Tensor,
    weight_t: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    relu: bool = True,
    debug: bool = False,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 32,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    """Compute y = relu(x @ weight_t + bias) if relu=True else x @ weight_t + bias."""
    ok, reason = _triton_eligible(x, weight_t, bias)
    if not ok:
        warnings.warn(f"Using fallback fused path: {reason}", stacklevel=2)
        return _fallback_addmm_relu(x, weight_t, bias, relu)

    if debug:
        warnings.warn("Using Triton fused linear+bias+relu kernel", stacklevel=2)

    M, K = x.shape
    N = weight_t.shape[1]
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    _fused_linear_bias_relu_kernel[grid](
        x,
        weight_t,
        bias if bias is not None else y,
        y,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        weight_t.stride(0),
        weight_t.stride(1),
        y.stride(0),
        y.stride(1),
        0 if bias is None else bias.stride(0),
        HAS_BIAS=bias is not None,
        DO_RELU=relu,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return y


class FusedLinearReLU(nn.Module):
    """Linear layer with fused optional ReLU in one call path."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        relu: bool = True,
        device: str = "cuda",
        debug_fallback: bool = False,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device))
        self.bias = nn.Parameter(torch.empty(out_features, device=device)) if bias else None
        self.relu = relu
        self.debug_fallback = debug_fallback
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            bound = 1.0 / (self.weight.shape[1] ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_linear_relu(
            x,
            self.weight.t().contiguous(),
            self.bias,
            relu=self.relu,
            debug=self.debug_fallback,
        )
