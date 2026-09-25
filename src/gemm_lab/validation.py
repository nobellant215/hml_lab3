"""Provided input checks, shared by baseline and student entry points."""

import torch


def check_gemm_inputs(a: torch.Tensor, b: torch.Tensor) -> tuple[int, int, int]:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Expected rank-2 tensors.")
    if a.shape[1] != b.shape[0] or min(*a.shape, *b.shape) <= 0:
        raise ValueError("Expected compatible, positive matrix dimensions.")
    if not a.is_cuda or a.device != b.device:
        raise ValueError("Expected tensors on the same CUDA device.")
    if a.dtype != b.dtype or a.dtype != torch.float16:
        raise ValueError("This lab requires FP16 inputs.")
    if not a.is_contiguous() or not (b.is_contiguous() or b.t().is_contiguous()):
        raise ValueError(
            "A must be contiguous; B may also be a transposed contiguous view."
        )
    return a.shape[0], b.shape[1], a.shape[1]


def check_linear_inputs(x, weight_t, bias):
    m, n, k = check_gemm_inputs(x, weight_t)
    if bias is not None and (
        bias.shape != (n,)
        or bias.device != x.device
        or bias.dtype != x.dtype
        or not bias.is_contiguous()
    ):
        raise ValueError(
            "Bias must be a contiguous FP16 vector of length N on the input device."
        )
    return m, n, k
