from __future__ import annotations

import torch
from torch import nn

from triton_gemm_lab.kernels import triton_linear


class MyLinear(nn.Module):
    """
    Linear layer backed by Triton GEMM.

    Note: stores weight as [out_features, in_features], then transposes on forward.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: str = "cuda"):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device))
        self.bias = nn.Parameter(torch.empty(out_features, device=device)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            bound = 1 / (self.weight.shape[1] ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, in_features], weight.T: [in_features, out_features]
        return triton_linear(x, self.weight.t().contiguous(), self.bias)
