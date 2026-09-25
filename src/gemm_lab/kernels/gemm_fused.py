"""Student task C: replace the TODO with a single Triton GEMM/epilogue kernel."""

import torch
from torch import nn
from gemm_lab.validation import check_linear_inputs


def fused_linear_relu(x, weight_t, bias=None, *, relu=True):
    """FP32 accumulate, bias, optional ReLU, then one final FP16 store.

    weight_t is (K,N), including a view of contiguous PyTorch (N,K) weights.
    No library fallback, intermediate GEMM output, or per-call weight copy.
    """
    check_linear_inputs(x, weight_t, bias)
    raise NotImplementedError("TODO C: implement GEMM with fused bias/optional ReLU")


class FusedLinearReLU(nn.Module):
    def __init__(
        self, in_features, out_features, *, bias=True, relu=True, device="cuda"
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device)
        )
        self.bias = (
            nn.Parameter(torch.empty(out_features, device=device)) if bias else None
        )
        self.relu = relu
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            bound = in_features**-0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return fused_linear_relu(x, self.weight.t(), self.bias, relu=self.relu)
