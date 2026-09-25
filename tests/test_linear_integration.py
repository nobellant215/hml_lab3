import pytest
import torch
from gemm_lab.kernels.gemm_fused import FusedLinearReLU, fused_linear_relu
from gemm_lab.linear import MyLinear

try:
    import triton
except ImportError:
    triton = None

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or triton is None, reason="CUDA+Triton required"
)


def test_mylinear():
    torch.manual_seed(42)
    x = torch.randn(17, 33, device="cuda", dtype=torch.float16) * 0.25
    layer = MyLinear(33, 65).half()
    with torch.inference_mode():
        expected = torch.nn.functional.linear(x, layer.weight, layer.bias)
        torch.testing.assert_close(layer(x), expected, atol=0.01, rtol=0.02)


@pytest.mark.student
@pytest.mark.parametrize("bias_enabled", [False, True])
@pytest.mark.parametrize("relu", [False, True])
def test_fused(bias_enabled, relu):
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    x = torch.randn(17, 33, device="cuda", dtype=torch.float16) * 0.25
    weight = torch.randn(65, 33, device="cuda", dtype=torch.float16) * 0.25
    bias = (
        torch.randn(65, device="cuda", dtype=torch.float16) * 0.25
        if bias_enabled
        else None
    )
    ref = x.float() @ weight.t().float()
    if bias is not None:
        ref = ref + bias.float()
    if relu:
        ref = torch.relu(ref)
    with torch.inference_mode():
        out = fused_linear_relu(x, weight.t(), bias, relu=relu)
    torch.testing.assert_close(out, ref.half(), atol=0.02, rtol=0.02)
    assert out.dtype == x.dtype and out.device == x.device


@pytest.mark.student
def test_fused_module():
    x = torch.randn(3, 17, device="cuda", dtype=torch.float16) * 0.25
    layer = FusedLinearReLU(17, 19).half()
    with torch.inference_mode():
        ref = torch.relu(
            x.float() @ layer.weight.t().float() + layer.bias.float()
        ).half()
        torch.testing.assert_close(layer(x), ref, atol=0.02, rtol=0.02)
