import pytest
import torch
from gemm_lab.ops import GemmConfig, gemm

try:
    import triton
except ImportError:
    triton = None

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or triton is None, reason="CUDA+Triton required"
)


@pytest.mark.parametrize("kernel", ["baseline", "optimized"])
@pytest.mark.parametrize(
    "shape",
    [
        (128, 128, 128),
        (65, 64, 32),
        (64, 65, 32),
        (64, 64, 33),
        (65, 97, 33),
        (1, 129, 31),
        (127, 1, 257),
    ],
)
@pytest.mark.parametrize("transposed", [False, True])
def test_gemm(kernel, shape, transposed):
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    m, n, k = shape
    a = torch.randn(m, k, device="cuda", dtype=torch.float16) * 0.25
    b = (
        torch.randn(n, k, device="cuda", dtype=torch.float16).t()
        if transposed
        else torch.randn(k, n, device="cuda", dtype=torch.float16)
    )
    b = b * 0.25
    before_a, before_b = a.clone(), b.clone()
    out = gemm(a, b, cfg=GemmConfig(kernel))
    torch.testing.assert_close(
        out, (a.float() @ b.float()).half(), atol=0.02, rtol=0.02
    )
    assert out.dtype == a.dtype and out.device == a.device and out.shape == (m, n)
    assert out.data_ptr() not in (a.data_ptr(), b.data_ptr())
    assert torch.equal(a, before_a) and torch.equal(b, before_b)
