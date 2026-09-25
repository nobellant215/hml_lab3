import pytest
import torch
from gemm_lab.validation import check_gemm_inputs
from gemm_lab.utils.bench import measure


def test_cpu_inputs_are_explicitly_rejected():
    with pytest.raises(ValueError, match="CUDA"):
        check_gemm_inputs(torch.ones(2, 3), torch.ones(3, 4))


def test_timer_repetitions():
    calls = []
    row = measure(lambda: calls.append(1), device="cpu", warmup=2, iters=3, repeats=2)
    assert len(calls) == 8
    assert len(row["samples_ms"]) == 2
    assert row["median_ms"] > 0
    with pytest.raises(ValueError):
        measure(lambda: None, device="cpu", iters=0)
