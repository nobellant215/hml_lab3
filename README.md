# Triton GEMM Lab

This repo is a lab skeleton for Triton GEMM optimization with a fused linear path.

## Repo Tree

```text
triton-gemm-lab/
  README.md
  pyproject.toml
  requirements.txt
  .gitignore
  Makefile

  src/
    gemm_lab/
      __init__.py
      ops.py
      linear.py
      gemm_fused.py
      kernels/
        __init__.py
        gemm_naive.py
        gemm_tiled.py
        gemm_autotuned.py
      utils/
        __init__.py
        correctness.py
        bench.py

  tests/
    test_gemm_correctness.py
    test_linear_integration.py

  scripts/
    bench_gemm.py
    run_mlp_demo.py
    profile_gemm.py
```

## Student Tasks

Implement and improve:
- `src/gemm_lab/kernels/gemm_naive.py`
- `src/gemm_lab/kernels/gemm_tiled.py`
- `src/gemm_lab/kernels/gemm_autotuned.py`

Keep correctness tests passing and report speedups plus profiling screenshots/analysis.

## Fused Path

`src/gemm_lab/gemm_fused.py` provides:
- `fused_linear_relu(x, weight_t, bias=None, relu=True)`
- `FusedLinearReLU(nn.Module)`

Fallback behavior:
- If Triton fused constraints are not met, it warns and falls back to:
  - `torch.addmm(...)`
  - optional `torch.relu(...)`

## MNIST Demo

`run_mlp_demo.py` runs inference only on a 3-layer MLP (`784 -> 256 -> 128 -> 10`):
- hidden layers: fused (`linear + bias + relu`)
- output layer: non-fused linear logits

Data behavior:
- Attempts MNIST auto-download.
- Falls back to synthetic tensors with warning if MNIST is unavailable.

## Commands

```bash
pip install -r requirements.txt
pip install -e .

make test
make bench
make demo
make profile
```
