# Masters-Level ML Systems Lab 3

## Triton GEMM: Tensor Cores, Double Buffering, and PyTorch Integration

This repository is a starter scaffold for implementing and analyzing Triton GEMM kernels.

## Repository Layout

- `src/triton_gemm_lab/kernels.py`: Triton GEMM kernels (naive + tiled + pipeline).
- `src/triton_gemm_lab/validate.py`: correctness checks against `torch.matmul`.
- `src/triton_gemm_lab/benchmark.py`: TFLOP/s benchmark harness.
- `src/triton_gemm_lab/sweep.py`: `num_stages` sweep and CSV output.
- `src/triton_gemm_lab/mylinear.py`: `MyLinear` module backed by Triton GEMM.
- `src/triton_gemm_lab/model_integration.py`: Tiny MLP eager vs `torch.compile`.
- `src/triton_gemm_lab/vision_integration.py`: open-source vision model integration.
- `scripts/`: shell wrappers for common assignment workflows.
- `results/`: output CSV/tables.
- `profiles/`: Nsight Compute profile output.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Quick Start

```bash
# Correctness against torch.matmul for multiple shapes
bash scripts/run_correctness.sh

# Baseline and tiled benchmark
bash scripts/bench.sh

# Sweep num_stages in {1,2,3,4,5}
bash scripts/sweep_num_stages.sh

# Optional Nsight Compute profile (requires ncu + NVIDIA GPU)
bash scripts/profile_ncu.sh

# Tiny MLP integration path (eager + torch.compile)
python -m triton_gemm_lab.model_integration --device cuda --dtype fp16

# Open-source vision models with Triton linear replacement
bash scripts/run_vision_models.sh
```

## Assignment Mapping

- Part A (Baseline + Tiled GEMM): `naive_gemm` and `tiled_gemm` in `kernels.py`.
- Tensor Core Investigation: `tl.dot`, tile multiples of 16, compare FP16/BF16 vs FP32.
- Double Buffering: vary `num_stages` via `pipeline_gemm` and run `sweep.py`.
- Model Integration: `MyLinear` for tiny MLP and torchvision models.

## Deliverables Checklist

- [ ] Working naive and optimized Triton GEMM.
- [ ] Tuned configuration with documented best parameters.
- [ ] Performance table (naive vs optimized vs torch.matmul).
- [ ] Tensor-core evidence from Nsight Compute.
- [ ] `num_stages` sweep results + analysis.
- [ ] 2-4 page report.
