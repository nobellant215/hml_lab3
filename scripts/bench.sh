#!/usr/bin/env bash
set -euo pipefail

python -m triton_gemm_lab.benchmark \
  --m 4096 --n 4096 --k 4096 \
  --dtype fp16 \
  --warmup 20 --iters 100
