#!/usr/bin/env bash
set -euo pipefail

python -m triton_gemm_lab.sweep \
  --m 4096 --n 4096 --k 4096 \
  --dtype fp16 \
  --stages "1,2,3,4,5" \
  --warmup 20 --iters 100 \
  --out results/num_stages_sweep.csv
