#!/usr/bin/env bash
set -euo pipefail

python -m triton_gemm_lab.validate \
  --dtype fp16 \
  --shapes "512x512x512,1024x1024x1024,2048x256x1024,256x2048x1024"
