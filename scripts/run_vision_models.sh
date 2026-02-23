#!/usr/bin/env bash
set -euo pipefail

python -m triton_gemm_lab.vision_integration \
  --models "resnet18,mobilenet_v3_small" \
  --batch 64 \
  --image-size 224 \
  --dtype fp16 \
  --compile
