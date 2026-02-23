#!/usr/bin/env bash
set -euo pipefail

mkdir -p profiles

if ! command -v ncu >/dev/null 2>&1; then
  echo "ncu (Nsight Compute) not found in PATH"
  exit 1
fi

ncu \
  --set full \
  --target-processes all \
  --export profiles/gemm_profile \
  python -m triton_gemm_lab.benchmark --m 4096 --n 4096 --k 4096 --dtype fp16

cat <<'MSG'
Profile generated under profiles/.
TODO(student): report one tensor-core metric and one stall/latency metric.
MSG
