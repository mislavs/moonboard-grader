#!/usr/bin/env bash
set -euo pipefail
export PYTHONHASHSEED=1729
export CUBLAS_WORKSPACE_CONFIG=:4096:8
(
  cd generator
  uv.exe run pytest -q tests 2>&1 | tail -50
)
uv.exe run --project generator python .auto/benchmark.py 2>&1 | tee .auto/results.txt
