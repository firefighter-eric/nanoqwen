#!/usr/bin/env bash
set -euo pipefail

uv run python scripts/report.py \
  --checkpoint out/smoke \
  --max-batches 1 \
  --device cpu

