#!/usr/bin/env bash
set -euo pipefail

uv run python scripts/sft.py \
  --checkpoint out/smoke \
  --data examples/sft_tiny.jsonl \
  --out-dir out/sft-smoke \
  --steps 2 \
  --batch-size 2 \
  --block-size 64 \
  --device cpu

