#!/usr/bin/env bash
set -euo pipefail

uv run python scripts/train.py \
  --out-dir out/tiny \
  --steps 500 \
  --batch-size 16 \
  --block-size 128 \
  --hidden-size 128 \
  --intermediate-size 384 \
  --layers 4 \
  --heads 4 \
  --kv-heads 2

