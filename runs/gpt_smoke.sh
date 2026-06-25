#!/usr/bin/env bash
set -euo pipefail

uv run python scripts/train.py \
  --model gpt \
  --out-dir out/gpt-smoke \
  --steps 4 \
  --batch-size 2 \
  --block-size 32 \
  --hidden-size 64 \
  --layers 2 \
  --heads 4 \
  --eval-every 2 \
  --save-every 4 \
  --device cpu
