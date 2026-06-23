#!/usr/bin/env bash
set -euo pipefail

uv run python scripts/train.py \
  --out-dir out/smoke \
  --steps 20 \
  --batch-size 4 \
  --block-size 64 \
  --hidden-size 64 \
  --intermediate-size 192 \
  --layers 2 \
  --heads 4 \
  --kv-heads 2 \
  --eval-every 10 \
  --save-every 20 \
  --device cpu

