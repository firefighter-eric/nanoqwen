#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f out/gpt-smoke/checkpoint.pt ]]; then
  bash runs/gpt_smoke.sh
fi

uv run python scripts/sft.py \
  --checkpoint out/gpt-smoke \
  --data examples/sft_tiny.jsonl \
  --out-dir out/gpt-sft-smoke \
  --steps 2 \
  --batch-size 2 \
  --block-size 32 \
  --device cpu
