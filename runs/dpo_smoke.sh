#!/usr/bin/env bash
set -euo pipefail

uv run python scripts/dpo.py \
  --checkpoint out/smoke \
  --data examples/preferences_tiny.jsonl \
  --out-dir out/dpo-smoke \
  --steps 2 \
  --batch-size 2 \
  --block-size 64 \
  --device cpu

