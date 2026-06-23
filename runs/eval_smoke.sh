#!/usr/bin/env bash
set -euo pipefail

uv run python scripts/eval.py \
  --checkpoint out/smoke \
  --multiple-choice examples/multiple_choice_tiny.jsonl \
  --choice-prefix "" \
  --max-batches 2 \
  --device cpu
