#!/usr/bin/env bash
set -euo pipefail

uv run python scripts/qwen3_llm_generate.py \
  --prompt "What can you help me with?" \
  --dry-run \
  --device cpu

