#!/usr/bin/env bash
set -euo pipefail

uv run python scripts/qwen35_llm_generate.py \
  --model models/Qwen/Qwen3.5-0.8B \
  --prompt "What can you help me with?" \
  --dry-run \
  --device cpu
