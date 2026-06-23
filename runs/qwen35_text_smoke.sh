#!/usr/bin/env bash
set -euo pipefail

uv run python scripts/qwen35_llm_generate.py \
  --model models/Qwen/Qwen3.5-0.8B \
  --prompt "Say hello in one short sentence." \
  --max-new-tokens 8 \
  --device cpu
