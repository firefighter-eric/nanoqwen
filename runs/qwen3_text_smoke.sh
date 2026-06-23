#!/usr/bin/env bash
set -euo pipefail

uv run python scripts/qwen3_llm_generate.py \
  --prompt "Say hello in one short sentence." \
  --max-new-tokens 8 \
  --device cpu

