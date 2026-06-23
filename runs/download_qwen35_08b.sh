#!/usr/bin/env bash
set -euo pipefail

uv run hf download Qwen/Qwen3.5-0.8B \
  --local-dir models/Qwen/Qwen3.5-0.8B \
  --max-workers 8

