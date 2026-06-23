#!/usr/bin/env bash
set -euo pipefail

uv run python scripts/chat_web.py \
  --checkpoint out/smoke \
  --host 127.0.0.1 \
  --port 8000 \
  --device cpu

