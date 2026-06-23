#!/usr/bin/env bash
set -euo pipefail

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

uv run python scripts/export_hf.py \
  --checkpoint out/smoke \
  --out-dir "$tmp_dir"

uv run python scripts/hf_smoke.py "$tmp_dir" --weights --device cpu --max-new-tokens 4

