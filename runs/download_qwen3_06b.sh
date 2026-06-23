#!/usr/bin/env bash
set -euo pipefail

model_dir=models/Qwen/Qwen3-0.6B
weights_url='https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/model.safetensors?download=true'
expected_bytes=1503300328

mkdir -p "$model_dir"

HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" uv run hf download Qwen/Qwen3-0.6B \
  --local-dir "$model_dir" \
  --exclude model.safetensors \
  --max-workers 4

current_bytes=0
if [[ -f "$model_dir/model.safetensors" ]]; then
  current_bytes=$(stat -c%s "$model_dir/model.safetensors")
fi

if [[ "$current_bytes" -eq "$expected_bytes" ]]; then
  echo "$model_dir/model.safetensors already complete"
  exit 0
fi

curl -L --fail --retry 5 --retry-delay 2 -C - \
  -o "$model_dir/model.safetensors" \
  "$weights_url"

current_bytes=$(stat -c%s "$model_dir/model.safetensors")
if [[ "$current_bytes" -ne "$expected_bytes" ]]; then
  echo "unexpected model.safetensors size: $current_bytes, expected $expected_bytes" >&2
  exit 1
fi
