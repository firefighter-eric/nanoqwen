#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

uv run python autoresearch/pretrain_arch_compare/prepare_tokenizer.py "$@"
