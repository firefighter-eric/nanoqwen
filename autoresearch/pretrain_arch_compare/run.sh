#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

uv run python -m autoresearch.pretrain_arch_compare run-suite \
  --config autoresearch/pretrain_arch_compare/experiments.json
