#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

uv run python -m autoresearch.imdb_sft run-suite \
  --config autoresearch/imdb_sft/experiments_full_e2.json
