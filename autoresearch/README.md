# Autoresearch

This directory contains experiment-specific code that should not live under the
generic `nanoqwen/` package.

Each dataset or task should own its experiment code, configs, and README in a
dedicated subdirectory. Keep the top-level README as an index so experiment
notes do not get mixed together.

## Experiments

| Dataset / task | Directory | Notes |
| --- | --- | --- |
| IMDb SFT | [`imdb_sft/`](imdb_sft/) | Qwen3-0.6B sentiment SFT experiments on IMDb. |
