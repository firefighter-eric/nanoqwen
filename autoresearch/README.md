# Autoresearch

Autoresearch suites collect reproducible experiment grids and result summaries.

Available suites:

- `imdb_sft/`: Qwen3-0.6B IMDb SFT experiments.
- `pretrain_arch_compare/`: GPT vs Qwen-like pretraining architecture
  comparison with a Qwen3-0.6B-proportional downscaled Qwen config.

This directory contains experiment-specific code that should not live under the
generic `nanoqwen/` package.

Each dataset or task should own its experiment code, configs, and README in a
dedicated subdirectory. Keep the top-level README as an index so experiment
notes do not get mixed together.

## Experiments

| Dataset / task | Directory | Notes |
| --- | --- | --- |
| IMDb SFT | [`imdb_sft/`](imdb_sft/) | Qwen3-0.6B sentiment SFT experiments on IMDb. |
