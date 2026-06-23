from __future__ import annotations

from dataset.base import ParquetTextDatasetSpec


SPEC = ParquetTextDatasetSpec(
    name="climbmix",
    description="karpathy/climbmix-400b-shuffle parquet text shards",
    base_url="https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main",
    max_train_shards=6542,
    val_shard=6542,
    prepare_script="dataset/climbmix/prepare.py",
)

ALIASES = {
    "climbmix-400b": SPEC.name,
    "karpathy/climbmix-400b-shuffle": SPEC.name,
}
