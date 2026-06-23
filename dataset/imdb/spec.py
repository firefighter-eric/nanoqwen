from __future__ import annotations

from dataset.base import ParquetTextDatasetSpec


SPEC = ParquetTextDatasetSpec(
    name="imdb",
    description="stanfordnlp/imdb movie review sentiment dataset",
    base_url="https://huggingface.co/datasets/stanfordnlp/imdb/resolve/main",
    prepare_script="dataset/imdb/prepare.py",
    split_paths={
        "train": "plain_text/train-00000-of-00001.parquet",
        "test": "plain_text/test-00000-of-00001.parquet",
        "unsupervised": "plain_text/unsupervised-00000-of-00001.parquet",
    },
    default_splits=("train", "test", "unsupervised"),
)

ALIASES = {
    "stanfordnlp/imdb": SPEC.name,
}
