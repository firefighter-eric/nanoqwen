from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ParquetTextDatasetSpec:
    name: str
    description: str
    base_url: str
    prepare_script: str
    max_train_shards: int | None = None
    val_shard: int | None = None
    default_num_shards: int = 10
    shard_template: str = "shard_{index:05d}.parquet"
    split_paths: dict[str, str] = field(default_factory=dict)
    default_splits: tuple[str, ...] = ("train", "val")
    text_column: str = "text"
    document_separator: str = "\n\n"

    @property
    def is_numbered_shards(self) -> bool:
        return self.max_train_shards is not None and self.val_shard is not None

    def shard_filename(self, index: int) -> str:
        return self.shard_template.format(index=index)

    def shard_url(self, index: int) -> str:
        return f"{self.base_url.rstrip('/')}/{self.shard_filename(index)}"

    def split_url(self, split: str) -> str:
        return f"{self.base_url.rstrip('/')}/{self.split_paths[split]}"
