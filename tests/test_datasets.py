from __future__ import annotations

import pytest

from dataset.registry import (
    DEFAULT_DATA_DIR,
    available_text_datasets,
    get_text_dataset_spec,
    materialize_text_dataset,
    planned_shard_ids,
    prepared_text_path,
    split_parquet_path,
)


def test_climbmix_is_registered() -> None:
    spec = get_text_dataset_spec("climbmix")

    assert "climbmix" in available_text_datasets()
    assert spec.base_url.endswith("/karpathy/climbmix-400b-shuffle/resolve/main")
    assert spec.max_train_shards == 6542
    assert spec.val_shard == 6542


def test_climbmix_alias_resolves() -> None:
    spec = get_text_dataset_spec("karpathy/climbmix-400b-shuffle")

    assert spec.name == "climbmix"


def test_imdb_is_registered() -> None:
    spec = get_text_dataset_spec("stanfordnlp/imdb")

    assert "imdb" in available_text_datasets()
    assert spec.name == "imdb"
    assert spec.split_paths["train"] == "plain_text/train-00000-of-00001.parquet"
    assert spec.default_splits == ("train", "test", "unsupervised")


def test_planned_shards_pin_validation_shard() -> None:
    train_ids, val_id = planned_shard_ids("climbmix", num_shards=3)

    assert train_ids == [0, 1, 2]
    assert val_id == 6542


def test_prepared_text_path_includes_train_shard_count(tmp_path) -> None:
    train_path = prepared_text_path("climbmix", "train", num_shards=3, data_dir=tmp_path)
    val_path = prepared_text_path("climbmix", "val", num_shards=3, data_dir=tmp_path)

    assert train_path == tmp_path / "climbmix" / "prepared" / "climbmix-train-00003.txt"
    assert val_path == tmp_path / "climbmix" / "prepared" / "climbmix-val.txt"


def test_imdb_paths_use_split_names(tmp_path) -> None:
    raw_path = split_parquet_path("imdb", "train", data_dir=tmp_path)
    prepared_path = prepared_text_path("imdb", "train", data_dir=tmp_path)

    assert raw_path == tmp_path / "imdb" / "raw" / "plain_text" / "train-00000-of-00001.parquet"
    assert prepared_path == tmp_path / "imdb" / "prepared" / "imdb-train.txt"


def test_default_dataset_dir_is_repo_data_dir() -> None:
    assert DEFAULT_DATA_DIR.name == "data"
    assert prepared_text_path("climbmix", "train").is_relative_to(DEFAULT_DATA_DIR)


def test_materialize_missing_dataset_shows_prepare_hint(tmp_path) -> None:
    with pytest.raises(FileNotFoundError) as excinfo:
        materialize_text_dataset("climbmix", num_shards=1, data_dir=tmp_path)

    message = str(excinfo.value)
    assert "shard_00000.parquet" in message
    assert "dataset/climbmix/prepare.py" in message


def test_materialize_missing_imdb_shows_prepare_hint(tmp_path) -> None:
    with pytest.raises(FileNotFoundError) as excinfo:
        materialize_text_dataset("imdb", split="train", data_dir=tmp_path)

    message = str(excinfo.value)
    assert "train-00000-of-00001.parquet" in message
    assert "dataset/imdb/prepare.py" in message
