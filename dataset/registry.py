from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from dataset.base import ParquetTextDatasetSpec
from dataset.climbmix.spec import ALIASES as CLIMBMIX_ALIASES
from dataset.climbmix.spec import SPEC as CLIMBMIX_SPEC
from dataset.imdb.spec import ALIASES as IMDB_ALIASES
from dataset.imdb.spec import SPEC as IMDB_SPEC


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = (
    Path(os.environ["NANOQWEN_DATA_DIR"]).expanduser()
    if "NANOQWEN_DATA_DIR" in os.environ
    else PROJECT_ROOT / "data"
)
DOWNLOAD_CHUNK_BYTES = 1024 * 1024


_TEXT_DATASETS: dict[str, ParquetTextDatasetSpec] = {
    CLIMBMIX_SPEC.name: CLIMBMIX_SPEC,
    IMDB_SPEC.name: IMDB_SPEC,
}

_TEXT_DATASET_ALIASES = {
    **CLIMBMIX_ALIASES,
    **IMDB_ALIASES,
}


def available_text_datasets() -> tuple[str, ...]:
    return tuple(sorted(_TEXT_DATASETS))


def get_text_dataset_spec(name: str) -> ParquetTextDatasetSpec:
    key = _TEXT_DATASET_ALIASES.get(name, name)
    try:
        return _TEXT_DATASETS[key]
    except KeyError as exc:
        choices = ", ".join(available_text_datasets())
        raise ValueError(f"unknown text dataset {name!r}; available: {choices}") from exc


def resolve_num_shards(spec: ParquetTextDatasetSpec, num_shards: int | None) -> int:
    if not spec.is_numbered_shards:
        if num_shards is not None:
            raise ValueError(f"dataset {spec.name!r} does not use numbered shards")
        return 0

    if num_shards is None:
        num_shards = spec.default_num_shards
    if num_shards == -1:
        assert spec.max_train_shards is not None
        return spec.max_train_shards
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1, or -1 for all training shards")
    assert spec.max_train_shards is not None
    return min(num_shards, spec.max_train_shards)


def planned_shard_ids(name: str, num_shards: int | None = None) -> tuple[list[int], int]:
    spec = get_text_dataset_spec(name)
    if not spec.is_numbered_shards:
        raise ValueError(f"dataset {spec.name!r} does not use numbered shards")
    train_count = resolve_num_shards(spec, num_shards)
    assert spec.val_shard is not None
    return list(range(train_count)), spec.val_shard


def text_dataset_dir(name: str, data_dir: str | Path | None = None) -> Path:
    spec = get_text_dataset_spec(name)
    root = Path(data_dir).expanduser() if data_dir is not None else DEFAULT_DATA_DIR
    return root / spec.name


def shard_path(name: str, index: int, data_dir: str | Path | None = None) -> Path:
    spec = get_text_dataset_spec(name)
    if not spec.is_numbered_shards:
        raise ValueError(f"dataset {spec.name!r} does not use numbered shards")
    return text_dataset_dir(name, data_dir) / "shards" / spec.shard_filename(index)


def split_parquet_path(name: str, split: str, data_dir: str | Path | None = None) -> Path:
    spec = get_text_dataset_spec(name)
    if split not in spec.split_paths:
        raise ValueError(f"unknown split {split!r} for dataset {spec.name!r}")
    return text_dataset_dir(name, data_dir) / "raw" / spec.split_paths[split]


def prepared_text_path(
    name: str,
    split: str,
    num_shards: int | None = None,
    data_dir: str | Path | None = None,
) -> Path:
    spec = get_text_dataset_spec(name)
    if spec.is_numbered_shards and split == "train":
        train_count = resolve_num_shards(spec, num_shards)
        filename = f"{spec.name}-train-{train_count:05d}.txt"
    elif spec.is_numbered_shards and split == "val":
        filename = f"{spec.name}-val.txt"
    elif not spec.is_numbered_shards and split in spec.split_paths:
        if num_shards is not None:
            raise ValueError(f"dataset {spec.name!r} does not use numbered shards")
        filename = f"{spec.name}-{split}.txt"
    else:
        choices = ", ".join(spec.default_splits)
        raise ValueError(f"unknown split {split!r} for dataset {spec.name!r}; available: {choices}")
    return text_dataset_dir(name, data_dir) / "prepared" / filename


def required_shard_paths(
    name: str,
    num_shards: int | None = None,
    data_dir: str | Path | None = None,
    splits: tuple[str, ...] | None = None,
) -> list[Path]:
    spec = get_text_dataset_spec(name)
    if not spec.is_numbered_shards:
        split_names = splits if splits is not None else spec.default_splits
        return [split_parquet_path(name, split, data_dir) for split in split_names]

    train_ids, val_id = planned_shard_ids(name, num_shards)
    ids = [*train_ids, val_id]
    return [shard_path(name, index, data_dir) for index in dict.fromkeys(ids)]


def _download_file(url: str, destination: Path, max_attempts: int = 5) -> Path:
    if destination.exists() and destination.stat().st_size > 0:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            with urlopen(url, timeout=30) as response, tmp_path.open("wb") as file:
                while True:
                    chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    file.write(chunk)
            tmp_path.replace(destination)
            return destination
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
            tmp_path.unlink(missing_ok=True)
            if attempt < max_attempts:
                time.sleep(2**attempt)

    raise RuntimeError(f"failed to download {url} to {destination}: {last_error}")


def download_text_dataset(
    name: str,
    num_shards: int | None = None,
    data_dir: str | Path | None = None,
    workers: int = 4,
    splits: tuple[str, ...] | None = None,
) -> list[Path]:
    spec = get_text_dataset_spec(name)
    if spec.is_numbered_shards:
        train_ids, val_id = planned_shard_ids(name, num_shards)
        ids = list(dict.fromkeys([*train_ids, val_id]))
        jobs = [
            (spec.shard_url(index), shard_path(name, index, data_dir))
            for index in ids
            if not shard_path(name, index, data_dir).exists()
        ]
    else:
        split_names = splits if splits is not None else spec.default_splits
        jobs = [
            (spec.split_url(split), split_parquet_path(name, split, data_dir))
            for split in split_names
            if not split_parquet_path(name, split, data_dir).exists()
        ]

    if jobs:
        max_workers = max(1, min(workers, len(jobs)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_download_file, url, path) for url, path in jobs]
            for future in as_completed(futures):
                future.result()

    return required_shard_paths(name, num_shards, data_dir, splits=splits)


def ensure_text_dataset_shards(
    name: str,
    num_shards: int | None = None,
    data_dir: str | Path | None = None,
    download: bool = False,
    workers: int = 4,
    splits: tuple[str, ...] | None = None,
) -> list[Path]:
    if download:
        return download_text_dataset(
            name,
            num_shards=num_shards,
            data_dir=data_dir,
            workers=workers,
            splits=splits,
        )

    paths = required_shard_paths(name, num_shards, data_dir, splits=splits)
    missing = [path for path in paths if not path.exists()]
    if missing:
        first_missing = missing[0]
        raise FileNotFoundError(
            f"missing dataset shard {first_missing}. Run "
            f"`uv run python {get_text_dataset_spec(name).prepare_script}` "
            "or pass --download."
        )
    return paths


def _source_paths_for_split(
    name: str,
    split: str,
    num_shards: int | None,
    data_dir: str | Path | None,
) -> list[Path]:
    spec = get_text_dataset_spec(name)
    if not spec.is_numbered_shards:
        if split in spec.split_paths:
            return [split_parquet_path(name, split, data_dir)]
        choices = ", ".join(spec.default_splits)
        raise ValueError(f"unknown split {split!r} for dataset {spec.name!r}; available: {choices}")

    train_ids, val_id = planned_shard_ids(name, num_shards)
    if split == "train":
        return [shard_path(name, index, data_dir) for index in train_ids]
    if split == "val":
        return [shard_path(name, val_id, data_dir)]
    raise ValueError("split must be 'train' or 'val'")


def materialize_text_dataset(
    name: str,
    split: str = "train",
    num_shards: int | None = None,
    data_dir: str | Path | None = None,
    download: bool = False,
    workers: int = 4,
    force: bool = False,
) -> Path:
    output_path = prepared_text_path(name, split=split, num_shards=num_shards, data_dir=data_dir)
    if output_path.exists() and not force:
        return output_path

    ensure_text_dataset_shards(
        name,
        num_shards=num_shards,
        data_dir=data_dir,
        download=download,
        workers=workers,
        splits=(split,),
    )

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "reading parquet datasets requires pyarrow. Install it with: uv sync --extra data"
        ) from exc

    spec = get_text_dataset_spec(name)
    source_paths = _source_paths_for_split(name, split, num_shards, data_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    with tmp_path.open("w", encoding="utf-8") as output:
        for source_path in source_paths:
            parquet_file = pq.ParquetFile(source_path)
            for row_group_idx in range(parquet_file.num_row_groups):
                row_group = parquet_file.read_row_group(
                    row_group_idx,
                    columns=[spec.text_column],
                )
                for text in row_group.column(spec.text_column).to_pylist():
                    if text is None:
                        continue
                    output.write(str(text))
                    output.write(spec.document_separator)

    tmp_path.replace(output_path)
    return output_path


def prepare_text_dataset(
    name: str,
    splits: tuple[str, ...] | None = None,
    num_shards: int | None = None,
    data_dir: str | Path | None = None,
    workers: int = 4,
    force: bool = False,
) -> dict[str, Path]:
    spec = get_text_dataset_spec(name)
    if splits is None:
        splits = spec.default_splits

    paths = {}
    for split in splits:
        paths[split] = materialize_text_dataset(
            name,
            split=split,
            num_shards=num_shards,
            data_dir=data_dir,
            download=True,
            workers=workers,
            force=force,
        )
    return paths
