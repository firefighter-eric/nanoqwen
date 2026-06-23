from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.registry import (
    available_text_datasets,
    get_text_dataset_spec,
    prepare_text_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare named text datasets for nanoqwen.")
    parser.add_argument(
        "--dataset",
        choices=available_text_datasets(),
        default="climbmix",
        help="Named dataset to prepare.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Number of training shards to prepare. Use -1 for all shards.",
    )
    parser.add_argument(
        "--split",
        action="append",
        choices=["train", "val"],
        dest="splits",
        help="Split to materialize. Can be passed more than once. Defaults to train and val.",
    )
    parser.add_argument("--data-dir", default=None, help="Override data directory.")
    parser.add_argument("--download-workers", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="Rebuild prepared text files.")
    parser.add_argument("--list", action="store_true", help="List available datasets and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list:
        for name in available_text_datasets():
            spec = get_text_dataset_spec(name)
            print(f"{name}: {spec.description}")
        return

    splits = tuple(args.splits) if args.splits else None
    spec = get_text_dataset_spec(args.dataset)
    print(f"dataset: {spec.name}")
    print(f"source: {spec.base_url}")

    paths = prepare_text_dataset(
        args.dataset,
        splits=splits,
        num_shards=args.num_shards,
        data_dir=args.data_dir,
        workers=args.download_workers,
        force=args.force,
    )
    for split, path in paths.items():
        print(f"{split}: {path}")


if __name__ == "__main__":
    main()
