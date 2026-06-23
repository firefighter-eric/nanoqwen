from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataset.imdb.spec import SPEC
from dataset.registry import prepare_text_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the IMDb movie review text dataset.")
    parser.add_argument(
        "--split",
        action="append",
        choices=sorted(SPEC.split_paths),
        dest="splits",
        help="Split to materialize. Can be passed more than once. Defaults to all splits.",
    )
    parser.add_argument("--data-dir", default=None, help="Override data directory.")
    parser.add_argument("--download-workers", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="Rebuild prepared text files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = tuple(args.splits) if args.splits else SPEC.default_splits
    print(f"dataset: {SPEC.name}")
    print(f"source: {SPEC.base_url}")

    paths = prepare_text_dataset(
        SPEC.name,
        splits=splits,
        data_dir=args.data_dir,
        workers=args.download_workers,
        force=args.force,
    )
    for split, path in paths.items():
        print(f"{split}: {path}")


if __name__ == "__main__":
    main()
