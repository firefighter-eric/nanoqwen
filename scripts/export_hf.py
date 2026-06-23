from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanoqwen.checkpoint import export_hf_checkpoint, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a nanoqwen checkpoint in HF-style format.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tokenizer-source", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, _ = load_checkpoint(args.checkpoint)
    export_hf_checkpoint(model, args.out_dir, tokenizer_source=args.tokenizer_source)
    print(f"saved HF-style checkpoint to {args.out_dir}")


if __name__ == "__main__":
    main()
