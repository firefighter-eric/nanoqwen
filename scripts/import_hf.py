from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanoqwen.checkpoint import import_hf_checkpoint, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import a compatible HF Qwen checkpoint.")
    parser.add_argument("model_name_or_path")
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = import_hf_checkpoint(args.model_name_or_path)
    save_checkpoint(model, args.out_dir)
    print(f"saved native checkpoint to {args.out_dir}")


if __name__ == "__main__":
    main()
