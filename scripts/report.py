from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanoqwen.report import checkpoint_report, report_to_json, report_to_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a nanoqwen checkpoint report.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default=None, help="Optional UTF-8 text file for loss/ppl eval.")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    parser.add_argument("--out", default=None, help="Optional output file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = checkpoint_report(
        args.checkpoint,
        data=args.data,
        block_size=args.block_size,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        device=args.device,
    )
    rendered = report_to_json(report) if args.format == "json" else report_to_markdown(report)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
        print(f"wrote report to {path}")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()

