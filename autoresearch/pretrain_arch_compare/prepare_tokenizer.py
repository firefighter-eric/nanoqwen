from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataset.registry import materialize_text_dataset


SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS_TOKEN = SPECIAL_TOKENS[0]
SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| """
    r"""?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a fixed BPE tokenizer for pretraining compares.")
    parser.add_argument("--dataset", default="climbmix")
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-shards", type=int, default=10)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--download-workers", type=int, default=4)
    parser.add_argument("--text-file", default=None, help="Use an existing text file instead of a dataset split.")
    parser.add_argument("--output", default="data/climbmix/tokenizer-bpe-8192")
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument(
        "--max-chars",
        type=int,
        default=50_000_000,
        help="Train on the first N chars. Use -1 for the whole text file.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def write_sample(source: Path, output_dir: Path, max_chars: int, force: bool) -> Path:
    if max_chars < 0:
        return source
    sample_path = output_dir / f"tokenizer-train-sample-{max_chars}.txt"
    if sample_path.exists() and not force:
        return sample_path
    output_dir.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8") as src, sample_path.open("w", encoding="utf-8") as dst:
        dst.write(src.read(max_chars))
    return sample_path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    tokenizer_json = output_dir / "tokenizer.json"
    token_bytes_path = output_dir / "token_bytes.pt"
    if tokenizer_json.exists() and token_bytes_path.exists() and not args.force:
        print(f"tokenizer already exists: {output_dir}")
        return

    if args.text_file:
        source = Path(args.text_file)
    else:
        source = materialize_text_dataset(
            args.dataset,
            split=args.split,
            num_shards=args.num_shards,
            data_dir=args.data_dir,
            download=args.download,
            workers=args.download_workers,
        )

    try:
        from tokenizers import ByteLevelBPETokenizer
        from transformers import PreTrainedTokenizerFast
    except ImportError as exc:
        raise ImportError("Preparing the BPE tokenizer requires transformers[tokenizers].") from exc

    train_file = write_sample(source, output_dir, args.max_chars, args.force)
    tokenizer = ByteLevelBPETokenizer(add_prefix_space=False)
    tokenizer.train(
        files=[str(train_file)],
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=SPECIAL_TOKENS,
    )

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer._tokenizer,
        bos_token=BOS_TOKEN,
        additional_special_tokens=SPECIAL_TOKENS[1:],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(output_dir)

    special_set = set(SPECIAL_TOKENS)
    token_bytes = []
    for token_id in range(int(fast.vocab_size)):
        token_text = fast.decode([token_id], skip_special_tokens=False)
        token_bytes.append(0 if token_text in special_set else len(token_text.encode("utf-8")))
    torch.save(torch.tensor(token_bytes, dtype=torch.int32), token_bytes_path)

    test = "Hello world! Numbers: 123. Unicode: 你好"
    decoded = fast.decode(fast.encode(test, add_special_tokens=False), skip_special_tokens=False)
    if decoded != test:
        raise RuntimeError(f"Tokenizer roundtrip failed: {test!r} -> {decoded!r}")

    with (output_dir / "tokenizer_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset": args.dataset,
                "karpathy_split_pattern": SPLIT_PATTERN,
                "karpathy_special_tokens": SPECIAL_TOKENS,
                "source": str(source),
                "implementation": "tokenizers.ByteLevelBPETokenizer",
                "compatibility_note": (
                    "Matches Karpathy autoresearch vocab size and reserved tokens, "
                    "but is not bit-exact rustbpe/tiktoken."
                ),
                "train_file": str(train_file),
                "vocab_size": int(fast.vocab_size),
                "requested_vocab_size": args.vocab_size,
                "min_frequency": args.min_frequency,
                "max_chars": args.max_chars,
                "special_tokens": SPECIAL_TOKENS,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    print(f"saved tokenizer to {output_dir}")
    print(f"saved token_bytes to {token_bytes_path}")
    print(f"vocab_size={fast.vocab_size}")


if __name__ == "__main__":
    main()
