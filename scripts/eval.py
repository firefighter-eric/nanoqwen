from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanoqwen.checkpoint import load_checkpoint
from nanoqwen.data import built_in_tiny_dataset, load_text_dataset
from nanoqwen.eval import evaluate_lm_loss, evaluate_multiple_choice_file, evaluate_prompt_file
from nanoqwen.tokenizer import ByteTokenizer, load_hf_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a nanoqwen checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default=None, help="Optional UTF-8 text file for loss/perplexity eval.")
    parser.add_argument("--prompts", default=None, help="Optional JSONL with prompt/completion rows.")
    parser.add_argument("--multiple-choice", default=None, help="Optional JSONL with prompt/question, choices, answer.")
    parser.add_argument("--choice-prefix", default=" ", help="Prefix added before each answer choice for scoring.")
    parser.add_argument("--hf-tokenizer", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, metadata = load_checkpoint(args.checkpoint, map_location=args.device)
    model.to(args.device)

    tokenizer = load_hf_tokenizer(args.hf_tokenizer) if args.hf_tokenizer else None
    if tokenizer is None:
        byte_tokenizer = ByteTokenizer(eos_token_id=model.config.eos_token_id or 256)
        encode = byte_tokenizer.encode
        decode = byte_tokenizer.decode
    else:
        encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
        decode = lambda ids: tokenizer.decode(ids, skip_special_tokens=True)

    dataset = (
        load_text_dataset(args.data, block_size=args.block_size, tokenizer=tokenizer)
        if args.data
        else built_in_tiny_dataset(block_size=args.block_size)
    )
    loss_result = evaluate_lm_loss(
        model,
        dataset,
        batch_size=args.batch_size,
        device=args.device,
        max_batches=args.max_batches,
    )
    print(
        f"loss={loss_result.loss:.4f} "
        f"ppl={loss_result.perplexity:.2f} "
        f"tokens={loss_result.tokens} "
        f"batches={loss_result.batches}"
    )

    if args.prompts:
        prompt_result = evaluate_prompt_file(
            model,
            args.prompts,
            encode=encode,
            decode=decode,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
        )
        print(
            f"prompt_exact_match={prompt_result.exact_match:.3f} "
            f"correct={prompt_result.correct}/{prompt_result.total}"
        )

    if args.multiple_choice:
        mc_result = evaluate_multiple_choice_file(
            model,
            args.multiple_choice,
            encode=encode,
            device=args.device,
            choice_prefix=args.choice_prefix,
        )
        print(
            f"multiple_choice_accuracy={mc_result.accuracy:.3f} "
            f"correct={mc_result.correct}/{mc_result.total}"
        )

    if metadata.get("step"):
        print(f"checkpoint_step={metadata['step']}")


if __name__ == "__main__":
    main()
