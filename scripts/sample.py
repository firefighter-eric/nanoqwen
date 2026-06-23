from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanoqwen.checkpoint import load_checkpoint
from nanoqwen.generation import generate
from nanoqwen.tokenizer import ByteTokenizer, load_hf_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample from a nanoqwen checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="nanoqwen")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hf-tokenizer", type=str, default=None)
    parser.add_argument("--greedy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, metadata = load_checkpoint(args.checkpoint, map_location=args.device)
    model.to(args.device)

    if args.hf_tokenizer:
        tokenizer = load_hf_tokenizer(args.hf_tokenizer)
        input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(args.device)
        decode = lambda ids: tokenizer.decode(ids, skip_special_tokens=True)
    else:
        tokenizer = ByteTokenizer(eos_token_id=model.config.eos_token_id or 256)
        input_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=args.device)
        decode = tokenizer.decode

    output_ids = generate(
        model,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=not args.greedy,
    )
    print(decode(output_ids[0].tolist()))
    if metadata.get("step"):
        print(f"\n[checkpoint step: {metadata['step']}]")


if __name__ == "__main__":
    main()
