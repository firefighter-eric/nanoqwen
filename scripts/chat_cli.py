from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanoqwen.checkpoint import load_checkpoint
from nanoqwen.generation import generate
from nanoqwen.tokenizer import ByteTokenizer, apply_chat_template, load_hf_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with a nanoqwen checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--hf-tokenizer", type=str, default=None)
    parser.add_argument("--system", type=str, default="You are nanoqwen, a concise assistant.")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, _ = load_checkpoint(args.checkpoint, map_location=args.device)
    model.to(args.device)

    if args.hf_tokenizer:
        tokenizer = load_hf_tokenizer(args.hf_tokenizer)
        encode_prompt = lambda messages: tokenizer(
            apply_chat_template(tokenizer, messages),
            return_tensors="pt",
        ).input_ids.to(args.device)
        decode = lambda ids: tokenizer.decode(ids, skip_special_tokens=True)
    else:
        tokenizer = ByteTokenizer(eos_token_id=model.config.eos_token_id or 256)

        def encode_prompt(messages):
            text = ""
            for message in messages:
                text += f"{message['role']}: {message['content']}\n"
            text += "assistant:"
            return torch.tensor([tokenizer.encode(text)], dtype=torch.long, device=args.device)

        decode = tokenizer.decode

    messages = [{"role": "system", "content": args.system}]
    print("Type Ctrl-D or an empty line to exit.")
    while True:
        try:
            user_text = input("> ").strip()
        except EOFError:
            print()
            break
        if not user_text:
            break
        messages.append({"role": "user", "content": user_text})
        input_ids = encode_prompt(messages)
        output_ids = generate(
            model,
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        generated = decode(output_ids[0, input_ids.shape[1] :].tolist()).strip()
        print(generated)
        messages.append({"role": "assistant", "content": generated})


if __name__ == "__main__":
    main()
