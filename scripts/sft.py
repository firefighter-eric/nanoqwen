from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanoqwen.checkpoint import load_checkpoint, save_checkpoint
from nanoqwen.sft import SupervisedTokenDataset
from nanoqwen.tokenizer import load_hf_tokenizer
from scripts.train import batch_loss, infinite_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal supervised fine-tuning on JSONL text/messages.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True, help="JSONL with either {'text': ...} or {'messages': [...]}.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--hf-tokenizer", default=None)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, metadata = load_checkpoint(args.checkpoint, map_location=args.device)
    model.to(args.device)
    model.train()
    tokenizer = load_hf_tokenizer(args.hf_tokenizer) if args.hf_tokenizer else None
    pad_token_id = (
        getattr(tokenizer, "pad_token_id", None)
        or getattr(tokenizer, "eos_token_id", None)
        or (model.config.eos_token_id if isinstance(model.config.eos_token_id, int) else None)
        or 0
    )

    dataset = SupervisedTokenDataset.from_jsonl(
        args.data,
        block_size=args.block_size,
        tokenizer=tokenizer,
        pad_token_id=pad_token_id,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    data_iter = infinite_loader(loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    for step in tqdm(range(1, args.steps + 1), desc="sft"):
        loss = batch_loss(model, next(data_iter), args.device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 25 == 0:
            print(f"step {step}: loss={loss.item():.4f}")

    save_checkpoint(
        model,
        args.out_dir,
        step=metadata.get("step", 0) + args.steps,
        optimizer=optimizer,
        extra={"base_checkpoint": args.checkpoint},
    )
    print(f"saved SFT checkpoint to {args.out_dir}")


if __name__ == "__main__":
    main()
