from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanoqwen.checkpoint import save_checkpoint
from nanoqwen.config import NanoqwenConfig
from nanoqwen.data import built_in_tiny_dataset, load_text_dataset
from nanoqwen.model import NanoqwenForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small Qwen-style language model.")
    parser.add_argument("--data", type=str, default=None, help="UTF-8 text file. Uses built-in text if omitted.")
    parser.add_argument("--out-dir", type=str, default="out/train")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-iters", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--vocab-size", type=int, default=257)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--intermediate-size", type=int, default=384)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--no-qk-norm", action="store_true")
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> NanoqwenConfig:
    return NanoqwenConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        num_key_value_heads=args.kv_heads,
        head_dim=args.hidden_size // args.heads,
        max_position_embeddings=max(512, args.block_size),
        rope_theta=10_000.0,
        use_qk_norm=not args.no_qk_norm,
        eos_token_id=args.vocab_size - 1,
    )


def batch_loss(model: NanoqwenForCausalLM, batch: tuple[torch.Tensor, torch.Tensor], device: str) -> torch.Tensor:
    input_ids, targets = (tensor.to(device) for tensor in batch)
    logits = model(input_ids=input_ids).logits
    return F.cross_entropy(logits.view(-1, model.config.vocab_size), targets.reshape(-1))


@torch.no_grad()
def estimate_loss(
    model: NanoqwenForCausalLM,
    loader: DataLoader,
    device: str,
    eval_iters: int,
) -> float:
    model.eval()
    losses = []
    for i, batch in enumerate(loader):
        if i >= eval_iters:
            break
        losses.append(batch_loss(model, batch, device).item())
    model.train()
    return float(sum(losses) / max(1, len(losses)))


def infinite_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    dataset = (
        load_text_dataset(args.data, block_size=args.block_size)
        if args.data
        else built_in_tiny_dataset(block_size=args.block_size)
    )
    val_size = max(1, min(len(dataset) // 10, 512))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    config = make_config(args)
    model = NanoqwenForCausalLM(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    data_iter = infinite_loader(train_loader)

    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    progress = tqdm(range(1, args.steps + 1), desc="train")
    for step in progress:
        loss = batch_loss(model, next(data_iter), args.device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        progress.set_postfix(loss=f"{loss.item():.3f}", ppl=f"{math.exp(min(loss.item(), 20)):.1f}")

        if args.eval_every > 0 and step % args.eval_every == 0:
            val_loss = estimate_loss(model, val_loader, args.device, args.eval_iters)
            print(f"step {step}: train_loss={loss.item():.4f} val_loss={val_loss:.4f}")

        if args.save_every > 0 and step % args.save_every == 0:
            save_checkpoint(model, args.out_dir, step=step, optimizer=optimizer)

    save_checkpoint(model, args.out_dir, step=args.steps, optimizer=optimizer)
    print(f"saved checkpoint to {Path(args.out_dir).resolve()}")


if __name__ == "__main__":
    main()
