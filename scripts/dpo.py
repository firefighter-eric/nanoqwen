from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanoqwen.checkpoint import load_checkpoint, save_checkpoint
from nanoqwen.dpo import (
    DPOBatch,
    PreferenceDataset,
    collate_preference_batch,
    dpo_loss,
    sequence_logprobs,
)
from nanoqwen.tokenizer import load_hf_tokenizer
from scripts.train import infinite_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minimal DPO preference tuning.")
    parser.add_argument("--checkpoint", required=True, help="Policy checkpoint to tune.")
    parser.add_argument("--reference-checkpoint", default=None, help="Frozen reference checkpoint. Defaults to policy.")
    parser.add_argument("--data", required=True, help="JSONL with prompt/messages, chosen, rejected.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--hf-tokenizer", default=None)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--average-logprob", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def move_batch(batch: DPOBatch, device: str) -> DPOBatch:
    return DPOBatch(
        chosen_input_ids=batch.chosen_input_ids.to(device),
        chosen_labels=batch.chosen_labels.to(device),
        rejected_input_ids=batch.rejected_input_ids.to(device),
        rejected_labels=batch.rejected_labels.to(device),
    )


def main() -> None:
    args = parse_args()
    model, metadata = load_checkpoint(args.checkpoint, map_location=args.device)
    reference_path = args.reference_checkpoint or args.checkpoint
    reference_model, _ = load_checkpoint(reference_path, map_location=args.device)
    model.to(args.device).train()
    reference_model.to(args.device).eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad_(False)

    tokenizer = load_hf_tokenizer(args.hf_tokenizer) if args.hf_tokenizer else None
    pad_token_id = (
        getattr(tokenizer, "pad_token_id", None)
        or getattr(tokenizer, "eos_token_id", None)
        or (model.config.eos_token_id if isinstance(model.config.eos_token_id, int) else None)
        or 0
    )
    dataset = PreferenceDataset.from_jsonl(
        args.data,
        block_size=args.block_size,
        tokenizer=tokenizer,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=lambda samples: collate_preference_batch(samples, pad_token_id=pad_token_id),
    )
    data_iter = infinite_loader(loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    progress = tqdm(range(1, args.steps + 1), desc="dpo")
    for step in progress:
        batch = move_batch(next(data_iter), args.device)
        policy_chosen = sequence_logprobs(
            model,
            batch.chosen_input_ids,
            batch.chosen_labels,
            average_logprob=args.average_logprob,
        )
        policy_rejected = sequence_logprobs(
            model,
            batch.rejected_input_ids,
            batch.rejected_labels,
            average_logprob=args.average_logprob,
        )
        with torch.no_grad():
            ref_chosen = sequence_logprobs(
                reference_model,
                batch.chosen_input_ids,
                batch.chosen_labels,
                average_logprob=args.average_logprob,
            )
            ref_rejected = sequence_logprobs(
                reference_model,
                batch.rejected_input_ids,
                batch.rejected_labels,
                average_logprob=args.average_logprob,
            )

        stats = dpo_loss(
            policy_chosen,
            policy_rejected,
            ref_chosen,
            ref_rejected,
            beta=args.beta,
        )
        optimizer.zero_grad(set_to_none=True)
        stats.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        progress.set_postfix(
            loss=f"{stats.loss.item():.3f}",
            margin=f"{stats.reward_margin.item():.3f}",
            acc=f"{stats.accuracy.item():.2f}",
        )
        if step % 25 == 0:
            print(
                f"step {step}: "
                f"loss={stats.loss.item():.4f} "
                f"margin={stats.reward_margin.item():.4f} "
                f"acc={stats.accuracy.item():.3f}"
            )

    save_checkpoint(
        model,
        args.out_dir,
        step=metadata.get("step", 0) + args.steps,
        optimizer=optimizer,
        extra={
            "base_checkpoint": args.checkpoint,
            "reference_checkpoint": reference_path,
            "objective": "dpo",
            "beta": args.beta,
        },
    )
    print(f"saved DPO checkpoint to {args.out_dir}")


if __name__ == "__main__":
    main()
