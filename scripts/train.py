from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanoqwen.checkpoint import save_checkpoint
from nanoqwen.attention import ATTN_IMPLEMENTATION_CHOICES
from nanoqwen.config import NanoqwenConfig
from nanoqwen.data import built_in_tiny_dataset, load_text_dataset, make_autoresearch_packed_loader
from nanoqwen.models import CausalLM, GPTConfig, ModelConfig, NanoGPTConfig, model_from_config
from nanoqwen.tokenizer import load_hf_tokenizer
from dataset.registry import (
    available_text_datasets,
    ensure_text_dataset_shards,
    get_text_dataset_spec,
    materialize_text_dataset,
    parquet_source_paths_for_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small decoder-only language model.")
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument(
        "--data",
        type=str,
        default=None,
        help="UTF-8 text file. Uses built-in text if omitted.",
    )
    data_group.add_argument(
        "--dataset",
        choices=available_text_datasets(),
        default=None,
        help="Named dataset prepared under data/.",
    )
    parser.add_argument("--data-dir", default=None, help="Override named dataset data directory.")
    parser.add_argument("--val-data", default=None, help="Optional fixed validation text file.")
    parser.add_argument("--use-dataset-val", action="store_true", help="Use the named dataset validation split.")
    parser.add_argument(
        "--dataset-num-shards",
        type=int,
        default=None,
        help="Number of named dataset training shards to use. Use -1 for all shards.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download and prepare missing named dataset shards before training.",
    )
    parser.add_argument("--download-workers", type=int, default=4)
    parser.add_argument("--out-dir", type=str, default="out/train")
    parser.add_argument("--model", choices=("qwen", "gpt", "nanogpt"), default="qwen")
    parser.add_argument(
        "--data-format",
        choices=("text", "autoresearch"),
        default="text",
        help="Use text materialization or autoresearch parquet BOS packing.",
    )
    parser.add_argument("--tokenizer", default=None, help="Optional local/HF tokenizer path.")
    parser.add_argument("--max-data-chars", type=int, default=None)
    parser.add_argument("--max-val-chars", type=int, default=None)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument(
        "--time-budget-sec",
        type=float,
        default=None,
        help="Optional wall-clock training budget, excluding setup. Stops after this many seconds.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Number of micro-batches to accumulate before each optimizer step.",
    )
    parser.add_argument(
        "--total-batch-tokens",
        type=int,
        default=None,
        help="Optional effective tokens per optimizer step; overrides --grad-accum-steps.",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-iters", type=int, default=10)
    parser.add_argument(
        "--eval-tokens",
        type=int,
        default=None,
        help="Optional fixed validation token budget. Autoresearch uses 40*524288.",
    )
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--mixed-precision",
        choices=("auto", "none", "bf16"),
        default="auto",
        help="Mixed precision mode. auto uses bf16 on CUDA/MPS and fp32 on CPU.",
    )
    parser.add_argument(
        "--compile",
        choices=("auto", "on", "off"),
        default="off",
        help="Compile model forward with torch.compile. auto enables it on CUDA.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--vocab-size", type=int, default=257)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--intermediate-size", type=int, default=384)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--rope-theta", type=float, default=1_000_000.0)
    parser.add_argument("--window-pattern", default="L")
    parser.add_argument("--no-gpt-bias", action="store_true")
    parser.add_argument("--tie-word-embeddings", action="store_true")
    parser.add_argument("--no-qk-norm", action="store_true")
    parser.add_argument("--attn-implementation", choices=ATTN_IMPLEMENTATION_CHOICES, default="eager")
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> ModelConfig:
    if args.model == "gpt":
        return GPTConfig(
            vocab_size=args.vocab_size,
            block_size=args.block_size,
            n_layer=args.layers,
            n_head=args.heads,
            n_embd=args.hidden_size,
            dropout=args.dropout,
            bias=not args.no_gpt_bias,
            attn_implementation=args.attn_implementation,
            eos_token_id=args.vocab_size - 1,
        )
    if args.model == "nanogpt":
        return NanoGPTConfig(
            sequence_len=args.block_size,
            vocab_size=args.vocab_size,
            n_layer=args.layers,
            n_head=args.heads,
            n_kv_head=args.kv_heads or args.heads,
            n_embd=args.hidden_size,
            window_pattern=args.window_pattern,
            attn_implementation=args.attn_implementation,
            eos_token_id=args.vocab_size - 1,
        )

    return NanoqwenConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        num_key_value_heads=args.kv_heads or 2,
        head_dim=args.hidden_size // args.heads,
        max_position_embeddings=max(512, args.block_size),
        rope_theta=args.rope_theta,
        use_qk_norm=not args.no_qk_norm,
        tie_word_embeddings=args.tie_word_embeddings,
        attn_implementation=args.attn_implementation,
        eos_token_id=args.vocab_size - 1,
    )


def device_type(device: str | torch.device) -> str:
    return torch.device(device).type


def resolve_autocast_dtype(mixed_precision: str, device: str | torch.device) -> torch.dtype | None:
    if mixed_precision == "none":
        return None
    if mixed_precision == "bf16":
        return torch.bfloat16
    if mixed_precision == "auto":
        return torch.bfloat16 if device_type(device) in {"cuda", "mps"} else None
    raise ValueError(f"unknown mixed precision mode: {mixed_precision}")


def autocast_context(
    device: str | torch.device,
    dtype: torch.dtype | None,
) -> contextlib.AbstractContextManager:
    if dtype is None:
        return contextlib.nullcontext()
    return torch.amp.autocast(device_type=device_type(device), dtype=dtype)


def dtype_name(dtype: torch.dtype | None) -> str:
    return "none" if dtype is None else str(dtype).removeprefix("torch.")


def model_vocab_size(model: CausalLM) -> int:
    return int(model.config.vocab_size)


def resolve_compile_enabled(compile_mode: str, device: str | torch.device) -> bool:
    if compile_mode == "off":
        return False
    if compile_mode == "on":
        return True
    if compile_mode == "auto":
        return device_type(device) == "cuda"
    raise ValueError(f"unknown compile mode: {compile_mode}")


def batch_loss(
    model: CausalLM,
    batch: tuple[torch.Tensor, torch.Tensor],
    device: str,
    vocab_size: int,
    autocast_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    input_ids, targets = (tensor.to(device) for tensor in batch)
    with autocast_context(device, autocast_dtype):
        logits = model(input_ids=input_ids).logits
    return F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))


def resolve_grad_accum_steps(args: argparse.Namespace) -> int:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.block_size <= 0:
        raise ValueError("--block-size must be positive")
    if args.grad_accum_steps <= 0:
        raise ValueError("--grad-accum-steps must be positive")
    if args.total_batch_tokens is None:
        return args.grad_accum_steps

    micro_batch_tokens = args.batch_size * args.block_size
    if args.total_batch_tokens <= 0:
        raise ValueError("--total-batch-tokens must be positive")
    if args.total_batch_tokens % micro_batch_tokens != 0:
        raise ValueError(
            "--total-batch-tokens must be divisible by "
            f"--batch-size * --block-size ({micro_batch_tokens})"
        )
    return args.total_batch_tokens // micro_batch_tokens


def load_training_dataset(args: argparse.Namespace, tokenizer=None):
    if args.dataset:
        path = materialize_text_dataset(
            args.dataset,
            split="train",
            num_shards=args.dataset_num_shards,
            data_dir=args.data_dir,
            download=args.download,
            workers=args.download_workers,
        )
        print(f"dataset: {args.dataset} ({path})")
        return load_text_dataset(
            path,
            block_size=args.block_size,
            tokenizer=tokenizer,
            max_chars=args.max_data_chars,
        )

    if args.data:
        return load_text_dataset(
            args.data,
            block_size=args.block_size,
            tokenizer=tokenizer,
            max_chars=args.max_data_chars,
        )

    return built_in_tiny_dataset(block_size=args.block_size)


def load_validation_dataset(args: argparse.Namespace, tokenizer=None):
    if args.val_data:
        return load_text_dataset(
            args.val_data,
            block_size=args.block_size,
            tokenizer=tokenizer,
            max_chars=args.max_val_chars,
        )
    if args.dataset and args.use_dataset_val:
        path = materialize_text_dataset(
            args.dataset,
            split="val",
            num_shards=args.dataset_num_shards,
            data_dir=args.data_dir,
            download=args.download,
            workers=args.download_workers,
        )
        print(f"validation dataset: {args.dataset} ({path})")
        return load_text_dataset(
            path,
            block_size=args.block_size,
            tokenizer=tokenizer,
            max_chars=args.max_val_chars,
        )
    return None


def load_autoresearch_loaders(args: argparse.Namespace, tokenizer):
    if args.dataset is None:
        raise ValueError("--data-format autoresearch requires --dataset")
    if tokenizer is None:
        raise ValueError("--data-format autoresearch requires --tokenizer")

    ensure_text_dataset_shards(
        args.dataset,
        num_shards=args.dataset_num_shards,
        data_dir=args.data_dir,
        download=args.download,
        workers=args.download_workers,
        splits=("train", "val"),
    )
    spec = get_text_dataset_spec(args.dataset)
    train_paths = parquet_source_paths_for_split(
        args.dataset,
        "train",
        num_shards=args.dataset_num_shards,
        data_dir=args.data_dir,
    )
    val_paths = parquet_source_paths_for_split(
        args.dataset,
        "val",
        num_shards=args.dataset_num_shards,
        data_dir=args.data_dir,
    )
    print(f"dataset: {args.dataset} autoresearch parquet train shards={len(train_paths)}")
    print(f"validation dataset: {args.dataset} autoresearch parquet val shards={len(val_paths)}")
    train_loader = make_autoresearch_packed_loader(
        train_paths,
        tokenizer,
        batch_size=args.batch_size,
        block_size=args.block_size,
        text_column=spec.text_column,
        device=args.device,
    )
    val_loader = make_autoresearch_packed_loader(
        val_paths,
        tokenizer,
        batch_size=args.batch_size,
        block_size=args.block_size,
        text_column=spec.text_column,
        device=args.device,
    )
    data_info = {
        "format": "autoresearch",
        "dataset": args.dataset,
        "data": None,
        "val_data": None,
        "tokenizer": args.tokenizer,
        "max_data_chars": None,
        "max_val_chars": None,
        "train_size": None,
        "val_size": None,
        "train_paths": [str(path) for path in train_paths],
        "val_paths": [str(path) for path in val_paths],
        "eval_tokens": args.eval_tokens,
    }
    return train_loader, val_loader, data_info


@torch.no_grad()
def estimate_metrics(
    model: CausalLM,
    loader: DataLoader,
    device: str,
    eval_iters: int,
    vocab_size: int | None = None,
    token_bytes: torch.Tensor | None = None,
    autocast_dtype: torch.dtype | None = None,
) -> dict[str, float | None]:
    model.eval()
    vocab_size = vocab_size if vocab_size is not None else model_vocab_size(model)
    losses = []
    total_loss = 0.0
    total_bytes = 0
    for i, batch in enumerate(loader):
        if i >= eval_iters:
            break
        input_ids, targets = (tensor.to(device) for tensor in batch)
        with autocast_context(device, autocast_dtype):
            logits = model(input_ids=input_ids).logits
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1),
            reduction="mean",
        )
        losses.append(loss.item())
        if token_bytes is not None:
            loss_flat = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                targets.reshape(-1),
                reduction="none",
            )
            nbytes = token_bytes[targets.reshape(-1)]
            byte_mask = nbytes > 0
            total_loss += float((loss_flat * byte_mask).sum().item())
            total_bytes += int(nbytes.sum().item())
    model.train()
    val_loss = float(sum(losses) / max(1, len(losses)))
    val_bpb = None
    if token_bytes is not None and total_bytes > 0:
        val_bpb = total_loss / (total_bytes * math.log(2))
    return {"loss": val_loss, "bpb": val_bpb}


def estimate_loss(
    model: CausalLM,
    loader: DataLoader,
    device: str,
    eval_iters: int,
    vocab_size: int | None = None,
    autocast_dtype: torch.dtype | None = None,
) -> float:
    return float(
        estimate_metrics(
            model,
            loader,
            device,
            eval_iters,
            vocab_size,
            autocast_dtype=autocast_dtype,
        )["loss"]
    )


def infinite_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def get_tokenizer_vocab_size(tokenizer) -> int:
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is not None:
        return int(vocab_size)
    return int(len(tokenizer))


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    autocast_dtype = resolve_autocast_dtype(args.mixed_precision, args.device)
    compile_enabled = resolve_compile_enabled(args.compile, args.device)
    if device_type(args.device) == "cuda":
        torch.set_float32_matmul_precision("high")

    tokenizer = load_hf_tokenizer(args.tokenizer) if args.tokenizer else None
    if tokenizer is not None:
        tokenizer_vocab_size = get_tokenizer_vocab_size(tokenizer)
        if args.vocab_size != tokenizer_vocab_size:
            print(
                f"overriding --vocab-size {args.vocab_size} "
                f"with tokenizer vocab_size {tokenizer_vocab_size}"
            )
            args.vocab_size = tokenizer_vocab_size
    token_bytes = None
    if args.tokenizer:
        token_bytes_path = Path(args.tokenizer).expanduser() / "token_bytes.pt"
        if token_bytes_path.is_file():
            token_bytes = torch.load(token_bytes_path, map_location=args.device)

    if args.data_format == "autoresearch":
        if args.eval_tokens is None:
            args.eval_tokens = 40 * 524288
        train_loader, val_loader, data_info = load_autoresearch_loaders(args, tokenizer)
        train_size = None
        val_size = None
    else:
        dataset = load_training_dataset(args, tokenizer=tokenizer)
        fixed_val_dataset = load_validation_dataset(args, tokenizer=tokenizer)
        if fixed_val_dataset is None:
            val_size = max(1, min(len(dataset) // 10, 512))
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = random_split(
                dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(args.seed),
            )
        else:
            train_dataset = dataset
            val_dataset = fixed_val_dataset
            train_size = len(train_dataset)
            val_size = len(val_dataset)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
        data_info = {
            "format": "text",
            "dataset": args.dataset,
            "data": args.data,
            "val_data": args.val_data,
            "tokenizer": args.tokenizer,
            "max_data_chars": args.max_data_chars,
            "max_val_chars": args.max_val_chars,
            "train_size": train_size,
            "val_size": val_size,
            "eval_tokens": args.eval_tokens,
        }

    eval_iters = args.eval_iters
    if args.eval_tokens is not None:
        eval_iters = max(1, args.eval_tokens // (args.batch_size * args.block_size))

    grad_accum_steps = resolve_grad_accum_steps(args)
    effective_batch_tokens = args.batch_size * args.block_size * grad_accum_steps
    effective_batch_sequences = args.batch_size * grad_accum_steps

    config = make_config(args)
    model = model_from_config(config).to(args.device)
    train_model = torch.compile(model, dynamic=False) if compile_enabled else model
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    data_iter = train_loader if args.data_format == "autoresearch" else infinite_loader(train_loader)
    output_dir = Path(args.out_dir)
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    write_json(
        output_dir / "training_params.json",
        {
            "args": vars(args),
            "config": config.to_dict(),
            "data": data_info,
            "parameters": {
                "total": num_params,
                "trainable": trainable_params,
            },
            "batching": {
                "micro_batch_sequences": args.batch_size,
                "micro_batch_tokens": args.batch_size * args.block_size,
                "grad_accum_steps": grad_accum_steps,
                "effective_batch_sequences": effective_batch_sequences,
                "effective_batch_tokens": effective_batch_tokens,
            },
            "precision": {
                "mixed_precision": args.mixed_precision,
                "autocast_dtype": dtype_name(autocast_dtype),
                "device_type": device_type(args.device),
            },
            "compile": {
                "mode": args.compile,
                "enabled": compile_enabled,
            },
        },
    )

    print(f"parameters: {num_params / 1e6:.2f}M")
    print(
        "batching: "
        f"micro_sequences={args.batch_size} "
        f"micro_tokens={args.batch_size * args.block_size} "
        f"grad_accum_steps={grad_accum_steps} "
        f"effective_sequences={effective_batch_sequences} "
        f"effective_tokens={effective_batch_tokens}"
    )
    print(f"precision: mixed_precision={args.mixed_precision} autocast_dtype={dtype_name(autocast_dtype)}")
    print(f"compile: mode={args.compile} enabled={compile_enabled}")
    progress = tqdm(range(1, args.steps + 1), desc="train")
    train_start = time.monotonic()
    final_step = 0
    last_loss = None
    for step in progress:
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(grad_accum_steps):
            loss = batch_loss(
                train_model,
                next(data_iter),
                args.device,
                model.config.vocab_size,
                autocast_dtype=autocast_dtype,
            )
            accum_loss += float(loss.detach().item())
            (loss / grad_accum_steps).backward()
        last_loss = accum_loss / grad_accum_steps
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final_step = step

        progress.set_postfix(loss=f"{last_loss:.3f}", ppl=f"{math.exp(min(last_loss, 20)):.1f}")

        if args.eval_every > 0 and step % args.eval_every == 0:
            val_loss = estimate_loss(
                train_model,
                val_loader,
                args.device,
                eval_iters,
                model.config.vocab_size,
                autocast_dtype=autocast_dtype,
            )
            print(f"step {step}: train_loss={last_loss:.4f} val_loss={val_loss:.4f}")

        if args.save_every > 0 and step % args.save_every == 0:
            save_checkpoint(model, args.out_dir, step=step, optimizer=optimizer)

        if args.time_budget_sec is not None and (time.monotonic() - train_start) >= args.time_budget_sec:
            break

    elapsed_sec = time.monotonic() - train_start
    final_metrics = estimate_metrics(
        train_model,
        val_loader,
        args.device,
        eval_iters,
        model.config.vocab_size,
        token_bytes=token_bytes,
        autocast_dtype=autocast_dtype,
    )
    final_val_loss = float(final_metrics["loss"])
    final_val_bpb = final_metrics["bpb"]
    save_checkpoint(model, args.out_dir, step=final_step, optimizer=optimizer)
    write_json(
        output_dir / "result.json",
        {
            "elapsed_sec": elapsed_sec,
            "parameters": num_params,
            "step": final_step,
            "train_loss": last_loss,
            "val_loss": final_val_loss,
            "val_ppl": math.exp(min(final_val_loss, 20)),
            "val_bpb": final_val_bpb,
        },
    )
    bpb_text = f" val_bpb={final_val_bpb:.6f}" if final_val_bpb is not None else ""
    print(f"final: step={final_step} val_loss={final_val_loss:.4f}{bpb_text} elapsed_sec={elapsed_sec:.1f}")
    print(f"saved checkpoint to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
