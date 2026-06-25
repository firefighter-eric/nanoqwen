from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .tokenizer import ByteTokenizer


class PackedTokenDataset(Dataset):
    def __init__(self, tokens: list[int], block_size: int) -> None:
        if len(tokens) <= block_size:
            raise ValueError("Need more tokens than block_size")
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.tokens[idx : idx + self.block_size + 1]
        return chunk[:-1], chunk[1:]


def parquet_document_batches(
    paths: Iterable[str | Path],
    text_column: str = "text",
    tokenizer_batch_size: int = 128,
) -> Iterator[tuple[list[str], int]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("autoresearch parquet data loading requires pyarrow") from exc

    parquet_paths = [Path(path) for path in paths]
    if not parquet_paths:
        raise ValueError("at least one parquet path is required")

    epoch = 1
    while True:
        for path in parquet_paths:
            parquet_file = pq.ParquetFile(path)
            for row_group_idx in range(parquet_file.num_row_groups):
                row_group = parquet_file.read_row_group(row_group_idx, columns=[text_column])
                docs = row_group.column(text_column).to_pylist()
                for i in range(0, len(docs), tokenizer_batch_size):
                    yield docs[i : i + tokenizer_batch_size], epoch
        epoch += 1


def make_autoresearch_packed_loader(
    paths: Iterable[str | Path],
    tokenizer,
    batch_size: int,
    block_size: int,
    *,
    text_column: str = "text",
    buffer_size: int = 1000,
    device: str | torch.device | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Autoresearch-compatible BOS document packing over parquet text shards."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    get_bos_token_id = getattr(tokenizer, "get_bos_token_id", None)
    bos_token_id = get_bos_token_id() if get_bos_token_id is not None else getattr(tokenizer, "bos_token_id", None)
    if bos_token_id is None:
        raise ValueError("autoresearch data loading requires tokenizer.bos_token_id")

    target_device = torch.device(device) if device is not None else None
    row_capacity = block_size + 1
    row_buffer = torch.empty((batch_size, row_capacity), dtype=torch.long)
    doc_buffer: list[list[int]] = []
    batches = parquet_document_batches(paths, text_column=text_column)

    def refill_buffer() -> None:
        docs, _ = next(batches)
        try:
            token_lists = tokenizer.encode(docs, prepend=bos_token_id)
        except TypeError:
            token_lists = [[bos_token_id, *tokenizer.encode(doc, add_special_tokens=False)] for doc in docs]
        doc_buffer.extend(token_lists)

    while True:
        for row_idx in range(batch_size):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos : pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                    pos += len(doc)
                else:
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos : pos + remaining] = torch.tensor(
                        doc[:remaining],
                        dtype=torch.long,
                    )
                    pos += remaining

        inputs = row_buffer[:, :-1].clone()
        targets = row_buffer[:, 1:].clone()
        if target_device is not None:
            inputs = inputs.to(target_device, non_blocking=target_device.type == "cuda")
            targets = targets.to(target_device, non_blocking=target_device.type == "cuda")
        yield inputs, targets


def encode_text(text: str, tokenizer=None, add_eos: bool = True) -> list[int]:
    if tokenizer is None:
        tokenizer = ByteTokenizer()
    if isinstance(tokenizer, ByteTokenizer):
        return tokenizer.encode(text, add_eos=add_eos)

    tokens = tokenizer.encode(text, add_special_tokens=False)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if add_eos and eos_token_id is not None:
        tokens.append(eos_token_id)
    return tokens


def dataset_from_text(
    text: str,
    block_size: int,
    tokenizer=None,
    add_eos: bool = True,
) -> PackedTokenDataset:
    tokens = encode_text(text, tokenizer=tokenizer, add_eos=add_eos)
    return PackedTokenDataset(tokens, block_size=block_size)


def load_text_dataset(
    path: str | Path,
    block_size: int,
    tokenizer=None,
    add_eos: bool = True,
    max_chars: int | None = None,
) -> PackedTokenDataset:
    if max_chars is None:
        text = Path(path).read_text(encoding="utf-8")
    else:
        with Path(path).open("r", encoding="utf-8") as handle:
            text = handle.read(max_chars)
    return dataset_from_text(text, block_size=block_size, tokenizer=tokenizer, add_eos=add_eos)


def built_in_tiny_text(repeats: int = 128) -> str:
    seed = (
        "nanoqwen is a tiny qwen style model. "
        "it learns next token prediction from small text. "
        "readable code makes experiments easier.\n"
    )
    return seed * repeats


def built_in_tiny_dataset(block_size: int, repeats: int = 128) -> PackedTokenDataset:
    return dataset_from_text(built_in_tiny_text(repeats), block_size=block_size)
