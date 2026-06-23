from __future__ import annotations

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
) -> PackedTokenDataset:
    text = Path(path).read_text(encoding="utf-8")
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
