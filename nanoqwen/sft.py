from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset

from .tokenizer import ByteTokenizer

IGNORE_INDEX = -100


@dataclass
class SupervisedExample:
    input_ids: list[int]
    assistant_mask: list[bool]


def _encode(tokenizer: Any, text: str, add_special_tokens: bool = False) -> list[int]:
    if tokenizer is None:
        tokenizer = ByteTokenizer()
    if isinstance(tokenizer, ByteTokenizer):
        return tokenizer.encode(text)
    return tokenizer.encode(text, add_special_tokens=add_special_tokens)


def _eos_token_id(tokenizer: Any) -> int | None:
    if tokenizer is None:
        return ByteTokenizer().eos_token_id
    return getattr(tokenizer, "eos_token_id", None)


def encode_text_example(text: str, tokenizer: Any = None, add_eos: bool = True) -> SupervisedExample:
    input_ids = _encode(tokenizer, text)
    assistant_mask = [True] * len(input_ids)
    eos_token_id = _eos_token_id(tokenizer)
    if add_eos and eos_token_id is not None:
        input_ids.append(eos_token_id)
        assistant_mask.append(True)
    return SupervisedExample(input_ids=input_ids, assistant_mask=assistant_mask)


def _simple_chat_encoding(
    messages: list[dict[str, str]],
    tokenizer: Any = None,
    add_eos: bool = True,
) -> SupervisedExample:
    input_ids: list[int] = []
    assistant_mask: list[bool] = []

    for message in messages:
        role = message["role"]
        content = message["content"]
        prefix_ids = _encode(tokenizer, f"{role}: ")
        content_ids = _encode(tokenizer, content)
        suffix_ids = _encode(tokenizer, "\n")

        input_ids.extend(prefix_ids)
        assistant_mask.extend([False] * len(prefix_ids))
        input_ids.extend(content_ids)
        assistant_mask.extend([role == "assistant"] * len(content_ids))
        input_ids.extend(suffix_ids)
        assistant_mask.extend([role == "assistant"] * len(suffix_ids))

    eos_token_id = _eos_token_id(tokenizer)
    if add_eos and eos_token_id is not None:
        input_ids.append(eos_token_id)
        assistant_mask.append(bool(messages and messages[-1]["role"] == "assistant"))
    return SupervisedExample(input_ids=input_ids, assistant_mask=assistant_mask)


def _try_hf_chat_template(messages: list[dict[str, str]], tokenizer: Any) -> SupervisedExample | None:
    if tokenizer is None or isinstance(tokenizer, ByteTokenizer) or not hasattr(tokenizer, "apply_chat_template"):
        return None
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
    except Exception:
        return None

    input_ids = encoded.get("input_ids")
    assistant_mask = encoded.get("assistant_masks")
    if input_ids is None or assistant_mask is None:
        return None
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    if assistant_mask and isinstance(assistant_mask[0], list):
        assistant_mask = assistant_mask[0]
    if len(input_ids) != len(assistant_mask) or not any(assistant_mask):
        return None
    return SupervisedExample(
        input_ids=[int(token) for token in input_ids],
        assistant_mask=[bool(value) for value in assistant_mask],
    )


def encode_chat_example(
    messages: list[dict[str, str]],
    tokenizer: Any = None,
    add_eos: bool = True,
) -> SupervisedExample:
    hf_example = _try_hf_chat_template(messages, tokenizer)
    if hf_example is not None:
        return hf_example
    return _simple_chat_encoding(messages, tokenizer=tokenizer, add_eos=add_eos)


def load_supervised_examples(path: str | Path, tokenizer: Any = None) -> list[SupervisedExample]:
    examples: list[SupervisedExample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if "text" in item:
                examples.append(encode_text_example(item["text"], tokenizer=tokenizer))
            elif "messages" in item:
                examples.append(encode_chat_example(item["messages"], tokenizer=tokenizer))
            elif "input_ids" in item and "labels" in item:
                labels = item["labels"]
                examples.append(
                    SupervisedExample(
                        input_ids=[int(token) for token in item["input_ids"]],
                        assistant_mask=[int(label) != IGNORE_INDEX for label in labels],
                    )
                )
            else:
                raise ValueError(
                    f"Row {line_number} must contain 'text', 'messages', or input_ids/labels"
                )
    return examples


class SupervisedTokenDataset(Dataset):
    """Fixed-length next-token SFT dataset.

    Each item returns `(input_ids, targets)`. Targets are shifted next-token ids,
    with `-100` where loss should be ignored.
    """

    def __init__(
        self,
        examples: Iterable[SupervisedExample],
        block_size: int,
        pad_token_id: int = 0,
    ) -> None:
        self.block_size = block_size
        self.pad_token_id = pad_token_id
        self.samples: list[tuple[torch.Tensor, torch.Tensor]] = []

        for example in examples:
            if len(example.input_ids) != len(example.assistant_mask):
                raise ValueError("input_ids and assistant_mask must have the same length")
            self.samples.extend(self._chunk(example))

        if not self.samples:
            raise ValueError("No supervised tokens found in dataset")

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        block_size: int,
        tokenizer: Any = None,
        pad_token_id: int | None = None,
    ) -> "SupervisedTokenDataset":
        if pad_token_id is None:
            pad_token_id = _eos_token_id(tokenizer) or 0
        return cls(
            load_supervised_examples(path, tokenizer=tokenizer),
            block_size=block_size,
            pad_token_id=pad_token_id,
        )

    def _chunk(self, example: SupervisedExample) -> list[tuple[torch.Tensor, torch.Tensor]]:
        samples = []
        window = self.block_size + 1
        for start in range(0, max(1, len(example.input_ids) - 1), self.block_size):
            ids = example.input_ids[start : start + window]
            mask = example.assistant_mask[start : start + window]
            if len(ids) < 2:
                continue
            input_ids = ids[:-1]
            targets = [
                token if supervised else IGNORE_INDEX
                for token, supervised in zip(ids[1:], mask[1:], strict=True)
            ]
            if all(target == IGNORE_INDEX for target in targets):
                continue

            pad_len = self.block_size - len(input_ids)
            if pad_len > 0:
                input_ids = input_ids + [self.pad_token_id] * pad_len
                targets = targets + [IGNORE_INDEX] * pad_len
            samples.append(
                (
                    torch.tensor(input_ids, dtype=torch.long),
                    torch.tensor(targets, dtype=torch.long),
                )
            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]

