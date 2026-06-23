from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from .model import NanoqwenForCausalLM
from .sft import IGNORE_INDEX, _encode, _eos_token_id
from .tokenizer import ByteTokenizer


@dataclass
class PreferenceExample:
    prompt_ids: list[int]
    chosen_ids: list[int]
    rejected_ids: list[int]


@dataclass
class DPOBatch:
    chosen_input_ids: torch.Tensor
    chosen_labels: torch.Tensor
    rejected_input_ids: torch.Tensor
    rejected_labels: torch.Tensor


@dataclass
class DPOStats:
    loss: torch.Tensor
    chosen_reward: torch.Tensor
    rejected_reward: torch.Tensor
    reward_margin: torch.Tensor
    accuracy: torch.Tensor


def _encode_with_eos(tokenizer: Any, text: str, add_eos: bool = True) -> list[int]:
    ids = _encode(tokenizer, text)
    eos_token_id = _eos_token_id(tokenizer)
    if add_eos and eos_token_id is not None:
        ids.append(eos_token_id)
    return ids


def _format_simple_prompt(messages: list[dict[str, str]], tokenizer: Any = None) -> list[int]:
    text = ""
    for message in messages:
        text += f"{message['role']}: {message['content']}\n"
    text += "assistant: "
    return _encode(tokenizer, text)


def encode_preference_item(item: dict[str, Any], tokenizer: Any = None) -> PreferenceExample:
    if "messages" in item:
        if tokenizer is not None and not isinstance(tokenizer, ByteTokenizer) and hasattr(
            tokenizer, "apply_chat_template"
        ):
            prompt_ids = tokenizer.apply_chat_template(
                item["messages"],
                tokenize=True,
                add_generation_prompt=True,
            )
        else:
            prompt_ids = _format_simple_prompt(item["messages"], tokenizer=tokenizer)
    elif "prompt" in item:
        prompt_ids = _encode(tokenizer, item["prompt"])
    else:
        raise ValueError("Preference rows must contain 'prompt' or 'messages'")

    if "chosen" not in item or "rejected" not in item:
        raise ValueError("Preference rows must contain 'chosen' and 'rejected'")

    return PreferenceExample(
        prompt_ids=[int(token) for token in prompt_ids],
        chosen_ids=_encode_with_eos(tokenizer, item["chosen"]),
        rejected_ids=_encode_with_eos(tokenizer, item["rejected"]),
    )


def load_preference_examples(path: str | Path, tokenizer: Any = None) -> list[PreferenceExample]:
    examples: list[PreferenceExample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                examples.append(encode_preference_item(json.loads(line), tokenizer=tokenizer))
            except Exception as exc:
                raise ValueError(f"Invalid preference row {line_number}: {exc}") from exc
    return examples


def _truncate_prompt_response(
    prompt_ids: list[int],
    response_ids: list[int],
    block_size: int,
) -> tuple[list[int], list[int]]:
    if block_size < 2:
        raise ValueError("block_size must be at least 2")
    response_ids = response_ids[: block_size - 1]
    prompt_budget = block_size - len(response_ids)
    prompt_ids = prompt_ids[-prompt_budget:] if len(prompt_ids) > prompt_budget else prompt_ids
    return prompt_ids, response_ids


def _make_sequence(
    prompt_ids: list[int],
    response_ids: list[int],
    block_size: int,
) -> tuple[list[int], list[int]]:
    prompt_ids, response_ids = _truncate_prompt_response(prompt_ids, response_ids, block_size)
    input_ids = prompt_ids + response_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + response_ids
    if len(input_ids) < 2 or all(label == IGNORE_INDEX for label in labels):
        raise ValueError("Preference sequence has no supervised response tokens")
    return input_ids, labels


class PreferenceDataset(Dataset):
    def __init__(self, examples: Iterable[PreferenceExample], block_size: int) -> None:
        self.block_size = block_size
        self.samples: list[tuple[list[int], list[int], list[int], list[int]]] = []

        for example in examples:
            chosen_input_ids, chosen_labels = _make_sequence(
                example.prompt_ids, example.chosen_ids, block_size
            )
            rejected_input_ids, rejected_labels = _make_sequence(
                example.prompt_ids, example.rejected_ids, block_size
            )
            self.samples.append(
                (chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels)
            )

        if not self.samples:
            raise ValueError("No preference examples found")

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        block_size: int,
        tokenizer: Any = None,
    ) -> "PreferenceDataset":
        return cls(load_preference_examples(path, tokenizer=tokenizer), block_size=block_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[list[int], list[int], list[int], list[int]]:
        return self.samples[idx]


def _pad(sequences: list[list[int]], pad_value: int) -> torch.Tensor:
    max_len = max(len(sequence) for sequence in sequences)
    rows = [sequence + [pad_value] * (max_len - len(sequence)) for sequence in sequences]
    return torch.tensor(rows, dtype=torch.long)


def collate_preference_batch(
    samples: list[tuple[list[int], list[int], list[int], list[int]]],
    pad_token_id: int = 0,
) -> DPOBatch:
    chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels = zip(*samples, strict=True)
    return DPOBatch(
        chosen_input_ids=_pad(list(chosen_input_ids), pad_token_id),
        chosen_labels=_pad(list(chosen_labels), IGNORE_INDEX),
        rejected_input_ids=_pad(list(rejected_input_ids), pad_token_id),
        rejected_labels=_pad(list(rejected_labels), IGNORE_INDEX),
    )


def sequence_logprobs(
    model: NanoqwenForCausalLM,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    average_logprob: bool = False,
) -> torch.Tensor:
    logits = model(input_ids=input_ids).logits[:, :-1, :]
    target_ids = input_ids[:, 1:]
    target_labels = labels[:, 1:]
    mask = target_labels != IGNORE_INDEX
    token_logprobs = F.log_softmax(logits, dim=-1).gather(
        dim=-1,
        index=target_ids.unsqueeze(-1),
    ).squeeze(-1)
    logprob_sums = (token_logprobs * mask).sum(dim=-1)
    if average_logprob:
        counts = mask.sum(dim=-1).clamp_min(1)
        return logprob_sums / counts
    return logprob_sums


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> DPOStats:
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = beta * (policy_logratios - ref_logratios)
    losses = -F.logsigmoid(logits)
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()
    return DPOStats(
        loss=losses.mean(),
        chosen_reward=chosen_rewards.mean(),
        rejected_reward=rejected_rewards.mean(),
        reward_margin=(chosen_rewards - rejected_rewards).mean(),
        accuracy=(chosen_rewards > rejected_rewards).float().mean(),
    )

