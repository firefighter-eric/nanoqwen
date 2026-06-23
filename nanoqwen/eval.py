from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader

from .data import PackedTokenDataset
from .generation import generate
from .model import NanoqwenForCausalLM


@dataclass
class LossEvalResult:
    loss: float
    perplexity: float
    batches: int
    tokens: int


@dataclass
class PromptEvalResult:
    exact_match: float
    correct: int
    total: int


@dataclass
class CompletionScore:
    total_logprob: float
    mean_logprob: float
    tokens: int


@dataclass
class MultipleChoiceEvalResult:
    accuracy: float
    correct: int
    total: int


def batch_next_token_loss(
    model: NanoqwenForCausalLM,
    batch: tuple[torch.Tensor, torch.Tensor],
    device: str | torch.device,
) -> torch.Tensor:
    input_ids, targets = (tensor.to(device) for tensor in batch)
    logits = model(input_ids=input_ids).logits
    return torch.nn.functional.cross_entropy(
        logits.view(-1, model.config.vocab_size),
        targets.reshape(-1),
        ignore_index=-100,
    )


@torch.no_grad()
def evaluate_lm_loss(
    model: NanoqwenForCausalLM,
    dataset: PackedTokenDataset,
    batch_size: int,
    device: str | torch.device,
    max_batches: int | None = None,
) -> LossEvalResult:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    total_loss = 0.0
    total_weight = 0
    batches = 0

    for batch in loader:
        if max_batches is not None and batches >= max_batches:
            break
        input_ids, targets = (tensor.to(device) for tensor in batch)
        logits = model(input_ids=input_ids).logits
        loss_sum = torch.nn.functional.cross_entropy(
            logits.view(-1, model.config.vocab_size),
            targets.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        supervised = int((targets != -100).sum().item())
        total_loss += float(loss_sum.item())
        total_weight += supervised
        batches += 1

    loss = total_loss / max(1, total_weight)
    return LossEvalResult(
        loss=loss,
        perplexity=math.exp(min(loss, 20)),
        batches=batches,
        tokens=total_weight,
    )


@torch.no_grad()
def evaluate_prompt_file(
    model: NanoqwenForCausalLM,
    path: str | Path,
    encode: Callable[[str], list[int]],
    decode: Callable[[list[int]], str],
    device: str | torch.device,
    max_new_tokens: int = 64,
) -> PromptEvalResult:
    model.eval()
    correct = 0
    total = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            prompt = item["prompt"]
            expected = item["completion"]
            input_ids = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
            output_ids = generate(
                model,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
            )
            generated = decode(output_ids[0, input_ids.shape[1] :].tolist())
            if generated.strip().startswith(expected.strip()):
                correct += 1
            total += 1
    return PromptEvalResult(
        exact_match=correct / max(1, total),
        correct=correct,
        total=total,
    )


@torch.no_grad()
def score_completion(
    model: NanoqwenForCausalLM,
    prompt_ids: list[int],
    completion_ids: list[int],
    device: str | torch.device,
) -> CompletionScore:
    if not completion_ids:
        raise ValueError("completion_ids must not be empty")
    if not prompt_ids:
        raise ValueError("prompt_ids must not be empty")

    input_ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long, device=device)
    logits = model(input_ids=input_ids).logits
    logprobs = torch.nn.functional.log_softmax(logits, dim=-1)

    start = len(prompt_ids)
    total = 0.0
    for offset, token_id in enumerate(completion_ids):
        position = start + offset
        total += float(logprobs[0, position - 1, token_id].item())
    return CompletionScore(
        total_logprob=total,
        mean_logprob=total / len(completion_ids),
        tokens=len(completion_ids),
    )


def _answer_to_index(answer, choices: list[str]) -> int:
    if isinstance(answer, int):
        return answer
    if isinstance(answer, str):
        stripped = answer.strip()
        if stripped in choices:
            return choices.index(stripped)
        letter = stripped.upper()
        if len(letter) == 1 and "A" <= letter <= "Z":
            return ord(letter) - ord("A")
    raise ValueError(f"Could not map answer {answer!r} to a choice")


@torch.no_grad()
def evaluate_multiple_choice_file(
    model: NanoqwenForCausalLM,
    path: str | Path,
    encode: Callable[[str], list[int]],
    device: str | torch.device,
    choice_prefix: str = " ",
) -> MultipleChoiceEvalResult:
    model.eval()
    correct = 0
    total = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            prompt = item.get("prompt", item.get("question"))
            choices = item["choices"]
            answer_index = _answer_to_index(item["answer"], choices)
            if answer_index < 0 or answer_index >= len(choices):
                raise ValueError(f"Row {line_number} answer index out of range")
            prompt_ids = encode(prompt)
            scores = [
                score_completion(
                    model,
                    prompt_ids=prompt_ids,
                    completion_ids=encode(choice_prefix + choice),
                    device=device,
                ).mean_logprob
                for choice in choices
            ]
            prediction = max(range(len(scores)), key=scores.__getitem__)
            correct += int(prediction == answer_index)
            total += 1
    return MultipleChoiceEvalResult(
        accuracy=correct / max(1, total),
        correct=correct,
        total=total,
    )
