from __future__ import annotations

import json
from types import SimpleNamespace

import torch

from nanoqwen.config import NanoqwenConfig
from nanoqwen.data import built_in_tiny_dataset
from nanoqwen.eval import evaluate_lm_loss, evaluate_multiple_choice_file, score_completion
from nanoqwen.model import NanoqwenForCausalLM
from nanoqwen.tokenizer import ByteTokenizer


def test_evaluate_lm_loss_smoke() -> None:
    model = NanoqwenForCausalLM(NanoqwenConfig.tiny(vocab_size=257))
    dataset = built_in_tiny_dataset(block_size=16, repeats=4)

    result = evaluate_lm_loss(model, dataset, batch_size=2, device="cpu", max_batches=2)

    assert result.loss > 0
    assert result.perplexity > 1
    assert result.batches == 2
    assert result.tokens == 64


class TokenFavoringModel(torch.nn.Module):
    def __init__(self, vocab_size: int, favored_token: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(vocab_size=vocab_size)
        self.favored_token = favored_token

    def forward(self, input_ids):
        logits = torch.zeros(*input_ids.shape, self.config.vocab_size)
        logits[..., self.favored_token] = 10.0
        return SimpleNamespace(logits=logits)


def test_score_completion_prefers_high_logprob_token() -> None:
    tokenizer = ByteTokenizer()
    model = TokenFavoringModel(tokenizer.vocab_size, favored_token=ord("A"))

    score_a = score_completion(model, tokenizer.encode("q:"), tokenizer.encode("A"), device="cpu")
    score_b = score_completion(model, tokenizer.encode("q:"), tokenizer.encode("B"), device="cpu")

    assert score_a.mean_logprob > score_b.mean_logprob


def test_multiple_choice_eval_scores_choices(tmp_path) -> None:
    tokenizer = ByteTokenizer()
    model = TokenFavoringModel(tokenizer.vocab_size, favored_token=ord("A"))
    path = tmp_path / "mc.jsonl"
    path.write_text(
        json.dumps({"question": "letter?", "choices": ["A", "B"], "answer": 0}) + "\n",
        encoding="utf-8",
    )

    result = evaluate_multiple_choice_file(
        model,
        path,
        encode=tokenizer.encode,
        device="cpu",
        choice_prefix="",
    )

    assert result.accuracy == 1.0
    assert result.correct == 1
    assert result.total == 1
