from __future__ import annotations

import torch

from nanoqwen.config import NanoqwenConfig
from nanoqwen.generation import generate
from nanoqwen.model import NanoqwenForCausalLM


def tiny_model() -> NanoqwenForCausalLM:
    torch.manual_seed(1234)
    return NanoqwenForCausalLM(NanoqwenConfig.tiny(vocab_size=64))


def test_forward_shapes_and_loss() -> None:
    model = tiny_model()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 8))
    outputs = model(input_ids=input_ids, labels=input_ids)

    assert outputs.logits.shape == (2, 8, model.config.vocab_size)
    assert outputs.loss is not None
    assert outputs.loss.ndim == 0


def test_kv_cache_matches_full_forward_for_last_token() -> None:
    model = tiny_model().eval()
    input_ids = torch.randint(0, model.config.vocab_size, (1, 7))

    with torch.no_grad():
        full = model(input_ids=input_ids).logits[:, -1, :]
        prefix = model(input_ids=input_ids[:, :-1], use_cache=True)
        cached = model(
            input_ids=input_ids[:, -1:],
            past_key_values=prefix.past_key_values,
            use_cache=True,
        ).logits[:, -1, :]

    assert torch.allclose(full, cached, atol=1e-5, rtol=1e-4)


def test_generate_extends_sequence() -> None:
    model = tiny_model().eval()
    input_ids = torch.tensor([[1, 2, 3]])

    output_ids = generate(model, input_ids, max_new_tokens=4, do_sample=False)

    assert output_ids.shape == (1, 7)
    assert torch.equal(output_ids[:, :3], input_ids)

