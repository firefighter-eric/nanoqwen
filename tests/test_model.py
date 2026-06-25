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


def test_gradient_checkpointing_backward_disables_cache() -> None:
    model = tiny_model().train()
    model.gradient_checkpointing_enable()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 8))

    outputs = model(input_ids=input_ids, labels=input_ids, use_cache=True)
    assert outputs.loss is not None
    assert outputs.past_key_values is None
    outputs.loss.backward()

    assert model.is_gradient_checkpointing
    assert model.model.embed_tokens.weight.grad is not None
    model.gradient_checkpointing_disable()
    assert not model.is_gradient_checkpointing


def test_gradient_checkpointing_matches_regular_gradients() -> None:
    base = tiny_model().train()
    checkpointed = tiny_model().train()
    checkpointed.load_state_dict(base.state_dict())
    checkpointed.gradient_checkpointing_enable()
    input_ids = torch.randint(0, base.config.vocab_size, (2, 8))

    base_loss = base(input_ids=input_ids, labels=input_ids, use_cache=False).loss
    checkpointed_loss = checkpointed(input_ids=input_ids, labels=input_ids, use_cache=False).loss
    assert base_loss is not None
    assert checkpointed_loss is not None
    base_loss.backward()
    checkpointed_loss.backward()

    assert torch.allclose(base_loss, checkpointed_loss, atol=1e-6, rtol=1e-6)
    for (base_name, base_param), (checkpointed_name, checkpointed_param) in zip(
        base.named_parameters(),
        checkpointed.named_parameters(),
    ):
        assert base_name == checkpointed_name
        assert checkpointed_param.grad is not None
        assert torch.allclose(base_param.grad, checkpointed_param.grad, atol=1e-5, rtol=1e-4)


def test_generate_extends_sequence() -> None:
    model = tiny_model().eval()
    input_ids = torch.tensor([[1, 2, 3]])

    output_ids = generate(model, input_ids, max_new_tokens=4, do_sample=False)

    assert output_ids.shape == (1, 7)
    assert torch.equal(output_ids[:, :3], input_ids)
