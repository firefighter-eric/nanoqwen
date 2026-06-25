from __future__ import annotations

import torch

from nanoqwen.generation import generate
from nanoqwen.gpt_model import GPTConfig, GPTForCausalLM
from nanoqwen.models import GPTForCausalLM as ModelsGPTForCausalLM


def tiny_gpt() -> GPTForCausalLM:
    torch.manual_seed(1234)
    return GPTForCausalLM(GPTConfig.tiny(vocab_size=64))


def test_gpt_forward_shapes_and_loss() -> None:
    model = tiny_gpt()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 8))
    outputs = model(input_ids=input_ids, labels=input_ids)

    assert outputs.logits.shape == (2, 8, model.config.vocab_size)
    assert outputs.loss is not None
    assert outputs.loss.ndim == 0


def test_gpt_backward() -> None:
    model = tiny_gpt().train()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 8))

    outputs = model(input_ids=input_ids, labels=input_ids)
    assert outputs.loss is not None
    outputs.loss.backward()

    assert model.model.transformer.wte.weight.grad is not None


def test_gpt_attention_mask_changes_padded_logits() -> None:
    model = tiny_gpt().eval()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 8))
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 0, 0],
        ]
    )

    with torch.no_grad():
        unmasked = model(input_ids=input_ids).logits
        masked = model(input_ids=input_ids, attention_mask=attention_mask).logits

    assert not torch.allclose(unmasked[1, -1], masked[1, -1])


def test_gpt_generate_extends_sequence_without_cache() -> None:
    model = tiny_gpt().eval()
    input_ids = torch.tensor([[1, 2, 3]])

    output_ids = generate(model, input_ids, max_new_tokens=4, do_sample=False)

    assert output_ids.shape == (1, 7)
    assert torch.equal(output_ids[:, :3], input_ids)


def test_gpt_new_and_compat_imports_share_class() -> None:
    assert GPTForCausalLM is ModelsGPTForCausalLM
