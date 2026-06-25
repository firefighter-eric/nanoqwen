from __future__ import annotations

from dataclasses import replace

import torch

from nanoqwen.generation import generate
from nanoqwen.nanogpt_model import NanoGPTConfig, NanoGPTForCausalLM
from nanoqwen.models import NanoGPTForCausalLM as ModelsNanoGPTForCausalLM
from nanoqwen.models import config_from_dict, model_from_config


def tiny_nanogpt() -> NanoGPTForCausalLM:
    torch.manual_seed(1234)
    return NanoGPTForCausalLM(NanoGPTConfig.tiny(vocab_size=64))


def test_nanogpt_forward_shapes_and_loss() -> None:
    model = tiny_nanogpt()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 8))
    outputs = model(input_ids=input_ids, labels=input_ids)

    assert outputs.logits.shape == (2, 8, model.config.vocab_size)
    assert outputs.loss is not None
    assert outputs.loss.ndim == 0


def test_nanogpt_backward() -> None:
    model = tiny_nanogpt().train()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 8))

    outputs = model(input_ids=input_ids, labels=input_ids)
    assert outputs.loss is not None
    outputs.loss.backward()

    assert model.model.transformer.wte.weight.grad is not None


def test_nanogpt_sdpa_matches_eager_for_window_attention() -> None:
    torch.manual_seed(1234)
    eager_config = NanoGPTConfig(
        sequence_len=8,
        vocab_size=64,
        n_layer=2,
        n_head=4,
        n_kv_head=4,
        n_embd=64,
        window_pattern="S",
        attn_implementation="eager",
    )
    sdpa_config = replace(eager_config, attn_implementation="sdpa")
    eager = NanoGPTForCausalLM(eager_config).eval()
    sdpa = NanoGPTForCausalLM(sdpa_config).eval()
    sdpa.load_state_dict(eager.state_dict())
    input_ids = torch.randint(0, eager.config.vocab_size, (2, 8))

    with torch.no_grad():
        eager_logits = eager(input_ids=input_ids).logits
        sdpa_logits = sdpa(input_ids=input_ids).logits

    assert torch.allclose(eager_logits, sdpa_logits, atol=1e-5, rtol=1e-5)


def test_nanogpt_generate_extends_sequence_without_cache() -> None:
    model = tiny_nanogpt().eval()
    input_ids = torch.tensor([[1, 2, 3]])

    output_ids = generate(model, input_ids, max_new_tokens=4, do_sample=False)

    assert output_ids.shape == (1, 7)
    assert torch.equal(output_ids[:, :3], input_ids)


def test_nanogpt_new_and_compat_imports_share_class() -> None:
    assert NanoGPTForCausalLM is ModelsNanoGPTForCausalLM


def test_nanogpt_factory_from_config_dict() -> None:
    config = config_from_dict(NanoGPTConfig.tiny(vocab_size=64).to_dict())
    model = model_from_config(config)

    assert isinstance(config, NanoGPTConfig)
    assert isinstance(model, NanoGPTForCausalLM)
