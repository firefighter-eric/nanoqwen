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


def test_nanogpt_autoresearch_optimizer_updates() -> None:
    model = tiny_nanogpt().train()
    optimizer = model.setup_autoresearch_optimizer(compile_steps=False)

    group_kinds = {group["kind"] for group in optimizer.param_groups}
    assert {"adamw", "muon"} <= group_kinds
    assert all("initial_lr" in group for group in optimizer.param_groups)

    input_ids = torch.randint(0, model.config.vocab_size, (2, 8))
    lm_head_before = model.lm_head.weight.detach().clone()
    matrix_before = model.model.transformer.h[0].attn.c_q.weight.detach().clone()
    outputs = model(input_ids=input_ids, labels=input_ids)
    assert outputs.loss is not None
    outputs.loss.backward()
    optimizer.step()

    assert not torch.equal(model.lm_head.weight, lm_head_before)
    assert not torch.equal(model.model.transformer.h[0].attn.c_q.weight, matrix_before)


def test_nanogpt_autoresearch_optimizer_can_ablate_muon_matrix_updates() -> None:
    model = tiny_nanogpt().train()
    optimizer = model.setup_autoresearch_optimizer(matrix_optimizer="adamw", compile_steps=False)

    matrix_groups = [group for group in optimizer.param_groups if group.get("role") == "matrix"]
    assert matrix_groups
    assert {group["kind"] for group in matrix_groups} == {"adamw"}

    input_ids = torch.randint(0, model.config.vocab_size, (2, 8))
    matrix_before = model.model.transformer.h[0].attn.c_q.weight.detach().clone()
    outputs = model(input_ids=input_ids, labels=input_ids)
    assert outputs.loss is not None
    outputs.loss.backward()
    optimizer.step()

    assert not torch.equal(model.model.transformer.h[0].attn.c_q.weight, matrix_before)


def test_nanogpt_autoresearch_training_dtype_casts_high_traffic_embeddings() -> None:
    model = tiny_nanogpt()
    model.prepare_autoresearch_training_dtype(torch.bfloat16)

    assert model.model.transformer.wte.weight.dtype == torch.bfloat16
    assert {embedding.weight.dtype for embedding in model.model.value_embeds.values()} == {torch.bfloat16}
    assert model.lm_head.weight.dtype == torch.float32


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
