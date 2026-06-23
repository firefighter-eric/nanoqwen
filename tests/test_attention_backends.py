from __future__ import annotations

import importlib.util
from dataclasses import replace

import pytest
import torch

from nanoqwen.attention import causal_attention_bias, normalize_attn_implementation
from nanoqwen.config import NanoqwenConfig
from nanoqwen.model import NanoqwenForCausalLM
from nanoqwen.qwen35_model import Qwen35ForCausalLM, Qwen35TextConfig


def test_sdpa_matches_eager_for_tiny_model_with_padding() -> None:
    torch.manual_seed(1234)
    eager_config = NanoqwenConfig.tiny(vocab_size=64)
    sdpa_config = replace(eager_config, attn_implementation="sdpa")
    eager = NanoqwenForCausalLM(eager_config).eval()
    sdpa = NanoqwenForCausalLM(sdpa_config).eval()
    sdpa.load_state_dict(eager.state_dict())

    input_ids = torch.randint(0, eager_config.vocab_size, (2, 8))
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 0, 0],
        ]
    )

    with torch.no_grad():
        eager_logits = eager(input_ids=input_ids, attention_mask=attention_mask).logits
        sdpa_logits = sdpa(input_ids=input_ids, attention_mask=attention_mask).logits

    assert torch.allclose(eager_logits, sdpa_logits, atol=1e-5, rtol=1e-5)


def test_sdpa_kv_cache_matches_full_forward_for_last_token() -> None:
    torch.manual_seed(1234)
    config = NanoqwenConfig.tiny(vocab_size=64)
    config.attn_implementation = "sdpa"
    model = NanoqwenForCausalLM(config).eval()
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


def tiny_qwen35_config(attn_implementation: str) -> Qwen35TextConfig:
    return Qwen35TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        layer_types=["full_attention"],
        attn_implementation=attn_implementation,
        eos_token_id=63,
    )


def test_qwen35_full_attention_sdpa_matches_eager() -> None:
    torch.manual_seed(1234)
    eager = Qwen35ForCausalLM(tiny_qwen35_config("eager")).eval()
    sdpa = Qwen35ForCausalLM(tiny_qwen35_config("sdpa")).eval()
    sdpa.load_state_dict(eager.state_dict())
    input_ids = torch.randint(0, 64, (2, 6))
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0],
        ]
    )

    with torch.no_grad():
        eager_logits = eager(input_ids=input_ids, attention_mask=attention_mask).logits
        sdpa_logits = sdpa(input_ids=input_ids, attention_mask=attention_mask).logits

    assert torch.allclose(eager_logits, sdpa_logits, atol=1e-5, rtol=1e-5)


def test_flash_attention_reports_unavailable_runtime() -> None:
    if torch.cuda.is_available() and importlib.util.find_spec("flash_attn") is not None:
        pytest.skip("flash_attn is available on CUDA")

    config = NanoqwenConfig.tiny(vocab_size=64)
    config.attn_implementation = "flash_attention_2"
    model = NanoqwenForCausalLM(config).eval()
    input_ids = torch.randint(0, model.config.vocab_size, (1, 4))

    with pytest.raises((ImportError, RuntimeError), match="flash_attention_2"):
        model(input_ids=input_ids)


def test_flash_attention_aliases_normalize_to_flash_attention_2() -> None:
    for name in ("flash-attn2", "flash_attn2", "flash_attention2", "flash2", "fa2"):
        assert normalize_attn_implementation(name) == "flash_attention_2"


def test_flash_attention_with_bias_uses_sdpa_fallback() -> None:
    torch.manual_seed(1234)
    query = torch.randn(1, 2, 4, 8)
    key = torch.randn(1, 2, 4, 8)
    value = torch.randn(1, 2, 4, 8)
    attention_mask = torch.tensor([[1, 1, 1, 0]])
    attention_bias = causal_attention_bias(
        4,
        4,
        dtype=query.dtype,
        device=query.device,
        attention_mask=attention_mask,
    )

    from nanoqwen.attention import attention_forward

    output = attention_forward(
        "flash-attn2",
        query,
        key,
        value,
        attention_bias=attention_bias,
        dropout_p=0.0,
        scaling=query.shape[-1] ** -0.5,
        training=False,
        is_causal=True,
    )

    assert output.shape == query.shape


def test_invalid_attention_implementation_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported attention implementation"):
        NanoqwenConfig(attn_implementation="unknown")
