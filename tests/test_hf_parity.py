from __future__ import annotations

import importlib.util

import pytest
import torch

from nanoqwen.checkpoint import export_hf_checkpoint, import_hf_checkpoint
from nanoqwen.config import NanoqwenConfig
from nanoqwen.model import NanoqwenForCausalLM

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("transformers") is None,
    reason="transformers is not installed",
)


def test_tiny_qwen3_logits_match_transformers() -> None:
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(1234)
    config = NanoqwenConfig.tiny(vocab_size=64)
    ours = NanoqwenForCausalLM(config).eval()
    hf = Qwen3ForCausalLM(Qwen3Config(**config.to_hf_dict())).eval()
    missing, unexpected = hf.load_state_dict(ours.state_dict(), strict=False)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))

    with torch.no_grad():
        our_logits = ours(input_ids=input_ids).logits
        hf_logits = hf(input_ids=input_ids).logits

    assert missing == []
    assert unexpected == []
    assert torch.allclose(our_logits, hf_logits, atol=1e-5, rtol=1e-5)


def test_hf_export_and_import_roundtrip(tmp_path) -> None:
    from transformers import AutoModelForCausalLM

    torch.manual_seed(1234)
    config = NanoqwenConfig.tiny(vocab_size=64)
    model = NanoqwenForCausalLM(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    with torch.no_grad():
        expected = model(input_ids=input_ids).logits

    export_hf_checkpoint(model, tmp_path)
    hf_model = AutoModelForCausalLM.from_pretrained(tmp_path).eval()
    imported = import_hf_checkpoint(str(tmp_path)).eval()

    with torch.no_grad():
        hf_logits = hf_model(input_ids=input_ids).logits
        imported_logits = imported(input_ids=input_ids).logits

    assert torch.allclose(expected, hf_logits, atol=1e-5, rtol=1e-5)
    assert torch.allclose(expected, imported_logits, atol=1e-5, rtol=1e-5)


def test_tiny_qwen2_logits_match_transformers() -> None:
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(1234)
    hf_config = Qwen2Config(
        vocab_size=64,
        hidden_size=128,
        intermediate_size=384,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=512,
        rope_parameters={"rope_type": "default", "rope_theta": 10_000.0},
        attention_dropout=0.0,
        eos_token_id=63,
    )
    hf = Qwen2ForCausalLM(hf_config).eval()
    config = NanoqwenConfig.from_dict(hf_config.to_dict())
    ours = NanoqwenForCausalLM(config).eval()
    missing, unexpected = ours.load_state_dict(hf.state_dict(), strict=False)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))

    with torch.no_grad():
        our_logits = ours(input_ids=input_ids).logits
        hf_logits = hf(input_ids=input_ids).logits

    assert config.use_qk_norm is False
    assert config.attention_bias is True
    assert config.attention_output_bias is False
    assert missing == []
    assert unexpected == []
    assert torch.allclose(our_logits, hf_logits, atol=1e-5, rtol=1e-5)
