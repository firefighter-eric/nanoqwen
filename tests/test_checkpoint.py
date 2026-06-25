from __future__ import annotations

import torch

from nanoqwen.checkpoint import load_checkpoint, save_checkpoint
from nanoqwen.config import NanoqwenConfig
from nanoqwen.gpt_model import GPTConfig, GPTForCausalLM
from nanoqwen.model import NanoqwenForCausalLM


def test_checkpoint_roundtrip(tmp_path) -> None:
    torch.manual_seed(1234)
    model = NanoqwenForCausalLM(NanoqwenConfig.tiny(vocab_size=32))
    input_ids = torch.randint(0, model.config.vocab_size, (1, 6))
    expected = model(input_ids=input_ids).logits

    save_checkpoint(model, tmp_path, step=12, extra={"name": "roundtrip"})
    loaded, metadata = load_checkpoint(tmp_path)
    actual = loaded(input_ids=input_ids).logits

    assert metadata["step"] == 12
    assert metadata["extra"] == {"name": "roundtrip"}
    assert torch.allclose(expected, actual)


def test_gpt_checkpoint_roundtrip(tmp_path) -> None:
    torch.manual_seed(1234)
    model = GPTForCausalLM(GPTConfig.tiny(vocab_size=32))
    input_ids = torch.randint(0, model.config.vocab_size, (1, 6))
    expected = model(input_ids=input_ids).logits

    save_checkpoint(model, tmp_path, step=12, extra={"name": "gpt-roundtrip"})
    loaded, metadata = load_checkpoint(tmp_path)
    actual = loaded(input_ids=input_ids).logits

    assert isinstance(loaded, GPTForCausalLM)
    assert metadata["step"] == 12
    assert metadata["extra"] == {"name": "gpt-roundtrip"}
    assert torch.allclose(expected, actual)
