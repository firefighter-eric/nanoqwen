from __future__ import annotations

import pytest

from nanoqwen.checkpoint import native_config_from_hf_config
from nanoqwen.hf_multimodal import build_messages, generation_kwargs, image_content_item, is_url


def test_image_content_item_distinguishes_urls_and_paths() -> None:
    assert is_url("https://example.com/image.png")
    assert image_content_item("https://example.com/image.png") == {
        "type": "image",
        "url": "https://example.com/image.png",
    }
    assert image_content_item("/tmp/image.png") == {"type": "image", "image": "/tmp/image.png"}


def test_build_messages_uses_multimodal_content_shape() -> None:
    messages = build_messages(
        "Describe it.",
        images=["https://example.com/image.png"],
        system="Be concise.",
    )

    assert messages[0] == {"role": "system", "content": "Be concise."}
    assert messages[1]["role"] == "user"
    assert messages[1]["content"][0]["type"] == "image"
    assert messages[1]["content"][1] == {"type": "text", "text": "Describe it."}


class FakeHFConfig:
    model_type = "qwen3_5"

    def to_dict(self):
        return {"model_type": self.model_type}


def test_native_hf_import_rejects_multimodal_qwen35() -> None:
    with pytest.raises(ValueError, match="not supported"):
        native_config_from_hf_config(FakeHFConfig())


def test_generation_kwargs_are_greedy_when_temperature_zero() -> None:
    assert generation_kwargs(max_new_tokens=8, temperature=0.0) == {
        "max_new_tokens": 8,
        "do_sample": False,
    }


def test_generation_kwargs_enable_sampling_when_temperature_positive() -> None:
    assert generation_kwargs(max_new_tokens=8, temperature=0.7, top_p=0.8) == {
        "max_new_tokens": 8,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.8,
    }
