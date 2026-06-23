from __future__ import annotations

from nanoqwen.hf_text import build_chat_messages, render_chat_prompt, text_generation_kwargs


def test_build_chat_messages_text_only() -> None:
    assert build_chat_messages("hello") == [{"role": "user", "content": "hello"}]
    assert build_chat_messages("hello", system="be brief") == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hello"},
    ]


def test_text_generation_kwargs_greedy_and_sampling() -> None:
    assert text_generation_kwargs(max_new_tokens=4, temperature=0.0) == {
        "max_new_tokens": 4,
        "do_sample": False,
    }
    assert text_generation_kwargs(max_new_tokens=4, temperature=0.8, top_p=0.7) == {
        "max_new_tokens": 4,
        "do_sample": True,
        "temperature": 0.8,
        "top_p": 0.7,
    }


def test_render_chat_prompt_falls_back_without_enable_thinking() -> None:
    class LegacyTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            if "enable_thinking" in kwargs:
                raise TypeError("unexpected keyword argument 'enable_thinking'")
            assert kwargs == {"tokenize": False, "add_generation_prompt": True}
            return messages[-1]["content"]

    assert render_chat_prompt(LegacyTokenizer(), "hello") == "hello"
