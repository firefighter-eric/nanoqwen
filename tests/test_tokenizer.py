from __future__ import annotations

from nanoqwen.tokenizer import ByteTokenizer


def test_byte_tokenizer_roundtrip() -> None:
    tokenizer = ByteTokenizer()
    text = "hello qwen"
    ids = tokenizer.encode(text, add_eos=True)

    assert ids[-1] == tokenizer.eos_token_id
    assert tokenizer.decode(ids) == text

