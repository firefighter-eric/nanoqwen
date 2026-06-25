from __future__ import annotations

import pickle

import pytest

from nanoqwen.tokenizer import ByteTokenizer, TiktokenTokenizer, load_hf_tokenizer


def test_byte_tokenizer_roundtrip() -> None:
    tokenizer = ByteTokenizer()
    text = "hello qwen"
    ids = tokenizer.encode(text, add_eos=True)

    assert ids[-1] == tokenizer.eos_token_id
    assert tokenizer.decode(ids) == text


def test_tiktoken_tokenizer_adapter_loads_tokenizer_pkl(tmp_path) -> None:
    tiktoken = pytest.importorskip("tiktoken")
    encoding = tiktoken.Encoding(
        name="tiny-test",
        pat_str=r"\S+|\s+",
        mergeable_ranks={
            b"hello": 0,
            b" ": 1,
            b"world": 2,
        },
        special_tokens={"<|reserved_0|>": 3},
    )
    with (tmp_path / "tokenizer.pkl").open("wb") as handle:
        pickle.dump(encoding, handle)

    tokenizer = load_hf_tokenizer(str(tmp_path))

    assert isinstance(tokenizer, TiktokenTokenizer)
    assert tokenizer.vocab_size == 4
    assert tokenizer.get_vocab_size() == 4
    assert tokenizer.bos_token_id == 3
    assert tokenizer.get_bos_token_id() == 3
    assert tokenizer.encode("hello world") == [0, 1, 2]
    assert tokenizer.encode("hello world", add_special_tokens=True) == [3, 0, 1, 2]
    assert tokenizer.encode(["hello", "world"], prepend=3) == [[3, 0], [3, 2]]
    assert tokenizer.decode([3, 0, 1, 2], skip_special_tokens=True) == "hello world"
