from __future__ import annotations

import json

from nanoqwen.sft import IGNORE_INDEX, SupervisedTokenDataset, encode_chat_example
from nanoqwen.tokenizer import ByteTokenizer


def supervised_text(targets, tokenizer: ByteTokenizer) -> str:
    ids = [int(token) for token in targets.tolist() if int(token) != IGNORE_INDEX]
    return tokenizer.decode(ids)


def test_chat_example_masks_user_tokens() -> None:
    tokenizer = ByteTokenizer()
    example = encode_chat_example(
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "ok"},
        ],
        tokenizer=tokenizer,
    )
    dataset = SupervisedTokenDataset([example], block_size=64, pad_token_id=tokenizer.eos_token_id)
    _, targets = dataset[0]
    text = supervised_text(targets, tokenizer)

    assert text.startswith("ok")
    assert "hello" not in text


def test_text_rows_supervise_all_tokens(tmp_path) -> None:
    tokenizer = ByteTokenizer()
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps({"text": "abc"}) + "\n", encoding="utf-8")

    dataset = SupervisedTokenDataset.from_jsonl(
        path,
        block_size=8,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id,
    )
    _, targets = dataset[0]
    text = supervised_text(targets, tokenizer)

    assert text.startswith("bc")

