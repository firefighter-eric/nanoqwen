from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ByteTokenizer:
    """Tiny deterministic tokenizer for smoke tests.

    Token ids 0..255 are raw UTF-8 bytes. `eos_token_id` defaults to 256.
    """

    eos_token_id: int = 256
    vocab_size: int = 257

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        ids = list(text.encode("utf-8"))
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: list[int] | tuple[int, ...]) -> str:
        raw = bytes(token for token in ids if 0 <= int(token) < 256)
        return raw.decode("utf-8", errors="replace")


def load_hf_tokenizer(model_name_or_path: str, **kwargs: Any) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("Install nanoqwen[hf] to use Hugging Face tokenizers") from exc
    return AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)


def apply_chat_template(tokenizer: Any, messages: list[dict[str, str]], add_generation_prompt: bool = True) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    return "\n".join(f"{message['role']}: {message['content']}" for message in messages) + "\nassistant:"

