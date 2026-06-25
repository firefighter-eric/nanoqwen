from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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


class TiktokenTokenizer:
    """Small adapter for Karpathy/autoresearch tokenizer.pkl files."""

    def __init__(self, encoding: Any, bos_token: str = "<|reserved_0|>") -> None:
        self.encoding = encoding
        self.vocab_size = int(encoding.n_vocab)
        self.bos_token = bos_token
        self.bos_token_id = encoding.encode_single_token(bos_token)
        self.eos_token_id = None

    @classmethod
    def from_directory(cls, path: str | Path) -> "TiktokenTokenizer":
        import pickle

        path = Path(path).expanduser()
        with (path / "tokenizer.pkl").open("rb") as handle:
            encoding = pickle.load(handle)
        return cls(encoding)

    def get_vocab_size(self) -> int:
        return self.vocab_size

    def get_bos_token_id(self) -> int:
        return self.bos_token_id

    def encode(
        self,
        text: str | list[str],
        add_special_tokens: bool = False,
        prepend: int | str | None = None,
        num_threads: int = 8,
    ) -> list[int] | list[list[int]]:
        prepend_id = None
        if add_special_tokens:
            prepend_id = self.bos_token_id
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encoding.encode_single_token(prepend)

        if isinstance(text, str):
            ids = self.encoding.encode_ordinary(text)
            if prepend_id is not None:
                ids.insert(0, prepend_id)
            return ids

        ids_batch = self.encoding.encode_ordinary_batch(text, num_threads=num_threads)
        if prepend_id is not None:
            for ids in ids_batch:
                ids.insert(0, prepend_id)
        return ids_batch

    def decode(self, ids: list[int] | tuple[int, ...], skip_special_tokens: bool = False) -> str:
        if skip_special_tokens:
            special_ids = set(self.encoding._special_tokens.values())
            ids = [token for token in ids if token not in special_ids]
        return self.encoding.decode(list(ids))


def load_hf_tokenizer(model_name_or_path: str, **kwargs: Any) -> Any:
    path = Path(model_name_or_path).expanduser()
    if path.is_dir() and (path / "tokenizer.pkl").is_file():
        try:
            import tiktoken  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Loading tokenizer.pkl requires tiktoken. Install it with `uv sync --extra data`."
            ) from exc
        return TiktokenTokenizer.from_directory(path)

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("Install nanoqwen[hf] to use Hugging Face tokenizers") from exc
    return AutoTokenizer.from_pretrained(path if path.exists() else model_name_or_path, **kwargs)


def apply_chat_template(tokenizer: Any, messages: list[dict[str, str]], add_generation_prompt: bool = True) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    return "\n".join(f"{message['role']}: {message['content']}" for message in messages) + "\nassistant:"
