from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .attention import normalize_attn_implementation


@dataclass
class NanoqwenConfig:
    """Small Qwen-style config.

    Defaults follow the public Qwen family shape, but local scripts use much
    smaller configs. Field names intentionally mirror Hugging Face Qwen configs
    where practical.
    """

    model_type: str = "qwen3"
    vocab_size: int = 151936
    hidden_size: int = 4096
    intermediate_size: int = 22016
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int | None = 32
    head_dim: int | None = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    attention_bias: bool = False
    attention_output_bias: bool | None = None
    attn_implementation: str = "eager"
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_theta: float = 1_000_000.0
    use_qk_norm: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None

    def __post_init__(self) -> None:
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.hidden_act != "silu":
            raise ValueError("Only silu is implemented in the minimal model")
        if self.attention_output_bias is None:
            self.attention_output_bias = self.attention_bias
        self.attn_implementation = normalize_attn_implementation(self.attn_implementation)

    @classmethod
    def tiny(cls, vocab_size: int = 257) -> "NanoqwenConfig":
        return cls(
            model_type="qwen3",
            vocab_size=vocab_size,
            hidden_size=128,
            intermediate_size=384,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            max_position_embeddings=512,
            rope_theta=10_000.0,
            eos_token_id=vocab_size - 1,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NanoqwenConfig":
        values = dict(data)
        if values.get("model_type") == "qwen2" and "use_qk_norm" not in values:
            values["use_qk_norm"] = False
        if values.get("model_type") == "qwen2" and values.get("attention_bias") is None:
            values["attention_bias"] = True
        if values.get("model_type") == "qwen2" and values.get("attention_output_bias") is None:
            values["attention_output_bias"] = False
        rope_parameters = values.pop("rope_parameters", None)
        if isinstance(rope_parameters, dict):
            values["rope_theta"] = rope_parameters.get(
                "rope_theta", rope_parameters.get("base", values.get("rope_theta", 1_000_000.0))
            )
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        filtered = {key: value for key, value in values.items() if key in allowed}
        return cls(**filtered)

    @classmethod
    def from_json_file(cls, path: str | Path) -> "NanoqwenConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["rope_parameters"] = {
            "rope_type": "default",
            "rope_theta": self.rope_theta,
        }
        return data

    def to_hf_dict(self) -> dict[str, Any]:
        data = self.to_dict()
        data.pop("use_qk_norm", None)
        data.pop("attention_output_bias", None)
        data.pop("attn_implementation", None)
        data.pop("rope_theta", None)
        return data

    def to_json_file(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
