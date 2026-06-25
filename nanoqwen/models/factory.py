from __future__ import annotations

import json
from pathlib import Path

from ..config import NanoqwenConfig
from .gpt import GPTConfig, GPTForCausalLM
from .qwen import NanoqwenForCausalLM

ModelConfig = NanoqwenConfig | GPTConfig
CausalLM = NanoqwenForCausalLM | GPTForCausalLM

QWEN_MODEL_TYPES = {"qwen2", "qwen3"}


def config_from_dict(data: dict) -> ModelConfig:
    model_type = data.get("model_type", "qwen3")
    if model_type == "gpt":
        return GPTConfig.from_dict(data)
    if model_type in QWEN_MODEL_TYPES:
        return NanoqwenConfig.from_dict(data)
    raise ValueError(f"Unsupported native model_type={model_type!r}")


def config_from_json_file(path: str | Path) -> ModelConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        return config_from_dict(json.load(handle))


def model_from_config(config: ModelConfig) -> CausalLM:
    if isinstance(config, GPTConfig):
        return GPTForCausalLM(config)
    if isinstance(config, NanoqwenConfig):
        return NanoqwenForCausalLM(config)
    raise TypeError(f"Unsupported native config class: {type(config).__name__}")


__all__ = [
    "CausalLM",
    "ModelConfig",
    "config_from_dict",
    "config_from_json_file",
    "model_from_config",
]
