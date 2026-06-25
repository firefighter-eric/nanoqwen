from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file

from ..attention import normalize_attn_implementation
from ..config import NanoqwenConfig
from ..manual_text import ManualLLM, first_existing_file, resolve_dtype
from .qwen import NanoqwenForCausalLM

MODEL_NAME = "qwen3"
REPO_ID = "Qwen/Qwen3-0.6B"
DEFAULT_MODEL_PATH = "models/Qwen/Qwen3-0.6B"
REQUIRED_FILES = ("config.json", "tokenizer_config.json", "model.safetensors")


class Qwen3ForCausalLM(NanoqwenForCausalLM):
    @classmethod
    def from_pretrained(
        cls,
        model_path: str = DEFAULT_MODEL_PATH,
        dtype: str = "auto",
        attn_implementation: str | None = None,
    ) -> "Qwen3ForCausalLM":
        root = Path(model_path)
        config = NanoqwenConfig.from_json_file(root / "config.json")
        if attn_implementation is not None:
            config.attn_implementation = normalize_attn_implementation(attn_implementation)
        state = load_file(first_existing_file(root, ("model.safetensors",)), device="cpu")
        target_dtype = resolve_dtype(dtype, state["model.embed_tokens.weight"].dtype)

        model = cls(config)
        model.to(dtype=target_dtype)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Qwen3 weight load mismatch: missing={missing}, unexpected={unexpected}")
        return model


class Qwen3LLM(ManualLLM):
    model_cls = Qwen3ForCausalLM

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        device: str = "cpu",
        dtype: str = "auto",
        attn_implementation: str = "eager",
    ) -> None:
        super().__init__(
            model_path=model_path,
            device=device,
            dtype=dtype,
            attn_implementation=attn_implementation,
        )


def missing_files(model_path: str = DEFAULT_MODEL_PATH) -> list[str]:
    root = Path(model_path)
    return [name for name in REQUIRED_FILES if not (root / name).is_file()]


def require_downloaded(model_path: str = DEFAULT_MODEL_PATH) -> None:
    missing = missing_files(model_path)
    if missing:
        raise FileNotFoundError(
            f"{REPO_ID} is missing under {model_path}. "
            "Run: bash runs/download_qwen3_06b.sh. "
            f"Missing files: {', '.join(missing)}"
        )


__all__ = [
    "DEFAULT_MODEL_PATH",
    "MODEL_NAME",
    "Qwen3ForCausalLM",
    "Qwen3LLM",
    "REPO_ID",
    "REQUIRED_FILES",
    "missing_files",
    "require_downloaded",
]
