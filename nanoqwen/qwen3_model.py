from __future__ import annotations

from pathlib import Path

from nanoqwen.hf_text import HFTextCausalLM

MODEL_NAME = "qwen3"
REPO_ID = "Qwen/Qwen3-0.6B"
DEFAULT_MODEL_PATH = "models/Qwen/Qwen3-0.6B"
REQUIRED_FILES = ("config.json", "tokenizer_config.json", "model.safetensors")


class Qwen3LLM(HFTextCausalLM):
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, device: str = "cpu", dtype: str = "auto") -> None:
        super().__init__(model_path=model_path, device=device, dtype=dtype)


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
