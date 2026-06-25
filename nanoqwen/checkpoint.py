from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch

from .config import NanoqwenConfig
from .models import CausalLM, config_from_json_file, model_from_config
from .model import NanoqwenForCausalLM

SUPPORTED_HF_CAUSAL_LM_TYPES = {"qwen2", "qwen3"}


def save_checkpoint(
    model: CausalLM,
    path: str | Path,
    step: int = 0,
    optimizer: torch.optim.Optimizer | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.config.to_json_file(path / "config.json")
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "step": step,
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path / "checkpoint.pt")


def load_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
    load_optimizer: bool = False,
) -> tuple[CausalLM, dict[str, Any]]:
    path = Path(path)
    config = config_from_json_file(path / "config.json")
    model = model_from_config(config)
    payload = torch.load(path / "checkpoint.pt", map_location=map_location)
    model.load_state_dict(payload["model"])
    metadata = {
        "step": payload.get("step", 0),
        "extra": payload.get("extra", {}),
    }
    if load_optimizer and "optimizer" in payload:
        metadata["optimizer"] = payload["optimizer"]
    return model, metadata


def native_config_from_hf_config(hf_config) -> NanoqwenConfig:
    model_type = getattr(hf_config, "model_type", None)
    if model_type not in SUPPORTED_HF_CAUSAL_LM_TYPES:
        raise ValueError(
            f"HF model_type={model_type!r} is not supported by the native text decoder. "
            "Use Transformers for multimodal or unsupported architectures."
        )
    return NanoqwenConfig.from_dict(hf_config.to_dict())


def import_hf_checkpoint(model_name_or_path: str, map_location: str | torch.device = "cpu") -> NanoqwenForCausalLM:
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as exc:
        raise ImportError("Install nanoqwen[hf] to import Hugging Face checkpoints") from exc

    hf_config = AutoConfig.from_pretrained(model_name_or_path)
    config = native_config_from_hf_config(hf_config)
    model = NanoqwenForCausalLM(config)
    hf_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype="auto")
    missing, unexpected = model.load_state_dict(hf_model.state_dict(), strict=False)
    allowed_missing = {"lm_head.weight"} if config.tie_word_embeddings else set()
    missing = [key for key in missing if key not in allowed_missing]
    if missing or unexpected:
        raise RuntimeError(f"HF state dict mismatch: missing={missing}, unexpected={unexpected}")
    return model.to(map_location)


def export_hf_checkpoint(
    model: NanoqwenForCausalLM,
    path: str | Path,
    tokenizer_source: str | Path | None = None,
) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    with (path / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(model.config.to_hf_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    torch.save(model.state_dict(), path / "pytorch_model.bin")

    if tokenizer_source is not None:
        source = Path(tokenizer_source)
        if source.is_dir():
            for child in source.iterdir():
                if child.is_file() and child.name.startswith(("tokenizer", "special_tokens", "generation_config")):
                    shutil.copy2(child, path / child.name)
