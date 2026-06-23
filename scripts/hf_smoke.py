from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanoqwen.checkpoint import SUPPORTED_HF_CAUSAL_LM_TYPES, import_hf_checkpoint
from nanoqwen.generation import generate
from nanoqwen.tokenizer import ByteTokenizer, load_hf_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test Hugging Face Qwen compatibility. By default this loads "
            "only config/tokenizer; pass --weights to import model weights."
        )
    )
    parser.add_argument("model_name_or_path", help="HF repo id or local HF-style checkpoint directory.")
    parser.add_argument("--weights", action="store_true", help="Also load weights into nanoqwen.")
    parser.add_argument("--prompt", default="Hello", help="Prompt used when --weights is set.")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def config_field(config, name: str):
    return getattr(config, name, None)


def print_config_summary(hf_config) -> bool:
    model_type = config_field(hf_config, "model_type")
    text_config = config_field(hf_config, "text_config")
    summary_config = text_config or hf_config
    native_supported = model_type in SUPPORTED_HF_CAUSAL_LM_TYPES
    print(
        "config_ok "
        f"model_type={model_type} "
        f"text_model_type={config_field(summary_config, 'model_type')} "
        f"layers={config_field(summary_config, 'num_hidden_layers')} "
        f"hidden={config_field(summary_config, 'hidden_size')} "
        f"heads={config_field(summary_config, 'num_attention_heads')} "
        f"kv_heads={config_field(summary_config, 'num_key_value_heads')} "
        f"vocab={config_field(summary_config, 'vocab_size')} "
        f"native_supported={native_supported}"
    )
    if not native_supported:
        print("native_import_unsupported use Transformers for this architecture")
    return native_supported


def main() -> None:
    args = parse_args()
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise ImportError("Install nanoqwen[hf] to use scripts/hf_smoke.py") from exc

    hf_config = AutoConfig.from_pretrained(args.model_name_or_path)
    native_supported = print_config_summary(hf_config)

    try:
        tokenizer = load_hf_tokenizer(args.model_name_or_path)
        tokenizer_vocab = getattr(tokenizer, "vocab_size", None)
        if tokenizer_vocab is not None and tokenizer_vocab <= 1:
            raise ValueError(f"tokenizer vocab_size={tokenizer_vocab} is not usable")
        print(
            "tokenizer_ok "
            f"class={tokenizer.__class__.__name__} "
            f"vocab={tokenizer_vocab if tokenizer_vocab is not None else 'unknown'} "
            f"eos={getattr(tokenizer, 'eos_token_id', None)}"
        )
    except Exception as exc:
        eos_token_id = getattr(getattr(hf_config, "text_config", hf_config), "eos_token_id", None)
        tokenizer = ByteTokenizer(eos_token_id=eos_token_id or 256)
        print(f"tokenizer_fallback class=ByteTokenizer reason={type(exc).__name__}")

    if not args.weights:
        print("weights_skipped pass --weights to import model weights")
        return
    if not native_supported:
        raise SystemExit("weights_import_skipped unsupported native architecture")

    model = import_hf_checkpoint(args.model_name_or_path, map_location=args.device)
    model.to(args.device).eval()
    if isinstance(tokenizer, ByteTokenizer):
        input_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=args.device)
    else:
        input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(args.device)
    output_ids = generate(
        model,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        do_sample=False,
    )
    print("weights_ok")
    if isinstance(tokenizer, ByteTokenizer):
        print(tokenizer.decode(output_ids[0].tolist()))
    else:
        print(tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True))


if __name__ == "__main__":
    main()
