from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from nanoqwen.hf_multimodal import (
    build_messages,
    generate_with_transformers,
    processor_inputs,
    resolve_torch_dtype,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate with local Qwen/Qwen3.5-0.8B through Transformers."
    )
    parser.add_argument("--model", default="models/Qwen/Qwen3.5-0.8B")
    parser.add_argument("--prompt", default="Hello, what can you help me with?")
    parser.add_argument("--image", action="append", default=[], help="Image URL or local path. Can repeat.")
    parser.add_argument("--system", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Build processor inputs without loading weights.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "bfloat16", "float16"],
        default="auto",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from transformers import AutoModelForMultimodalLM, AutoProcessor
    except ImportError as exc:
        raise ImportError("Install with `uv sync --extra dev --extra vision`.") from exc

    messages = build_messages(args.prompt, images=args.image, system=args.system)
    processor = AutoProcessor.from_pretrained(args.model)
    inputs = processor_inputs(processor, messages, enable_thinking=args.enable_thinking)
    print(f"input_tokens={inputs['input_ids'].shape[-1]}")

    if args.dry_run:
        print("dry_run_ok")
        return

    model = AutoModelForMultimodalLM.from_pretrained(
        args.model,
        dtype=resolve_torch_dtype(args.dtype),
    )
    model.to(args.device).eval()
    print(
        generate_with_transformers(
            model,
            processor,
            messages,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            enable_thinking=args.enable_thinking,
            device=args.device,
        )
    )


if __name__ == "__main__":
    main()
