from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanoqwen.qwen3_model import DEFAULT_MODEL_PATH as QWEN3_PATH
from nanoqwen.qwen3_model import Qwen3LLM
from nanoqwen.qwen35_model import DEFAULT_MODEL_PATH as QWEN35_PATH
from nanoqwen.qwen35_model import Qwen35LLM

MODEL_FAMILIES = {
    "qwen3": (Qwen3LLM, QWEN3_PATH),
    "qwen35": (Qwen35LLM, QWEN35_PATH),
    "qwen3.5": (Qwen35LLM, QWEN35_PATH),
}


def parse_args(default_family: str | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate with local Qwen text-only LLMs through Transformers.")
    if default_family is None:
        parser.add_argument("--family", choices=sorted(MODEL_FAMILIES), default="qwen3")
    else:
        parser.set_defaults(family=default_family)
    parser.add_argument("--model", default=None, help="Override the default local model path for the selected family.")
    parser.add_argument("--prompt", default="Hello, what can you help me with?")
    parser.add_argument("--system", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["auto", "float32", "bfloat16", "float16"], default="auto")
    return parser.parse_args()


def build_llm(args: argparse.Namespace):
    model_cls, default_model_path = MODEL_FAMILIES[args.family]
    return model_cls(
        model_path=args.model or default_model_path,
        device=args.device,
        dtype=args.dtype,
    )


def main(default_family: str | None = None) -> None:
    args = parse_args(default_family=default_family)
    llm = build_llm(args)
    print(f"family={args.family}")
    print(f"model={llm.model_path}")
    print(
        "input_tokens="
        f"{llm.count_input_tokens(args.prompt, system=args.system, enable_thinking=args.enable_thinking)}"
    )

    if args.dry_run:
        print("dry_run_ok")
        return

    print(
        llm.generate(
            args.prompt,
            system=args.system,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            enable_thinking=args.enable_thinking,
        )
    )


if __name__ == "__main__":
    main()
