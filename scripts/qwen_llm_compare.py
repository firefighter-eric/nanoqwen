from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanoqwen.hf_text import render_chat_prompt, text_generation_kwargs
from nanoqwen.qwen3_model import DEFAULT_MODEL_PATH as QWEN3_PATH
from nanoqwen.qwen3_model import Qwen3LLM
from nanoqwen.qwen35_model import DEFAULT_MODEL_PATH as QWEN35_PATH
from nanoqwen.qwen35_model import Qwen35LLM

MODEL_FAMILIES = {
    "qwen3": (Qwen3LLM, QWEN3_PATH, "qwen3_compare_ok"),
    "qwen35": (Qwen35LLM, QWEN35_PATH, "qwen35_compare_ok"),
    "qwen3.5": (Qwen35LLM, QWEN35_PATH, "qwen35_compare_ok"),
}


def parse_args(default_family: str | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare nanoqwen's hand-written Qwen LLMs against a direct Transformers call."
    )
    if default_family is None:
        parser.add_argument("--family", choices=sorted(MODEL_FAMILIES), default="qwen3")
    else:
        parser.set_defaults(family=default_family)
    parser.add_argument("--model", default=None, help="Override the default local model path for the selected family.")
    parser.add_argument("--prompt", default="Say hello in one short sentence.")
    parser.add_argument("--system", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["auto", "float32", "bfloat16", "float16"], default="auto")
    return parser.parse_args()


def build_llm(args: argparse.Namespace):
    model_cls, default_model_path, _ = MODEL_FAMILIES[args.family]
    return model_cls(
        model_path=args.model or default_model_path,
        device=args.device,
        dtype=args.dtype,
    )


@torch.no_grad()
def direct_transformers_generate(
    model,
    tokenizer,
    prompt: str,
    system: str | None,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    enable_thinking: bool,
    device: str,
) -> str:
    rendered = render_chat_prompt(
        tokenizer,
        prompt,
        system=system,
        enable_thinking=enable_thinking,
    )
    inputs = tokenizer(rendered, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        **text_generation_kwargs(max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p),
    )
    generated_ids = outputs[0][inputs.input_ids.shape[-1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def main(default_family: str | None = None) -> None:
    args = parse_args(default_family=default_family)
    llm = build_llm(args)
    tokenizer = llm.load_tokenizer()

    project_output = llm.generate(
        args.prompt,
        system=args.system,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        enable_thinking=args.enable_thinking,
    )
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise ImportError("Install with `uv sync --extra dev`.") from exc

    hf_model = AutoModelForCausalLM.from_pretrained(
        llm.model_path,
        dtype=args.dtype,
    )
    hf_model.to(args.device).eval()
    direct_output = direct_transformers_generate(
        hf_model,
        tokenizer,
        args.prompt,
        system=args.system,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        enable_thinking=args.enable_thinking,
        device=args.device,
    )

    _, _, success_label = MODEL_FAMILIES[args.family]
    print(f"family={args.family}")
    print(f"model={llm.model_path}")
    print(f"project={project_output!r}")
    print(f"direct={direct_output!r}")
    if project_output != direct_output:
        raise SystemExit(f"{args.family}_compare_failed")
    print(success_label)


if __name__ == "__main__":
    main()
