from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import torch


def is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"}


def image_content_item(image: str) -> dict:
    if is_url(image):
        return {"type": "image", "url": image}
    return {"type": "image", "image": str(Path(image))}


def build_messages(prompt: str, images: list[str] | None = None, system: str | None = None) -> list[dict]:
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})

    content: list[dict] = []
    for image in images or []:
        content.append(image_content_item(image))
    content.append({"type": "text", "text": prompt})
    messages.append({"role": "user", "content": content})
    return messages


def decode_generated(processor, outputs, input_ids) -> str:
    generated_ids = outputs[0][input_ids.shape[-1] :]
    return processor.decode(generated_ids, skip_special_tokens=True).strip()


def resolve_torch_dtype(name: str):
    if name == "auto":
        return "auto"
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[name]


def processor_inputs(
    processor,
    messages: list[dict],
    enable_thinking: bool = False,
    return_tensors: str = "pt",
):
    return processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors=return_tensors,
        enable_thinking=enable_thinking,
    )


def generation_kwargs(max_new_tokens: int, temperature: float = 0.0, top_p: float = 0.9) -> dict:
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p
    return kwargs


@torch.no_grad()
def generate_with_transformers(
    model,
    processor,
    messages: list[dict],
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    top_p: float = 0.9,
    enable_thinking: bool = False,
    device: str = "cpu",
) -> str:
    inputs = processor_inputs(
        processor,
        messages,
        enable_thinking=enable_thinking,
        return_tensors="pt",
    ).to(device)
    outputs = model.generate(
        **inputs,
        **generation_kwargs(max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p),
    )
    return decode_generated(processor, outputs, inputs["input_ids"])
