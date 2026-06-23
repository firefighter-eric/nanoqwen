from __future__ import annotations

import torch


def resolve_torch_dtype(name: str):
    if name == "auto":
        return "auto"
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[name]


def build_chat_messages(prompt: str, system: str | None = None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def render_chat_prompt(
    tokenizer,
    prompt: str,
    system: str | None = None,
    enable_thinking: bool = False,
) -> str:
    messages = build_chat_messages(prompt, system=system)
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
    }
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError as exc:
        if "enable_thinking" not in str(exc):
            raise
        kwargs.pop("enable_thinking")
        return tokenizer.apply_chat_template(messages, **kwargs)


def count_input_tokens(
    tokenizer,
    prompt: str,
    system: str | None = None,
    enable_thinking: bool = False,
) -> int:
    rendered = render_chat_prompt(
        tokenizer,
        prompt,
        system=system,
        enable_thinking=enable_thinking,
    )
    inputs = tokenizer(rendered, return_tensors="pt")
    return int(inputs.input_ids.shape[-1])


def load_tokenizer(model_path: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("Install with `uv sync --extra dev`.") from exc
    return AutoTokenizer.from_pretrained(model_path)


def load_causal_lm(model_path: str, dtype: str = "auto"):
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise ImportError("Install with `uv sync --extra dev`.") from exc
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=resolve_torch_dtype(dtype),
    )


class HFTextCausalLM:
    def __init__(self, model_path: str, device: str = "cpu", dtype: str = "auto") -> None:
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.tokenizer = None
        self.model = None

    def load_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = load_tokenizer(self.model_path)
        return self.tokenizer

    def load_model(self):
        if self.model is None:
            model = load_causal_lm(self.model_path, dtype=self.dtype)
            self.model = model.to(self.device).eval()
        return self.model

    def count_input_tokens(
        self,
        prompt: str,
        system: str | None = None,
        enable_thinking: bool = False,
    ) -> int:
        return count_input_tokens(
            self.load_tokenizer(),
            prompt,
            system=system,
            enable_thinking=enable_thinking,
        )

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_p: float = 0.9,
        enable_thinking: bool = False,
    ) -> str:
        return generate_with_causal_lm(
            self.load_model(),
            self.load_tokenizer(),
            prompt,
            system=system,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            enable_thinking=enable_thinking,
            device=self.device,
        )


def render_loaded_chat_prompt(
    model: HFTextCausalLM,
    prompt: str,
    system: str | None = None,
    enable_thinking: bool = False,
) -> str:
    return render_chat_prompt(
        model.load_tokenizer(),
        prompt,
        system=system,
        enable_thinking=enable_thinking,
    )


@torch.no_grad()
def generate_with_causal_lm(
    model,
    tokenizer,
    prompt: str,
    system: str | None = None,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    top_p: float = 0.9,
    enable_thinking: bool = False,
    device: str = "cpu",
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


def text_generation_kwargs(max_new_tokens: int, temperature: float = 0.0, top_p: float = 0.9) -> dict:
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p
    return kwargs
