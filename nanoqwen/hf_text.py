from __future__ import annotations


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


def text_generation_kwargs(max_new_tokens: int, temperature: float = 0.0, top_p: float = 0.9) -> dict:
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p
    return kwargs
