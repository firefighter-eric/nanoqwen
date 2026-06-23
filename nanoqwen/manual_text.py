from __future__ import annotations

from pathlib import Path

import torch

from nanoqwen.hf_text import count_input_tokens, load_tokenizer, render_chat_prompt


DTYPE_NAMES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def resolve_dtype(name: str, auto_dtype: torch.dtype) -> torch.dtype:
    if name == "auto":
        return auto_dtype
    return DTYPE_NAMES[name]


def first_existing_file(root: str | Path, names: tuple[str, ...]) -> Path:
    root = Path(root)
    for name in names:
        path = root / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"None of these files exist under {root}: {', '.join(names)}")


def top_p_sample(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    logits = logits / temperature
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = sorted_probs.cumsum(dim=-1)
        remove = cumulative_probs > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, torch.finfo(sorted_logits.dtype).min)
        logits = torch.full_like(logits, torch.finfo(logits.dtype).min)
        logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
    return torch.multinomial(torch.softmax(logits.float(), dim=-1), num_samples=1)


@torch.no_grad()
def generate_with_manual_model(
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
    inputs = tokenizer(rendered, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = getattr(inputs, "attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    generated: list[torch.Tensor] = []
    eos_token_id = tokenizer.eos_token_id
    eos_ids = {eos_token_id} if isinstance(eos_token_id, int) else set(eos_token_id or [])

    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=1,
        )
        next_logits = outputs.logits[:, -1, :]
        if temperature > 0:
            next_token = top_p_sample(next_logits, temperature=temperature, top_p=top_p)
        else:
            next_token = next_logits.argmax(dim=-1, keepdim=True)

        generated.append(next_token)
        input_ids = torch.cat((input_ids, next_token), dim=-1)
        if attention_mask is not None:
            attention_mask = torch.cat((attention_mask, torch.ones_like(next_token)), dim=-1)

        if eos_ids and all(int(token) in eos_ids for token in next_token.view(-1)):
            break

    if not generated:
        return ""
    generated_ids = torch.cat(generated, dim=-1)[0]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


class ManualLLM:
    model_cls = None

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        dtype: str = "auto",
        attn_implementation: str = "eager",
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.tokenizer = None
        self.model = None

    def load_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = load_tokenizer(self.model_path)
        return self.tokenizer

    def load_model(self):
        if self.model is None:
            if self.model_cls is None:
                raise TypeError("ManualLLM subclasses must define model_cls")
            model = self.model_cls.from_pretrained(
                self.model_path,
                dtype=self.dtype,
                attn_implementation=self.attn_implementation,
            )
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
        return generate_with_manual_model(
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
