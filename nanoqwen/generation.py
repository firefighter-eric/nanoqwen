from __future__ import annotations

import torch
from torch.nn import functional as F

from .model import NanoqwenForCausalLM


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    if top_k is not None and top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        cutoff = values[..., -1, None]
        logits = logits.masked_fill(logits < cutoff, torch.finfo(logits.dtype).min)

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        mask = sorted_mask.scatter(dim=-1, index=sorted_indices, src=sorted_mask)
        logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits


@torch.no_grad()
def generate(
    model: NanoqwenForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_token_id: int | list[int] | None = None,
    do_sample: bool = True,
) -> torch.Tensor:
    model.eval()
    generated = input_ids
    past_key_values = None

    if eos_token_id is None:
        eos_token_id = model.config.eos_token_id
    eos_ids = {eos_token_id} if isinstance(eos_token_id, int) else set(eos_token_id or [])

    for _ in range(max_new_tokens):
        step_input = generated if past_key_values is None else generated[:, -1:]
        outputs = model(
            input_ids=step_input,
            past_key_values=past_key_values,
            use_cache=True,
            logits_to_keep=1,
        )
        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values

        if temperature <= 0 or not do_sample:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat((generated, next_token), dim=-1)
        if eos_ids and all(token.item() in eos_ids for token in next_token[:, 0]):
            break
    return generated

