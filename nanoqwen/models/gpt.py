from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .qwen import CausalLMOutput


@dataclass
class GPTConfig:
    """GPT-2/nanoGPT-style decoder-only config."""

    model_type: str = "gpt"
    vocab_size: int = 50304
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    initializer_range: float = 0.02
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")

    @classmethod
    def tiny(cls, vocab_size: int = 257) -> "GPTConfig":
        return cls(
            vocab_size=vocab_size,
            block_size=64,
            n_layer=2,
            n_head=4,
            n_embd=64,
            eos_token_id=vocab_size - 1,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GPTConfig":
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in data.items() if key in allowed})

    @classmethod
    def from_json_file(cls, path: str | Path) -> "GPTConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json_file(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")


class LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, channels = x.size()
        query, key, value = self.c_attn(x).split(self.n_embd, dim=2)
        head_dim = channels // self.n_head

        key = key.view(batch_size, seq_len, self.n_head, head_dim).transpose(1, 2)
        query = query.view(batch_size, seq_len, self.n_head, head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.n_head, head_dim).transpose(1, 2)

        attn = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
        mask = self.bias[:, :, :seq_len, :seq_len].to(torch.bool)
        if attention_mask is not None:
            if attention_mask.shape[-1] != seq_len:
                raise ValueError("attention_mask length must match sequence length")
            key_mask = attention_mask[:, None, None, :].to(torch.bool)
            mask = mask & key_mask
        attn = attn.masked_fill(~mask, torch.finfo(attn.dtype).min)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        y = attn @ value
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPTModel(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "ln_f": LayerNorm(config.n_embd, bias=config.bias),
            }
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.transformer.wte(input_ids)

        _, seq_len, _ = inputs_embeds.size()
        if seq_len > self.config.block_size:
            raise ValueError(
                f"Cannot forward sequence of length {seq_len}; block_size is {self.config.block_size}"
            )
        if position_ids is None:
            position_ids = torch.arange(0, seq_len, dtype=torch.long, device=inputs_embeds.device).unsqueeze(0)

        pos_emb = self.transformer.wpe(position_ids)
        x = self.transformer.drop(inputs_embeds + pos_emb)
        if attention_mask is not None and torch.all(attention_mask == 1):
            attention_mask = None
        for block in self.transformer.h:
            x = block(x, attention_mask=attention_mask)
        return self.transformer.ln_f(x)


class GPTForCausalLM(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.model = GPTModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.model.transformer.wte.weight
        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    param,
                    mean=0.0,
                    std=config.initializer_range / math.sqrt(2 * config.n_layer),
                )

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: object | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
    ) -> CausalLMOutput:
        _ = use_cache
        if past_key_values is not None:
            raise ValueError("GPTForCausalLM does not implement KV cache")

        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )

        loss = None
        if labels is not None:
            logits = self.lm_head(hidden_states)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        else:
            if isinstance(logits_to_keep, int) and logits_to_keep > 0:
                hidden_states = hidden_states[:, -logits_to_keep:, :]
            elif not isinstance(logits_to_keep, int):
                hidden_states = hidden_states[:, logits_to_keep, :]
            logits = self.lm_head(hidden_states)

        return CausalLMOutput(logits=logits, loss=loss, past_key_values=None)


__all__ = [
    "Block",
    "CausalSelfAttention",
    "GPTConfig",
    "GPTForCausalLM",
    "GPTModel",
    "LayerNorm",
    "MLP",
]
