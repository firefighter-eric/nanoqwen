from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from ..attention import attention_forward, causal_attention_bias, normalize_attn_implementation
from .qwen import CausalLMOutput


@dataclass
class NanoGPTConfig:
    """Karpathy autoresearch GPT-style decoder config."""

    model_type: str = "nanogpt"
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int | None = None
    n_embd: int = 768
    window_pattern: str = "SSSL"
    attn_implementation: str = "sdpa"
    logit_softcap: float = 15.0
    eos_token_id: int | list[int] | None = None

    def __post_init__(self) -> None:
        if self.n_kv_head is None:
            self.n_kv_head = self.n_head
        if self.sequence_len <= 0:
            raise ValueError("sequence_len must be positive")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.n_kv_head <= 0 or self.n_head % self.n_kv_head != 0:
            raise ValueError("n_kv_head must divide n_head")
        if (self.n_embd // self.n_head) % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")
        pattern = self.window_pattern.upper()
        if not pattern or any(char not in "SL" for char in pattern):
            raise ValueError("window_pattern must contain only 'S' and 'L'")
        self.window_pattern = pattern
        self.attn_implementation = normalize_attn_implementation(self.attn_implementation)

    @property
    def block_size(self) -> int:
        return self.sequence_len

    @property
    def max_position_embeddings(self) -> int:
        return self.sequence_len

    @classmethod
    def tiny(cls, vocab_size: int = 257) -> "NanoGPTConfig":
        return cls(
            sequence_len=64,
            vocab_size=vocab_size,
            n_layer=2,
            n_head=4,
            n_kv_head=4,
            n_embd=64,
            window_pattern="L",
            eos_token_id=vocab_size - 1,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NanoGPTConfig":
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in data.items() if key in allowed})

    @classmethod
    def from_json_file(cls, path: str | Path) -> "NanoGPTConfig":
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


def norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


def has_value_embedding(layer_idx: int, n_layer: int) -> bool:
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat((y1, y2), dim=-1)


def causal_window_bias(
    seq_len: int,
    window_size: int,
    dtype: torch.dtype,
    device: torch.device,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if window_size >= seq_len and attention_mask is None:
        return None

    q_idx = torch.arange(seq_len, device=device)[:, None]
    k_idx = torch.arange(seq_len, device=device)[None, :]
    allowed = (k_idx <= q_idx) & (k_idx > q_idx - window_size)
    allowed = allowed[None, None, :, :]
    if attention_mask is not None:
        if attention_mask.shape[-1] != seq_len:
            raise ValueError("attention_mask length must match sequence length")
        allowed = allowed & attention_mask[:, None, None, :].to(torch.bool)
    bias = torch.zeros(allowed.shape, dtype=dtype, device=device)
    return bias.masked_fill(~allowed, torch.finfo(dtype).min)


class NanoGPTAttention(nn.Module):
    def __init__(self, config: NanoGPTConfig, layer_idx: int) -> None:
        super().__init__()
        assert config.n_kv_head is not None
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.scaling = self.head_dim**-0.5
        self.attn_implementation = config.attn_implementation
        self.c_q = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.ve_gate_channels = min(32, config.n_embd)
        self.ve_gate = (
            nn.Linear(self.ve_gate_channels, config.n_kv_head, bias=False)
            if has_value_embedding(layer_idx, config.n_layer)
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        value_embedding: torch.Tensor | None,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        window_size: int,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, channels = x.size()
        query = self.c_q(x).view(batch_size, seq_len, self.n_head, self.head_dim)
        key = self.c_k(x).view(batch_size, seq_len, self.n_kv_head, self.head_dim)
        value = self.c_v(x).view(batch_size, seq_len, self.n_kv_head, self.head_dim)

        if value_embedding is not None:
            value_embedding = value_embedding.view(batch_size, seq_len, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., : self.ve_gate_channels]))
            value = value + gate.unsqueeze(-1) * value_embedding

        cos, sin = cos_sin
        query = norm(apply_rotary_emb(query, cos, sin))
        key = norm(apply_rotary_emb(key, cos, sin))
        if self.n_kv_head != self.n_head:
            repeats = self.n_head // self.n_kv_head
            key = key.repeat_interleave(repeats, dim=2)
            value = value.repeat_interleave(repeats, dim=2)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attention_bias = causal_window_bias(
            seq_len,
            window_size,
            dtype=query.dtype,
            device=query.device,
            attention_mask=attention_mask,
        )
        if attention_bias is None and attention_mask is not None:
            attention_bias = causal_attention_bias(
                seq_len,
                seq_len,
                dtype=query.dtype,
                device=query.device,
                attention_mask=attention_mask,
            )

        y = attention_forward(
            self.attn_implementation,
            query,
            key,
            value,
            attention_bias=attention_bias,
            dropout_p=0.0,
            scaling=self.scaling,
            training=self.training,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        return self.c_proj(y)


class NanoGPTMLP(nn.Module):
    def __init__(self, config: NanoGPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.relu(self.c_fc(x)).square())


class NanoGPTBlock(nn.Module):
    def __init__(self, config: NanoGPTConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn = NanoGPTAttention(config, layer_idx)
        self.mlp = NanoGPTMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        value_embedding: torch.Tensor | None,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        window_size: int,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(norm(x), value_embedding, cos_sin, window_size, attention_mask=attention_mask)
        return x + self.mlp(norm(x))


class NanoGPTModel(nn.Module):
    def __init__(self, config: NanoGPTConfig) -> None:
        super().__init__()
        assert config.n_kv_head is not None
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "h": nn.ModuleList([NanoGPTBlock(config, i) for i in range(config.n_layer)]),
            }
        )
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict(
            {
                str(i): nn.Embedding(config.vocab_size, kv_dim)
                for i in range(config.n_layer)
                if has_value_embedding(i, config.n_layer)
            }
        )
        cos, sin = self._precompute_rotary_embeddings(config.sequence_len * 10, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _precompute_rotary_embeddings(
        self,
        seq_len: int,
        head_dim: int,
        base: int = 10000,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos = freqs.cos()[None, :, None, :]
        sin = freqs.sin()[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config: NanoGPTConfig) -> list[int]:
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": long_window, "S": short_window}
        window_sizes = [char_to_window[config.window_pattern[i % len(config.window_pattern)]] for i in range(config.n_layer)]
        window_sizes[-1] = long_window
        return window_sizes

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _, seq_len = input_ids.shape
        if seq_len > self.cos.size(1):
            raise ValueError(f"Cannot forward sequence of length {seq_len}; rotary cache has {self.cos.size(1)}")
        cos_sin = self.cos[:, :seq_len].to(dtype=self.transformer.wte.weight.dtype), self.sin[:, :seq_len].to(
            dtype=self.transformer.wte.weight.dtype
        )

        x = norm(self.transformer.wte(input_ids))
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            value_embedding = self.value_embeds[str(i)](input_ids) if str(i) in self.value_embeds else None
            x = block(x, value_embedding, cos_sin, self.window_sizes[i], attention_mask=attention_mask)
        return norm(x)


class NanoGPTForCausalLM(nn.Module):
    def __init__(self, config: NanoGPTConfig) -> None:
        super().__init__()
        self.config = config
        self.model = NanoGPTModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.init_weights()

    @torch.no_grad()
    def init_weights(self) -> None:
        torch.nn.init.normal_(self.model.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        scale = math.sqrt(3.0) * self.config.n_embd**-0.5
        for block in self.model.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -scale, scale)
            torch.nn.init.uniform_(block.attn.c_k.weight, -scale, scale)
            torch.nn.init.uniform_(block.attn.c_v.weight, -scale, scale)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -scale, scale)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        self.model.resid_lambdas.fill_(1.0)
        self.model.x0_lambdas.fill_(0.1)
        for value_embedding in self.model.value_embeds.values():
            torch.nn.init.uniform_(value_embedding.weight, -scale, scale)

    def prepare_autoresearch_training_dtype(self, param_dtype: torch.dtype) -> None:
        self.model.transformer.wte.to(dtype=param_dtype)
        for value_embedding in self.model.value_embeds.values():
            value_embedding.to(dtype=param_dtype)

    def setup_autoresearch_optimizer(
        self,
        *,
        unembedding_lr: float = 0.006,
        embedding_lr: float = 0.6,
        matrix_lr: float = 0.04,
        scalar_lr: float = 0.5,
        weight_decay: float = 0.2,
        adam_betas: tuple[float, float] = (0.8, 0.95),
        matrix_optimizer: str = "muon",
        compile_steps: bool = False,
    ) -> torch.optim.Optimizer:
        from ..optim import MuonAdamW

        if matrix_optimizer not in {"muon", "adamw"}:
            raise ValueError("matrix_optimizer must be 'muon' or 'adamw'")

        model_dim = self.config.n_embd
        matrix_params = list(self.model.transformer.h.parameters())
        value_embedding_params = list(self.model.value_embeds.parameters())
        embedding_params = list(self.model.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.model.resid_lambdas]
        x0_params = [self.model.x0_lambdas]
        expected_params = (
            len(matrix_params)
            + len(embedding_params)
            + len(lm_head_params)
            + len(value_embedding_params)
            + len(resid_params)
            + len(x0_params)
        )
        if len(list(self.parameters())) != expected_params:
            raise RuntimeError("NanoGPT autoresearch optimizer parameter grouping is incomplete")

        dmodel_lr_scale = (model_dim / 768) ** -0.5
        param_groups: list[dict[str, object]] = [
            {
                "kind": "adamw",
                "role": "lm_head",
                "params": lm_head_params,
                "lr": unembedding_lr * dmodel_lr_scale,
                "betas": adam_betas,
                "eps": 1e-10,
                "weight_decay": 0.0,
            },
            {
                "kind": "adamw",
                "role": "embedding",
                "params": embedding_params,
                "lr": embedding_lr * dmodel_lr_scale,
                "betas": adam_betas,
                "eps": 1e-10,
                "weight_decay": 0.0,
            },
            {
                "kind": "adamw",
                "role": "value_embedding",
                "params": value_embedding_params,
                "lr": embedding_lr * dmodel_lr_scale,
                "betas": adam_betas,
                "eps": 1e-10,
                "weight_decay": 0.0,
            },
            {
                "kind": "adamw",
                "role": "resid_scalar",
                "params": resid_params,
                "lr": scalar_lr * 0.01,
                "betas": adam_betas,
                "eps": 1e-10,
                "weight_decay": 0.0,
            },
            {
                "kind": "adamw",
                "role": "x0_scalar",
                "params": x0_params,
                "lr": scalar_lr,
                "betas": (0.96, 0.95),
                "eps": 1e-10,
                "weight_decay": 0.0,
            },
        ]
        for shape in sorted({p.shape for p in matrix_params}, key=tuple):
            group_params = [p for p in matrix_params if p.shape == shape]
            if matrix_optimizer == "muon":
                param_groups.append(
                    {
                        "kind": "muon",
                        "role": "matrix",
                        "params": group_params,
                        "lr": matrix_lr,
                        "momentum": 0.95,
                        "ns_steps": 5,
                        "beta2": 0.95,
                        "weight_decay": weight_decay,
                    }
                )
            else:
                param_groups.append(
                    {
                        "kind": "adamw",
                        "role": "matrix",
                        "params": group_params,
                        "lr": matrix_lr,
                        "betas": adam_betas,
                        "eps": 1e-10,
                        "weight_decay": weight_decay,
                    }
                )
        optimizer = MuonAdamW(param_groups, compile_steps=compile_steps)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def num_scaling_params(self) -> dict[str, int]:
        wte = sum(p.numel() for p in self.model.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.model.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.model.transformer.h.parameters())
        scalars = self.model.resid_lambdas.numel() + self.model.x0_lambdas.numel()
        return {
            "wte": wte,
            "value_embeds": value_embeds,
            "lm_head": lm_head,
            "transformer_matrices": transformer_matrices,
            "scalars": scalars,
            "total": wte + value_embeds + lm_head + transformer_matrices + scalars,
        }

    def estimate_flops(self) -> int:
        params = sum(p.numel() for p in self.parameters())
        value_embeds = sum(p.numel() for p in self.model.value_embeds.parameters())
        excluded = self.model.transformer.wte.weight.numel() + value_embeds
        excluded += self.model.resid_lambdas.numel() + self.model.x0_lambdas.numel()
        head_dim = self.config.n_embd // self.config.n_head
        attn_flops = sum(12 * self.config.n_head * head_dim * min(window, self.config.sequence_len) for window in self.model.window_sizes)
        return 6 * (params - excluded) + attn_flops

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
        _ = position_ids, use_cache
        if past_key_values is not None:
            raise ValueError("NanoGPTForCausalLM does not implement KV cache")
        if inputs_embeds is not None:
            raise ValueError("NanoGPTForCausalLM requires input_ids because it uses token value embeddings")
        if input_ids is None:
            raise ValueError("input_ids is required")

        hidden_states = self.model(input_ids=input_ids, attention_mask=attention_mask)
        if isinstance(logits_to_keep, int) and logits_to_keep > 0:
            kept_hidden_states = hidden_states[:, -logits_to_keep:, :]
        elif not isinstance(logits_to_keep, int):
            kept_hidden_states = hidden_states[:, logits_to_keep, :]
        else:
            kept_hidden_states = hidden_states
        logits = self.lm_head(kept_hidden_states).float()
        if self.config.logit_softcap > 0:
            logits = self.config.logit_softcap * torch.tanh(logits / self.config.logit_softcap)

        loss = None
        if labels is not None:
            full_logits = logits if kept_hidden_states is hidden_states else self.lm_head(hidden_states).float()
            if self.config.logit_softcap > 0 and full_logits is not logits:
                full_logits = self.config.logit_softcap * torch.tanh(full_logits / self.config.logit_softcap)
            shift_logits = full_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutput(logits=logits, loss=loss, past_key_values=None)


__all__ = [
    "NanoGPTAttention",
    "NanoGPTBlock",
    "NanoGPTConfig",
    "NanoGPTForCausalLM",
    "NanoGPTMLP",
    "NanoGPTModel",
    "apply_rotary_emb",
    "has_value_embedding",
    "norm",
]
