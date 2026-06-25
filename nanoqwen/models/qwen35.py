from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F

from ..attention import attention_forward, causal_attention_bias, normalize_attn_implementation
from ..manual_text import ManualLLM, first_existing_file, resolve_dtype
from .qwen import CausalLMOutput

MODEL_NAME = "qwen35"
REPO_ID = "Qwen/Qwen3.5-0.8B"
DEFAULT_MODEL_PATH = "models/Qwen/Qwen3.5-0.8B"
REQUIRED_FILES = ("config.json", "tokenizer_config.json", "model.safetensors-00001-of-00001.safetensors")


@dataclass
class Qwen35TextConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    layer_types: list[str]
    hidden_act: str = "silu"
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    attention_bias: bool = False
    attn_implementation: str = "eager"
    tie_word_embeddings: bool = True
    pad_token_id: int | None = None
    eos_token_id: int | None = None
    rope_theta: float = 10_000_000.0
    partial_rotary_factor: float = 0.25
    mrope_section: tuple[int, int, int] = (11, 11, 10)
    linear_num_value_heads: int = 16
    linear_num_key_heads: int = 16
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4

    def __post_init__(self) -> None:
        self.attn_implementation = normalize_attn_implementation(self.attn_implementation)

    @classmethod
    def from_json_file(cls, path: str | Path) -> "Qwen35TextConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        values = data.get("text_config", data)
        rope_parameters = values.get("rope_parameters") or {}
        kwargs: dict[str, Any] = {
            "vocab_size": values["vocab_size"],
            "hidden_size": values["hidden_size"],
            "intermediate_size": values["intermediate_size"],
            "num_hidden_layers": values["num_hidden_layers"],
            "num_attention_heads": values["num_attention_heads"],
            "num_key_value_heads": values["num_key_value_heads"],
            "head_dim": values["head_dim"],
            "layer_types": values["layer_types"],
            "hidden_act": values.get("hidden_act", "silu"),
            "max_position_embeddings": values.get("max_position_embeddings", 262144),
            "rms_norm_eps": values.get("rms_norm_eps", 1e-6),
            "attention_dropout": values.get("attention_dropout", 0.0),
            "attention_bias": values.get("attention_bias", False),
            "attn_implementation": normalize_attn_implementation(
                values.get("attn_implementation", "eager")
            ),
            "tie_word_embeddings": values.get("tie_word_embeddings", True),
            "pad_token_id": values.get("pad_token_id"),
            "eos_token_id": values.get("eos_token_id"),
            "rope_theta": rope_parameters.get("rope_theta", 10_000_000.0),
            "partial_rotary_factor": rope_parameters.get("partial_rotary_factor", 0.25),
            "mrope_section": tuple(rope_parameters.get("mrope_section", [11, 11, 10])),
            "linear_num_value_heads": values.get("linear_num_value_heads", 16),
            "linear_num_key_heads": values.get("linear_num_key_heads", 16),
            "linear_key_head_dim": values.get("linear_key_head_dim", 128),
            "linear_value_head_dim": values.get("linear_value_head_dim", 128),
            "linear_conv_kernel_dim": values.get("linear_conv_kernel_dim", 4),
        }
        if kwargs["hidden_act"] != "silu":
            raise ValueError("Only silu is implemented for Qwen3.5")
        return cls(**kwargs)


class Qwen35RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


class Qwen35RMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.float())
        return hidden_states.to(input_dtype)


class Qwen35MLP(nn.Module):
    def __init__(self, config: Qwen35TextConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen35TextRotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen35TextConfig) -> None:
        super().__init__()
        rotary_dim = int(config.head_dim * config.partial_rotary_factor)
        inv_freq = 1.0 / (
            config.rope_theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_scaling = 1.0
        self.mrope_section = config.mrope_section

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        inv_freq = self.inv_freq.to(device=x.device)
        inv_freq_expanded = inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)
        freqs = self.apply_interleaved_mrope(freqs)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def apply_interleaved_mrope(self, freqs: torch.Tensor) -> torch.Tensor:
        freqs_t = freqs[0].clone()
        for dim, offset in enumerate((1, 2), start=1):
            length = self.mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    return torch.cat([q_embed, q_pass], dim=-1), torch.cat([k_embed, k_pass], dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


def causal_mask(
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    return causal_attention_bias(
        seq_len,
        seq_len,
        dtype=dtype,
        device=device,
        attention_mask=attention_mask,
    ).expand(batch_size, 1, seq_len, seq_len)


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    use_qk_l2norm_in_kernel: bool = True,
) -> torch.Tensor:
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    query = query * (1 / (query.shape[-1] ** 0.5))

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = torch.zeros(
        batch_size,
        num_heads,
        k_head_dim,
        v_head_dim,
        dtype=value.dtype,
        device=value.device,
    )
    core_attn_out = torch.zeros_like(value)

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    core_attn_out = core_attn_out.reshape(batch_size, num_heads, -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    return core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)


def apply_mask_to_padding_states(hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(hidden_states.dtype)
    return hidden_states


class Qwen35GatedDeltaNet(nn.Module):
    def __init__(self, config: Qwen35TextConfig, layer_idx: int) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.ones(self.num_v_heads))
        self.norm = Qwen35RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)
        self.in_proj_qkv = nn.Linear(self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape

        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, : mixed_qkv.shape[-1]])
        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        core_attn_out = torch_chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            use_qk_l2norm_in_kernel=True,
        )
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        return self.out_proj(core_attn_out)


class Qwen35Attention(nn.Module):
    def __init__(self, config: Qwen35TextConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.attn_implementation = config.attn_implementation
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim * 2,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = Qwen35RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen35RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(batch_size, seq_len, -1, self.head_dim * 2),
            2,
            dim=-1,
        )
        gate = gate.reshape(batch_size, seq_len, -1)
        query_states = self.q_norm(query_states.view(batch_size, seq_len, -1, self.head_dim)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(batch_size, seq_len, -1, self.head_dim)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(batch_size, seq_len, -1, self.head_dim).transpose(1, 2)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, *position_embeddings)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_output = attention_forward(
            self.attn_implementation,
            query_states,
            key_states,
            value_states,
            attention_bias=attention_mask,
            dropout_p=self.attention_dropout,
            scaling=self.scaling,
            training=self.training,
            is_causal=True,
        )
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output)


class Qwen35DecoderLayer(nn.Module):
    def __init__(self, config: Qwen35TextConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen35GatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen35Attention(config, layer_idx)
        else:
            raise ValueError(f"Unsupported Qwen3.5 layer type: {self.layer_type}")
        self.mlp = Qwen35MLP(config)
        self.input_layernorm = Qwen35RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen35RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states=hidden_states, attention_mask=attention_mask)
        else:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Qwen35TextModel(nn.Module):
    def __init__(self, config: Qwen35TextConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([Qwen35DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = Qwen35RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen35TextRotaryEmbedding(config)

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
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_len, _ = inputs_embeds.shape
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=inputs_embeds.device).view(1, 1, -1)
            position_ids = position_ids.expand(4, batch_size, -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            rotary_position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids
            rotary_position_ids = position_ids

        linear_attention_mask = None
        full_attention_mask = None
        if attention_mask is not None and not torch.all(attention_mask == 1):
            full_attention_mask = causal_mask(
                batch_size,
                seq_len,
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
                attention_mask=attention_mask,
            )
            linear_attention_mask = attention_mask

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, rotary_position_ids)
        _ = text_position_ids  # kept to mirror HF's text position split
        for i, decoder_layer in enumerate(self.layers):
            layer_mask = linear_attention_mask if self.config.layer_types[i] == "linear_attention" else full_attention_mask
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
            )
        return self.norm(hidden_states)


class Qwen35ForCausalLM(nn.Module):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: Qwen35TextConfig) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen35TextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    @classmethod
    def from_pretrained(
        cls,
        model_path: str = DEFAULT_MODEL_PATH,
        dtype: str = "auto",
        attn_implementation: str | None = None,
    ) -> "Qwen35ForCausalLM":
        root = Path(model_path)
        config = Qwen35TextConfig.from_json_file(root / "config.json")
        if attn_implementation is not None:
            config.attn_implementation = normalize_attn_implementation(attn_implementation)
        weights_path = first_existing_file(root, ("model.safetensors", "model.safetensors-00001-of-00001.safetensors"))
        raw_state = load_file(weights_path, device="cpu")

        state: dict[str, torch.Tensor] = {}
        for key, value in raw_state.items():
            if key.startswith("model.language_model."):
                state["model." + key[len("model.language_model.") :]] = value
            elif key.startswith(("model.embed_tokens.", "model.layers.", "model.norm.")):
                state[key] = value
            elif key == "lm_head.weight":
                state[key] = value
        if "lm_head.weight" not in state and "model.embed_tokens.weight" in state:
            state["lm_head.weight"] = state["model.embed_tokens.weight"]

        target_dtype = resolve_dtype(dtype, state["model.embed_tokens.weight"].dtype)
        model = cls(config)
        model.to(dtype=target_dtype)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Qwen3.5 weight load mismatch: missing={missing}, unexpected={unexpected}")
        return model

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
    ) -> CausalLMOutput:
        _ = use_cache
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return CausalLMOutput(logits=logits, loss=loss, past_key_values=None)


class Qwen35LLM(ManualLLM):
    model_cls = Qwen35ForCausalLM

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        device: str = "cpu",
        dtype: str = "auto",
        attn_implementation: str = "eager",
    ) -> None:
        super().__init__(
            model_path=model_path,
            device=device,
            dtype=dtype,
            attn_implementation=attn_implementation,
        )


def missing_files(model_path: str = DEFAULT_MODEL_PATH) -> list[str]:
    root = Path(model_path)
    return [name for name in REQUIRED_FILES if not (root / name).is_file()]


def require_downloaded(model_path: str = DEFAULT_MODEL_PATH) -> None:
    missing = missing_files(model_path)
    if missing:
        raise FileNotFoundError(
            f"{REPO_ID} is missing under {model_path}. "
            "Run: bash runs/download_qwen35_08b.sh. "
            f"Missing files: {', '.join(missing)}"
        )


__all__ = [
    "DEFAULT_MODEL_PATH",
    "MODEL_NAME",
    "Qwen35ForCausalLM",
    "Qwen35LLM",
    "Qwen35TextConfig",
    "Qwen35TextModel",
    "REPO_ID",
    "REQUIRED_FILES",
    "missing_files",
    "require_downloaded",
]
