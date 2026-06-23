from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .config import NanoqwenConfig

KVCache = list[tuple[torch.Tensor, torch.Tensor]]


@dataclass
class BaseModelOutput:
    last_hidden_state: torch.Tensor
    past_key_values: KVCache | None = None


@dataclass
class CausalLMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    past_key_values: KVCache | None = None


class QwenRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class QwenRotaryEmbedding(nn.Module):
    def __init__(self, config: NanoqwenConfig) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, config.head_dim, 2, dtype=torch.float32) / config.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        position_ids: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq.to(device=device)
        freqs = torch.einsum("bt,d->btd", position_ids.float(), inv_freq.float())
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


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
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


def causal_mask(
    q_len: int,
    k_len: int,
    device: torch.device,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    past_len = k_len - q_len
    q_positions = torch.arange(q_len, device=device)[:, None] + past_len
    k_positions = torch.arange(k_len, device=device)[None, :]
    mask = k_positions > q_positions
    mask = mask[None, None, :, :]

    if attention_mask is not None:
        if attention_mask.shape[-1] != k_len:
            raise ValueError("attention_mask length must match key/value sequence length")
        padding_mask = attention_mask[:, None, None, :].to(torch.bool)
        mask = mask | ~padding_mask
    return mask


class QwenMLP(nn.Module):
    def __init__(self, config: NanoqwenConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class QwenAttention(nn.Module):
    def __init__(self, config: NanoqwenConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_output_bias,
        )

        if config.use_qk_norm:
            self.q_norm = QwenRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = QwenRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        batch_size, seq_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        key_states = self.k_proj(hidden_states).view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        )
        value_states = self.v_proj(hidden_states).view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        )

        query_states = self.q_norm(query_states).transpose(1, 2)
        key_states = self.k_norm(key_states).transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            key_states = torch.cat((past_key_value[0], key_states), dim=2)
            value_states = torch.cat((past_key_value[1], value_states), dim=2)

        present_key_value = (key_states, value_states) if use_cache else None
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        mask = causal_mask(seq_len, key_states.shape[-2], hidden_states.device, attention_mask)
        attn_weights = attn_weights.masked_fill(mask, torch.finfo(attn_weights.dtype).min)
        attn_weights = F.softmax(attn_weights.float(), dim=-1).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, present_key_value


class QwenDecoderLayer(nn.Module):
    def __init__(self, config: NanoqwenConfig, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = QwenAttention(config, layer_idx)
        self.mlp = QwenMLP(config)
        self.input_layernorm = QwenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = QwenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, present_key_value


class NanoqwenModel(nn.Module):
    def __init__(self, config: NanoqwenConfig) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [QwenDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = QwenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = QwenRotaryEmbedding(config)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: KVCache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
    ) -> BaseModelOutput:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        use_cache = self.config.use_cache if use_cache is None else use_cache
        past_seen_tokens = 0
        if past_key_values:
            past_seen_tokens = past_key_values[0][0].shape[-2]

        if position_ids is None:
            position_ids = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            ).unsqueeze(0)

        if attention_mask is not None and past_seen_tokens and attention_mask.shape[-1] != (
            past_seen_tokens + inputs_embeds.shape[1]
        ):
            prefix = torch.ones(
                attention_mask.shape[0],
                past_seen_tokens,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat((prefix, attention_mask), dim=-1)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(
            position_ids,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        next_cache: KVCache | None = [] if use_cache else None

        for layer_idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[layer_idx] if past_key_values is not None else None
            hidden_states, present_key_value = decoder_layer(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            if use_cache:
                next_cache.append(present_key_value)

        hidden_states = self.norm(hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states, past_key_values=next_cache)


class NanoqwenForCausalLM(nn.Module):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: NanoqwenConfig) -> None:
        super().__init__()
        self.config = config
        self.model = NanoqwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: KVCache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
    ) -> CausalLMOutput:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
        )
        hidden_states = outputs.last_hidden_state

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

        return CausalLMOutput(
            logits=logits,
            loss=loss,
            past_key_values=outputs.past_key_values,
        )
