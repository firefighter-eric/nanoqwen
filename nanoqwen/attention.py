from __future__ import annotations

import importlib

import torch
from torch.nn import functional as F

EAGER_ATTENTION = "eager"
SDPA_ATTENTION = "sdpa"
FLASH_ATTENTION_2 = "flash_attention_2"
ATTN_IMPLEMENTATIONS = (EAGER_ATTENTION, SDPA_ATTENTION, FLASH_ATTENTION_2)
ATTN_IMPLEMENTATION_CHOICES = (
    EAGER_ATTENTION,
    SDPA_ATTENTION,
    FLASH_ATTENTION_2,
    "flash-attn2",
    "flash_attn2",
    "flash_attention2",
    "flash2",
    "fa2",
)


def normalize_attn_implementation(name: str | None) -> str:
    if name is None:
        return EAGER_ATTENTION
    normalized = name.lower().replace("-", "_")
    if normalized in {
        "flash",
        "flash2",
        "fa2",
        "flash_attn",
        "flash_attn2",
        "flash_attention",
        "flash_attention2",
    }:
        normalized = FLASH_ATTENTION_2
    if normalized not in ATTN_IMPLEMENTATIONS:
        choices = ", ".join(ATTN_IMPLEMENTATIONS)
        raise ValueError(f"Unsupported attention implementation {name!r}; choose one of: {choices}")
    return normalized


def causal_bool_mask(q_len: int, k_len: int, device: torch.device) -> torch.Tensor:
    if k_len < q_len:
        raise ValueError("key/value sequence length must be >= query sequence length")
    past_len = k_len - q_len
    q_positions = torch.arange(q_len, device=device)[:, None] + past_len
    k_positions = torch.arange(k_len, device=device)[None, :]
    return (k_positions > q_positions)[None, None, :, :]


def attention_bias_from_bool(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    bias = torch.zeros(mask.shape, dtype=dtype, device=mask.device)
    return bias.masked_fill(mask, torch.finfo(dtype).min)


def causal_attention_bias(
    q_len: int,
    k_len: int,
    dtype: torch.dtype,
    device: torch.device,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    mask = causal_bool_mask(q_len, k_len, device)
    if attention_mask is not None:
        if attention_mask.shape[-1] != k_len:
            raise ValueError("attention_mask length must match key/value sequence length")
        padding_mask = attention_mask[:, None, None, :].to(torch.bool)
        mask = mask | ~padding_mask
    return attention_bias_from_bool(mask, dtype)


def eager_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_bias: torch.Tensor | None,
    dropout_p: float,
    scaling: float,
    training: bool,
    is_causal: bool,
) -> torch.Tensor:
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_bias is not None:
        attn_weights = attn_weights + attention_bias
    elif is_causal:
        mask = causal_bool_mask(query.shape[-2], key.shape[-2], query.device)
        attn_weights = attn_weights.masked_fill(mask, torch.finfo(attn_weights.dtype).min)

    attn_weights = F.softmax(attn_weights.float(), dim=-1).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout_p, training=training)
    return torch.matmul(attn_weights, value)


def sdpa_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_bias: torch.Tensor | None,
    dropout_p: float,
    scaling: float,
    is_causal: bool,
) -> torch.Tensor:
    if attention_bias is None and is_causal and query.shape[-2] != key.shape[-2]:
        attention_bias = causal_attention_bias(
            query.shape[-2],
            key.shape[-2],
            dtype=query.dtype,
            device=query.device,
        )
        is_causal = False
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_bias,
        dropout_p=dropout_p,
        is_causal=is_causal and attention_bias is None,
        scale=scaling,
    )


def flash_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float,
    scaling: float,
    is_causal: bool,
) -> torch.Tensor:
    if query.device.type != "cuda":
        raise RuntimeError("flash_attention_2 requires CUDA tensors; use --attn-implementation sdpa on CPU")
    try:
        flash_attn = importlib.import_module("flash_attn")
    except ImportError as exc:
        raise ImportError(
            "flash_attention_2 requires the optional `flash_attn` package. "
            "Install flash-attn for your CUDA/PyTorch build, or use --attn-implementation sdpa."
        ) from exc

    q = query.transpose(1, 2).contiguous()
    k = key.transpose(1, 2).contiguous()
    v = value.transpose(1, 2).contiguous()
    output = flash_attn.flash_attn_func(
        q,
        k,
        v,
        dropout_p=dropout_p,
        softmax_scale=scaling,
        causal=is_causal,
    )
    return output.transpose(1, 2).contiguous()


def attention_forward(
    implementation: str,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_bias: torch.Tensor | None,
    dropout_p: float,
    scaling: float,
    training: bool,
    is_causal: bool = True,
) -> torch.Tensor:
    implementation = normalize_attn_implementation(implementation)
    effective_dropout = dropout_p if training else 0.0

    if implementation == EAGER_ATTENTION:
        return eager_attention_forward(
            query,
            key,
            value,
            attention_bias,
            dropout_p=dropout_p,
            scaling=scaling,
            training=training,
            is_causal=is_causal,
        )
    if implementation == SDPA_ATTENTION:
        return sdpa_attention_forward(
            query,
            key,
            value,
            attention_bias,
            dropout_p=effective_dropout,
            scaling=scaling,
            is_causal=is_causal,
        )

    if attention_bias is not None:
        return sdpa_attention_forward(
            query,
            key,
            value,
            attention_bias,
            dropout_p=effective_dropout,
            scaling=scaling,
            is_causal=is_causal,
        )
    return flash_attention_forward(
        query,
        key,
        value,
        dropout_p=effective_dropout,
        scaling=scaling,
        is_causal=is_causal,
    )
