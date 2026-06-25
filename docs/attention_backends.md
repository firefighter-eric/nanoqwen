# Attention Backends

本文档记录 nanoqwen 当前支持的 attention backend、安装组合、使用方式、验证范围和
forward benchmark。

## 支持范围

通用 `nanoqwen/models/qwen.py`、`nanoqwen/models/qwen3.py` 和 Qwen3.5 的
`full_attention` 层支持以下 backend：

```text
eager
sdpa
flash_attention_2
flash-attn2
flash_attn2
flash_attention2
flash2
fa2
```

内部标准名是 `flash_attention_2`。其他 FlashAttention-2 写法只是 CLI/API
别名，都会归一化为 `flash_attention_2`。

Qwen3.5 的 `linear_attention` 层不使用 SDPA 或 FlashAttention-2，仍然是手写
GatedDeltaNet。也就是说：

- Qwen3：所有 attention 层都可以切 `eager` / `sdpa` / `flash_attention_2`。
- Qwen3.5：只有 `full_attention` 层切 backend，`linear_attention` 层保持
  GatedDeltaNet。

## Backend 语义

`eager` 是默认实现，使用显式 PyTorch matmul：

```text
q @ k.T -> causal/padding mask -> softmax -> @ v
```

它最适合阅读、调试和做 parity 基线。

`sdpa` 使用 PyTorch 的：

```python
torch.nn.functional.scaled_dot_product_attention
```

`flash_attention_2` 使用外部 `flash_attn` 包的 `flash_attn_func`。无 padding 的
prefill 和 KV cache decode 可以走 FlashAttention-2；带 padding mask 的
full-attention 场景会自动走 SDPA 安全路径。

## 当前验证环境

```text
GPU: NVIDIA GeForce RTX 5080, 16GB
Python: 3.12.8
torch: 2.8.0+cu129
torchvision: 0.23.0+cu129
CUDA runtime: 12.9
flash-attn: 2.8.3.post1
triton: 3.4.0
```

FlashAttention-2 使用官方 release wheel：

```text
flash_attn-2.8.3.post1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

## 使用方式

真实 Qwen 文本生成：

```bash
uv run python scripts/qwen_llm_generate.py \
  --family qwen3 \
  --attn-implementation sdpa \
  --prompt "你好"
```

FlashAttention-2：

```bash
uv run python scripts/qwen_llm_generate.py \
  --family qwen3 \
  --device cuda \
  --dtype bfloat16 \
  --attn-implementation flash-attn2 \
  --prompt "你好"
```

tiny 训练也可以指定 backend：

```bash
uv run python scripts/train.py --attn-implementation sdpa
```

## 与 HF 的输出对照

当前环境下，以下确定性生成对照均通过：

```text
qwen3  eager       vs HF: OK
qwen3  sdpa        vs HF: OK
qwen3  flash-attn2 vs HF: OK
qwen35 eager       vs HF: OK
qwen35 sdpa        vs HF: OK
qwen35 flash-attn2 vs HF: OK
```

对照命令示例：

```bash
uv run python scripts/qwen_llm_compare.py \
  --family qwen3 \
  --attn-implementation flash-attn2 \
  --device cuda \
  --temperature 0
```

注意：Qwen3.5 的 Transformers 对照当前提示 fast linear-attention 依赖未安装，
因此 HF 侧使用的是 torch fallback；这仍然是当前本地环境下可复现的正确对照。

## Forward Benchmark

测试方式：

- 模型：本地真实 HF safetensors 权重
- batch size：1
- 输入：随机 token ids
- `dtype=auto`
- `torch.inference_mode()`
- `use_cache=False`
- `logits_to_keep=1`
- 无 padding mask
- CUDA event 计时，包含完整 model forward，不只计 attention kernel

### Qwen3-0.6B

| backend | seq len | ms / forward | tokens/s | peak CUDA memory |
|---|---:|---:|---:|---:|
| eager | 1024 | 40.938 | 25,013 | 1.29 GB |
| sdpa | 1024 | 22.648 | 45,214 | 1.15 GB |
| flash-attn2 | 1024 | 24.238 | 42,248 | 1.15 GB |
| eager | 2048 | 124.697 | 16,424 | 1.78 GB |
| sdpa | 2048 | 40.956 | 50,005 | 1.18 GB |
| flash-attn2 | 2048 | 43.017 | 47,610 | 1.19 GB |
| eager | 4096 | 419.650 | 9,760 | 3.71 GB |
| sdpa | 4096 | 84.872 | 48,261 | 1.24 GB |
| flash-attn2 | 4096 | 87.167 | 46,990 | 1.25 GB |

结论：Qwen3-0.6B 的 full attention 层很多，SDPA/FlashAttention-2 在长上下文
收益明显。当前 RTX 5080 + torch 2.8.0 环境下，SDPA 略快于 FlashAttention-2。

### Qwen3.5-0.8B

| backend | seq len | ms / forward | tokens/s | peak CUDA memory |
|---|---:|---:|---:|---:|
| eager | 1024 | 110.882 | 9,235 | 1.53 GB |
| sdpa | 1024 | 107.861 | 9,494 | 1.53 GB |
| flash-attn2 | 1024 | 109.374 | 9,362 | 1.53 GB |
| eager | 2048 | 181.003 | 11,315 | 1.79 GB |
| sdpa | 2048 | 169.512 | 12,082 | 1.63 GB |
| flash-attn2 | 2048 | 169.389 | 12,090 | 1.63 GB |
| eager | 4096 | 333.263 | 12,291 | 2.79 GB |
| sdpa | 4096 | 296.144 | 13,831 | 1.82 GB |
| flash-attn2 | 4096 | 312.023 | 13,127 | 1.82 GB |

结论：Qwen3.5-0.8B 是 hybrid 架构，大部分层是 GatedDeltaNet
`linear_attention`，只有少数 `full_attention` 层受 SDPA/FlashAttention-2 影响。
因此 backend 切换收益明显小于 Qwen3，但在 2K/4K 下仍能降低显存并提升速度。

## 维护注意

- 保留 `eager`，它是可读基线和调试路径。
- 新 backend 必须和 `eager` 做数值/生成对照。
- Qwen3.5 的 `linear_attention` 不要替换成 SDPA/FlashAttention-2。
- 只有无 padding mask 的 full-attention 路径直接使用 FlashAttention-2；有
  padding mask 时走 SDPA。
