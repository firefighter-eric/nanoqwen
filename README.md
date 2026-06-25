# nanoqwen

nanoqwen 是一个小型、可读的 Qwen 风格语言模型项目。项目结构参考
`nanoGPT` 的简洁训练路径，也吸收了 `nanochat` 的端到端实验组织方式；真实
Qwen 权重加载和命名则尽量贴近 Hugging Face Transformers，方便做权重对齐和
生成结果校验。

这个项目的重点不是替代 `transformers`，而是提供一个易读的实验仓库：

- 用 `nanoqwen/models/qwen.py` 训练和调试小型 Qwen-like decoder-only 模型；
- 用 `nanoqwen/models/gpt.py` 对比 GPT-2/nanoGPT 风格 decoder-only 模型；
- 用 `nanoqwen/models/nanogpt.py` 对比 `../autoresearch` 的 NanoGPT 模型；
- 用手写 PyTorch 实现加载本地 Qwen3/Qwen3.5 文本 LLM 权重；
- 用 Transformers 只做 tokenizer、chat template、权重来源和结果对照。

## 当前目标

项目现在支持几条模型路径：

- `Qwen3-0.6B`：手写 text-only causal LM，代码在
  `nanoqwen/models/qwen3.py`。
- `Qwen3.5-0.8B`：手写 text-only causal LM，代码在
  `nanoqwen/models/qwen35.py`。
- `GPT`：GPT-2/nanoGPT 风格 text-only causal LM，代码在
  `nanoqwen/models/gpt.py`，用于结构和训练实验对比。
- `NanoGPT`：`../autoresearch/train.py` 里的 GPT 变体，代码在
  `nanoqwen/models/nanogpt.py`，用于同协议预训练架构对比。

注意：Qwen3.5 的公开 checkpoint 里包含多模态和 MTP 相关权重，但 nanoqwen
当前只实现并加载文本 LLM 部分，不实现视觉、多模态输入或 MTP。

## 设计原则

- **可读优先**：模型、训练、评估、采样脚本尽量保持 plain PyTorch。
- **命名兼容**：权重名保持类似 `model.layers.0.self_attn.q_proj`，方便和 HF
  checkpoint 对齐。
- **手写模型路径**：`models/qwen3.py` 和 `models/qwen35.py` 不包装
  `AutoModelForCausalLM`；HF 模型只在 compare 脚本里作为对照。
- **可切换 attention backend**：默认 `eager` 便于阅读和对齐，也支持 PyTorch
  SDPA；安装 `flash_attn` 后可在无 padding 的 full-attention 路径使用
  FlashAttention。
- **小测试可跑**：默认测试使用 byte tokenizer 和 tiny 模型，不需要下载真实
  Qwen 权重。
- **结果可验证**：下载真实权重后，可以用 compare smoke 验证手写模型和
  Transformers 在确定性生成下输出一致。

## 项目结构

```text
nanoqwen/
  config.py        # Qwen-like 配置 dataclass
  models/
    qwen.py        # 可训练的小型 Qwen-like decoder-only causal LM
    qwen3.py       # 手写 Qwen3-0.6B text-only 模型加载与推理
    qwen35.py      # 手写 Qwen3.5-0.8B text-only 模型加载与推理
    gpt.py         # GPT-2/nanoGPT 风格 decoder-only causal LM
    nanogpt.py     # autoresearch NanoGPT 变体
  model.py         # 兼容导出：from nanoqwen.models.qwen import *
  qwen3_model.py   # 兼容导出：from nanoqwen.models.qwen3 import *
  qwen35_model.py  # 兼容导出：from nanoqwen.models.qwen35 import *
  gpt_model.py     # 兼容导出：from nanoqwen.models.gpt import *
  nanogpt_model.py # 兼容导出：from nanoqwen.models.nanogpt import *
  manual_text.py   # 手写模型共用的 tokenizer、dtype、generation 工具
  hf_text.py       # chat template、tokenizer 加载、HF 对照生成参数
  hf_multimodal.py # 可选 HF 多模态 prompt 辅助；不是核心 LLM 实现
  tokenizer.py     # byte tokenizer 和可选 HF tokenizer wrapper
  generation.py    # greedy/top-k/top-p 采样工具
  data.py          # packed token dataset
  sft.py           # assistant-masked SFT 数据集
  dpo.py           # preference dataset 和 DPO loss
  eval.py          # loss/perplexity、prompt exact-match 评估
  checkpoint.py    # native checkpoint 保存/加载和 HF-style import/export
  report.py        # checkpoint 报告
  ui.html          # scripts/chat_web.py 使用的静态聊天页面

scripts/
  train.py             # nanoGPT 风格训练循环
  sample.py            # 从 native checkpoint 采样
  eval.py              # checkpoint 评估
  report.py            # checkpoint 报告生成
  chat_cli.py          # 本地终端聊天
  chat_web.py          # 本地浏览器聊天 UI
  import_hf.py         # 导入兼容的 HF Qwen checkpoint
  export_hf.py         # 导出 native checkpoint 为 HF-style 格式
  hf_smoke.py          # HF config/tokenizer/可选权重导入 smoke
  qwen_llm_generate.py # family-selecting 手写 Qwen3/Qwen3.5 文本生成
  qwen_llm_compare.py  # 手写模型 vs 直接 Transformers 生成对照
  qwen3_llm_generate.py
  qwen3_compare.py
  qwen35_llm_generate.py
  qwen35_compare.py
  qwen35_generate.py   # 可选 HF Qwen3.5 multimodal 生成，不是手写路径
  sft.py
  dpo.py

runs/
  check.sh
  download_qwen3_06b.sh
  download_qwen35_08b.sh
  qwen3_dry_smoke.sh
  qwen3_text_smoke.sh
  qwen3_compare_smoke.sh
  qwen35_dry_smoke.sh
  qwen35_text_smoke.sh
  qwen35_compare_smoke.sh
  smoke.sh
  gpt_smoke.sh
  gpt_sft_smoke.sh
  eval_smoke.sh
  sft_smoke.sh
  dpo_smoke.sh
  report_smoke.sh
  chat_web.sh
  hf_local_smoke.sh

examples/
  sft_tiny.jsonl
  preferences_tiny.jsonl
  prompts_tiny.jsonl
  multiple_choice_tiny.jsonl

tests/
```

## 安装

项目要求 Python `>=3.12`。当前本地开发/验证环境使用的是 Python `3.12.8`。

建议使用 `uv`：

```bash
uv sync --extra dev
```

只做 text-only Qwen3/Qwen3.5 手写模型推理和对照，`dev` extra 就够了。

如果要准备 parquet 文本数据集，例如 `climbmix`，再安装：

```bash
uv sync --extra dev --extra data
```

如果要运行可选的多模态 HF 脚本，例如 `scripts/qwen35_generate.py`，再安装：

```bash
uv sync --extra dev --extra vision
```

## 基础检查

运行单元测试：

```bash
uv run pytest
```

运行默认检查，也就是编译全部 Python 文件并执行 pytest：

```bash
bash runs/check.sh
```

运行更完整的本地 smoke，包括 tiny 训练、评估、SFT、DPO、HF 本地导入导出和
临时 web health check：

```bash
bash runs/check.sh --smoke
```

如果本地已经下载真实 Qwen 权重，可以加上：

```bash
bash runs/check.sh --qwen3
bash runs/check.sh --qwen35
bash runs/check.sh --qwen
```

## 下载真实模型

真实模型权重放在 `models/` 下，不提交到 git。

下载 Qwen3-0.6B：

```bash
bash runs/download_qwen3_06b.sh
```

默认路径：

```text
models/Qwen/Qwen3-0.6B
```

下载 Qwen3.5-0.8B：

```bash
bash runs/download_qwen35_08b.sh
```

默认路径：

```text
models/Qwen/Qwen3.5-0.8B
```

## 手写 Qwen 推理

先做 tokenizer/config dry-run，不加载大权重：

```bash
bash runs/qwen3_dry_smoke.sh
bash runs/qwen35_dry_smoke.sh
```

运行短文本生成：

```bash
bash runs/qwen3_text_smoke.sh
bash runs/qwen35_text_smoke.sh
```

也可以直接使用 family-selecting 入口：

```bash
uv run python scripts/qwen_llm_generate.py --family qwen3 --prompt "你好"
uv run python scripts/qwen_llm_generate.py --family qwen35 --prompt "你好"
```

指定设备和 dtype：

```bash
uv run python scripts/qwen_llm_generate.py \
  --family qwen3 \
  --device cuda \
  --dtype bfloat16 \
  --attn-implementation sdpa \
  --prompt "用一句话介绍你自己"
```

## Attention Backend

详细说明、验证范围和 1024/2K/4K forward benchmark 见：

[docs/attention_backends.md](docs/attention_backends.md)

常用 backend：

```bash
uv run python scripts/qwen_llm_generate.py \
  --family qwen3 \
  --attn-implementation sdpa \
  --prompt "你好"
```

```bash
uv run python scripts/qwen_llm_generate.py \
  --family qwen3 \
  --device cuda \
  --dtype bfloat16 \
  --attn-implementation flash-attn2 \
  --prompt "你好"
```

Qwen3.5 只有 `full_attention` 层会切换到 SDPA/FlashAttention；
`linear_attention` 层仍然使用手写 GatedDeltaNet。

## 如何确认输出一致

对照脚本会同时运行：

1. nanoqwen 的手写模型；
2. 同一路径下由 Transformers 直接加载的 HF 模型。

然后比较两边生成出的文本：

```bash
bash runs/qwen3_compare_smoke.sh
bash runs/qwen35_compare_smoke.sh
```

或者手动运行：

```bash
uv run python scripts/qwen_llm_compare.py \
  --family qwen3 \
  --prompt "Say hello in one short sentence." \
  --max-new-tokens 8 \
  --temperature 0
```

当前“一致”主要指确定性生成，也就是 `temperature=0` 的 greedy 输出一致。
采样模式会涉及随机数状态，不适合作为严格逐字节一致校验。

指定 backend 对照：

```bash
uv run python scripts/qwen_llm_compare.py \
  --family qwen3 \
  --attn-implementation sdpa \
  --temperature 0
```

## 模型文件的区别

`nanoqwen/models/qwen.py` 是项目里的通用小模型实现，用于 tiny 训练、SFT、
DPO、采样、HF-style import/export 等实验。它是 Qwen-like 结构，但不是专门
绑定某个真实公开 checkpoint。

`nanoqwen/models/qwen3.py` 是 Qwen3-0.6B 的手写加载路径。它复用
`NanoqwenForCausalLM` 的 decoder-only 实现，读取本地 Qwen3 config 和
`model.safetensors`，并要求权重名对齐。

`nanoqwen/models/qwen35.py` 是 Qwen3.5-0.8B 的手写文本模型。它单独实现了
Qwen3.5 text stack，包括 full attention、linear attention layer schedule、
GatedDeltaNet、MRoPE 和对应 RMSNorm 变体。它只读取 checkpoint 里的
`model.language_model.*` 文本权重。

`nanoqwen/models/gpt.py` 是 GPT-2/nanoGPT 风格的对比模型。它使用 learned
position embedding、pre-LN Transformer block、causal self-attention、GELU MLP
和 tied token/lm head 权重。

根目录下的 `model.py`、`qwen3_model.py`、`qwen35_model.py` 和 `gpt_model.py`
保留为兼容导出，新增模型定义优先放到 `nanoqwen/models/`。

## 训练和评估

运行 tiny CPU 训练：

```bash
bash runs/smoke.sh
```

运行 GPT-2/nanoGPT 风格 tiny CPU 训练：

```bash
bash runs/gpt_smoke.sh
```

等价的显式入口：

```bash
uv run python scripts/train.py \
  --model gpt \
  --out-dir out/gpt-smoke \
  --steps 4 \
  --batch-size 2 \
  --block-size 32 \
  --hidden-size 64 \
  --layers 2 \
  --heads 4 \
  --device cpu
```

运行 GPT / autoresearch NanoGPT / Qwen-like 预训练架构对比 smoke。这个 smoke
走轻量本地 text 路径，用于快速检查入口；完整 suite 会固定 tokenizer、parquet
数据流、context、batch、optimizer、seed 和 step/time budget；Qwen 配置按本地
`Qwen/Qwen3-0.6B` 的结构比例缩小，而不是硬凑到 GPT 参数量：

```bash
bash autoresearch/pretrain_arch_compare/run_smoke.sh
```

默认较完整配置使用 `vocab=8192, context=2048, layers=8, hidden=512, heads=4,
head_dim=128`；GPT 约 30,462,976 参数，autoresearch NanoGPT 约 50,332,176
参数，Qwen-like 使用
`intermediate_size=1536, kv_heads=2, tie_word_embeddings=true, rope_theta=1e6`，
约 29,370,880 参数。完整对比使用 `../autoresearch` 的 tokenizer、climbmix
parquet 文档流、per-doc BOS、best-fit packing 和 `40*524288` token BPB 测评。
先准备 autoresearch cache：

```bash
cd ../autoresearch
uv run python prepare.py
cd ../nanoqwen
```

然后运行：

```bash
bash autoresearch/pretrain_arch_compare/run.sh
```

完整对比会直接读取 autoresearch 产出的 `tokenizer.pkl` 和 `token_bytes.pt`；
旧的 Hugging Face ByteLevel BPE tokenizer 只用于本项目独立本地实验，不用于这组
架构对比。

从训练出的 checkpoint 采样：

```bash
uv run python scripts/sample.py --checkpoint out/smoke --prompt "Qwen is"
```

评估 checkpoint：

```bash
bash runs/eval_smoke.sh
```

生成 checkpoint 报告：

```bash
bash runs/report_smoke.sh
```

写出 JSON 报告：

```bash
uv run python scripts/report.py --checkpoint out/smoke --format json --out out/smoke/report.json
```

评估脚本默认输出 loss/perplexity，也可以评估 prompt completion 和
multiple-choice：

```bash
uv run python scripts/eval.py \
  --checkpoint out/smoke \
  --prompts examples/prompts_tiny.jsonl \
  --multiple-choice examples/multiple_choice_tiny.jsonl
```

运行 tiny assistant-masked SFT：

```bash
bash runs/sft_smoke.sh
```

对 GPT checkpoint 运行同一套 SFT 接口：

```bash
bash runs/gpt_sft_smoke.sh
```

运行 tiny DPO：

```bash
bash runs/dpo_smoke.sh
```

启动本地浏览器聊天 UI：

```bash
bash runs/chat_web.sh
```

然后打开：

```text
http://127.0.0.1:8000
```

## HF-style import/export

检查本地 HF-style 导入导出：

```bash
bash runs/hf_local_smoke.sh
```

只检查真实 HF repo 的 config/tokenizer，不下载权重：

```bash
uv run python scripts/hf_smoke.py Qwen/Qwen3-0.6B
```

只有明确需要下载并导入权重时，才传 `--weights`。

## 数据格式

基础训练使用普通 UTF-8 文本文件：

```bash
uv run python scripts/train.py --data data.txt --out-dir out/base
```

也可以使用命名数据集。数据集相关脚本放在 `dataset/<dataset_name>/`，共享
注册表放在 `dataset/registry.py`，真实数据默认写入仓库根目录的 `data/`（已被
git 忽略）。当前内置 `climbmix`，对应 `karpathy/climbmix-400b-shuffle`，和
`autoresearch` 一样使用 parquet shard；默认准备 10 个训练 shard，并固定使用
`shard_06542.parquet` 作为 validation shard：

```bash
uv run python dataset/climbmix/prepare.py --num-shards 10
uv run python scripts/train.py --dataset climbmix --out-dir out/climbmix
```

也可以使用统一 dispatcher：

```bash
uv run python dataset/prepare.py --dataset climbmix --num-shards 10
```

默认目录结构：

```text
dataset/climbmix/
  spec.py
  prepare.py
data/climbmix/shards/
data/climbmix/prepared/
```

快速试跑可以只准备 1 个训练 shard，或者让训练入口在缺失时自动下载和物化：

```bash
uv run python dataset/climbmix/prepare.py --num-shards 1
uv run python scripts/train.py --dataset climbmix --dataset-num-shards 1

uv run python scripts/train.py \
  --dataset climbmix \
  --dataset-num-shards 1 \
  --download \
  --out-dir out/climbmix-smoke
```

新增数据集时，建议新增 `dataset/<name>/spec.py` 和
`dataset/<name>/prepare.py`，再在 `dataset/registry.py` 注册这个 spec；训练脚本
无需再加分支。

当前还内置了 IMDb movie review 数据集：

```bash
uv run python dataset/imdb/prepare.py
uv run python scripts/train.py --dataset imdb --out-dir out/imdb
```

IMDb 会下载 `train`、`test` 和 `unsupervised` 三个 parquet split，并物化到
`data/imdb/prepared/`。它更适合情感分类或 prompt/SFT 实验；作为普通 LM 文本训练
也可以先用来跑轻量流程。

SFT 使用 JSONL。纯文本行会训练所有 token：

```json
{"text":"nanoqwen is a tiny Qwen-style model project."}
```

也可以使用 chat messages，此时只把 assistant token 计入 loss：

```json
{"messages":[{"role":"user","content":"Say hello."},{"role":"assistant","content":"hello"}]}
```

multiple-choice 评估使用每个候选答案的平均 log probability：

```json
{"question":"The capital of France is","choices":[" Paris"," Berlin"],"answer":0}
```

DPO 使用 preference JSONL，可以是纯 prompt：

```json
{"prompt":"user: Say hello.\nassistant: ","chosen":"hello","rejected":"goodbye"}
```

也可以用 chat messages 作为 prompt 上下文：

```json
{"messages":[{"role":"user","content":"What is this?"}],"chosen":"nanoqwen is a tiny Qwen-style model project.","rejected":"I cannot answer."}
```

## 当前已实现

- Qwen-like decoder-only causal LM：RMSNorm、RoPE、GQA、SwiGLU MLP、可选
  Q/K head norm、causal mask、next-token loss、KV cache。
- native checkpoint 保存/加载。
- byte-tokenized tiny 训练和采样。
- HF-style import/export。
- Qwen2/Qwen3 结构 parity 测试。
- assistant-only SFT masking。
- DPO preference tuning。
- loss/perplexity、prompt exact-match、multiple-choice 评估。
- 本地 web chat UI。
- Markdown/JSON checkpoint 报告。
- 手写 Qwen3-0.6B text-only 模型。
- 手写 Qwen3.5-0.8B text-only 模型。
- GPT-2/nanoGPT 风格 text-only 对比模型。
- 手写模型和 Transformers 的确定性生成对照 smoke。

## 非目标

- 不实现 Qwen3.5 多模态视觉路径。
- 不把 `models/qwen3.py` 或 `models/qwen35.py` 改成 HF wrapper。
- 不把真实模型权重提交到仓库。
- 不保证 sampling 输出和 Transformers 严格一致；严格对照请使用
  `temperature=0`。
- 不把 FlashAttention 用作唯一实现；`eager` 仍然是可读和调试基线。
