# AGENTS.md

这个文件给后续维护 nanoqwen 的 agent/开发者使用。仓库根目录下的所有文件都
遵循这里的说明。

## 项目定位

nanoqwen 是一个可读优先的小型 Qwen 风格 LLM 实验仓库。它有两类模型代码：

- `nanoqwen/models/qwen.py`：通用 Qwen-like decoder-only causal LM，用于
  tiny 训练、SFT、DPO、采样、评估和 HF-style import/export。
- `nanoqwen/models/qwen3.py` 与 `nanoqwen/models/qwen35.py`：手写真实 Qwen
  text-only 推理模型，用于加载本地 HF checkpoint 并和 Transformers 做生成
  结果对照。
- `nanoqwen/models/gpt.py`：GPT-2/nanoGPT 风格 decoder-only causal LM，用于
  和 Qwen-like 结构做实验对比。

根目录下的 `model.py`、`qwen3_model.py`、`qwen35_model.py`、`gpt_model.py`
是兼容导出 shim。新增模型定义默认放在 `nanoqwen/models/`。

不要把真实 Qwen 推理路径改成 `AutoModelForCausalLM` wrapper。Transformers 只
应该出现在 tokenizer/chat-template 辅助、HF-style import/export、smoke 或
compare 脚本里。

## 关键文件

- `nanoqwen/models/qwen.py`
  - 通用 Qwen-like decoder-only 模型定义。
  - 仍通过 `nanoqwen/model.py` 兼容导出。

- `nanoqwen/models/gpt.py`
  - GPT-2/nanoGPT 风格 decoder-only 模型定义。
  - 仍通过 `nanoqwen/gpt_model.py` 兼容导出。

- `nanoqwen/models/qwen3.py`
  - 目标模型：`Qwen/Qwen3-0.6B`
  - 默认路径：`models/Qwen/Qwen3-0.6B`
  - 加载文件：`config.json`、`tokenizer_config.json`、`model.safetensors`
  - 当前实现复用 `NanoqwenForCausalLM`，要求 state dict 名称和手写结构对齐。
  - 仍通过 `nanoqwen/qwen3_model.py` 兼容导出。

- `nanoqwen/models/qwen35.py`
  - 目标模型：`Qwen/Qwen3.5-0.8B`
  - 默认路径：`models/Qwen/Qwen3.5-0.8B`
  - 加载文件：`config.json`、`tokenizer_config.json`、
    `model.safetensors-00001-of-00001.safetensors`
  - 只加载 `model.language_model.*` 文本权重。
  - 不实现视觉 encoder、多模态输入或 MTP。
  - 仍通过 `nanoqwen/qwen35_model.py` 兼容导出。

- `nanoqwen/manual_text.py`
  - 手写真实模型共用的 dtype、tokenizer、chat prompt 和 generation 工具。

- `nanoqwen/attention.py`
  - 统一管理 `eager`、`sdpa`、`flash_attention_2` backend。
  - CLI/API 也接受 `flash-attn2`、`flash_attn2`、`flash2`、`fa2`，但内部统一
    归一化为 `flash_attention_2`。
  - `flash_attention_2` 是可选路径，依赖外部 `flash_attn` 包和 CUDA tensor。

- `docs/attention_backends.md`
  - attention backend 的专门文档，包含安装组合、使用方式、HF 对照范围和
    1024/2K/4K forward benchmark。

- `nanoqwen/hf_text.py`
  - 只放 tokenizer/chat-template 和 Transformers 对照所需的轻量 helper。

- `scripts/qwen_llm_generate.py`
  - 手写 Qwen3/Qwen3.5 family-selecting 生成入口。

- `scripts/qwen_llm_compare.py`
  - 手写模型 vs 直接 Transformers 的生成结果对照入口。

## 本地模型文件

真实模型权重放在 `models/` 下，不能提交到 git。

下载命令：

```bash
bash runs/download_qwen3_06b.sh
bash runs/download_qwen35_08b.sh
```

如果改下载脚本，优先加入更强的 revision/hash 校验；不要只依赖文件名。

## Worktree 本地资源

如果使用 `git worktree` 开新工作区，不要在每个 worktree 里重新下载或复制本地
大文件。新 worktree 创建后，把主工作区中的本地资源用软链接连接过来，至少包
括：

- `models/`
- `data/`
- `out/`
- `runs/`
- `.venv/`

示例：

```bash
git worktree add -b feat/exp ../nanoqwen-exp
cd ../nanoqwen-exp
rm -rf models data out runs .venv
ln -s ../nanoqwen/models models
ln -s ../nanoqwen/data data
ln -s ../nanoqwen/out out
ln -s ../nanoqwen/runs runs
ln -s ../nanoqwen/.venv .venv
```

注意 `runs/` 下有仓库跟踪的脚本；使用软链接后，在 worktree 里修改 `runs/`
会直接改到主工作区对应目录。上面的 `rm -rf` 只能在新建 worktree 目录中使用。

## 开发约束

- 保持代码可读，优先使用 plain PyTorch 和现有项目风格。
- 新增模型能力时，先确认它属于通用 tiny 模型路径、GPT 对比路径，还是手写真实
  Qwen 路径。
- 不要引入大框架式抽象，除非它明显减少重复或对齐已有模式。
- 不要提交 `models/`、`out/`、缓存、日志或真实权重文件。
- 训练、全量评测、长 benchmark 等长时间任务默认放到 `tmux` 里运行，并记录
  session 名、启动命令和输出目录；不要在普通前台 shell 里启动新的长任务。
- 改动真实模型实现时，必须考虑和 HF checkpoint 的 state dict 名称对齐。
- 生成 parity 以 `temperature=0` 的 greedy 输出为准；sampling 不作为严格一致
  标准。
- 保留 `eager` attention 作为可读基线。新增优化 backend 时，必须和 eager 做
  数值对齐测试。
- Qwen3.5 的 SDPA/Flash 只适用于 `full_attention` 层；不要把
  `linear_attention` 层替换成 SDPA/Flash，它是 GatedDeltaNet。

## 推荐检查

普通改动至少运行：

```bash
uv run python -m compileall nanoqwen scripts tests
uv run pytest -q
```

或者：

```bash
bash runs/check.sh
```

涉及训练、评估、SFT、DPO、web 或 checkpoint import/export 时，运行：

```bash
bash runs/check.sh --smoke
```

涉及 Qwen3-0.6B 手写模型时，确认本地已下载权重后运行：

```bash
bash runs/check.sh --qwen3
```

涉及 Qwen3.5-0.8B 手写模型时，确认本地已下载权重后运行：

```bash
bash runs/check.sh --qwen35
```

两者都涉及时：

```bash
bash runs/check.sh --qwen
```

改 attention backend 时，至少运行：

```bash
uv run pytest -q tests/test_attention_backends.py tests/test_model.py tests/test_hf_parity.py
```

## 已知边界

- Qwen3.5 当前只覆盖 text LLM，不覆盖多模态和 MTP。
- Qwen3/Qwen3.5 的严格生成一致性主要验证 greedy 解码。
- `flash_attention_2` 需要 CUDA 和可选 `flash_attn` 包；CPU 或未安装依赖时不应
  假装启用 FlashAttention。
- 带 padding mask 的 full-attention 场景会走 SDPA 安全路径；无 padding 的
  prefill 和 KV cache decode 可以走 FlashAttention-2。
- 本地 web chat 面向 localhost 开发使用；如果要暴露到外网，需要先加请求大小、
  token 数、并发和超时限制。
- checkpoint 读取默认信任本地文件；不要对不可信来源直接 `torch.load`。
