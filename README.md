# nanoqwen

nanoqwen is a small, readable Qwen-style language model project. The shape is
inspired by nanochat, the core training loop stays close to nanoGPT, and the
model naming follows Hugging Face Transformers so checkpoints can move between
the tiny codebase and the wider Qwen ecosystem.

The project is intentionally not a replacement for `transformers`. It is a
minimal experimental harness for learning, debugging, and training compact
Qwen-like causal language models end to end.

## Design

- **Readable first**: the model and scripts are plain PyTorch with few layers of
  indirection.
- **Qwen-compatible names**: modules use names such as
  `model.layers.0.self_attn.q_proj` to keep weight import/export simple.
- **End-to-end path**: base training, SFT, evaluation, sampling, chat, and future RL stages
  live as small scripts rather than a large framework.
- **Tiny smoke tests**: the default local workflow uses a byte tokenizer and a
  very small model, so basic behavior can be tested without downloading a Qwen
  checkpoint.

## Layout

```text
nanoqwen/
  config.py        # small dataclass config with Qwen-style fields
  model.py         # Qwen-like decoder-only causal LM
  qwen3_model.py   # local Qwen3-0.6B text-only Transformers LLM wrapper
  qwen35_model.py  # local Qwen3.5-0.8B text-only Transformers LLM wrapper
  generation.py    # greedy/top-k/top-p sampling helpers
  hf_text.py       # shared helpers for local HF text-only Qwen generation
  hf_multimodal.py # helpers for local HF multimodal model prompts
  tokenizer.py     # byte tokenizer plus optional HF tokenizer wrapper
  data.py          # packed token dataset for training
  sft.py           # assistant-masked supervised fine-tuning dataset
  dpo.py           # preference dataset and DPO loss
  eval.py          # loss/perplexity and prompt exact-match evaluation
  report.py        # checkpoint reports
  checkpoint.py    # native save/load plus HF-style import/export helpers
  ui.html          # static chat UI served by scripts/chat_web.py
scripts/
  train.py         # nanoGPT-style training loop
  eval.py          # checkpoint evaluation
  report.py        # checkpoint report generation
  sample.py        # prompt completion
  chat_cli.py      # simple terminal chat
  chat_web.py      # local browser chat UI
  hf_smoke.py      # HF config/tokenizer/optional weight import smoke
  qwen_llm_generate.py # family-selecting Qwen3/Qwen3.5 text-only generation
  qwen_llm_compare.py # compare project LLM wrappers to direct Transformers
  qwen3_llm_generate.py # local Qwen3-0.6B text-only Transformers generation
  qwen3_compare.py # compare Qwen3 wrapper to direct Transformers
  qwen35_llm_generate.py # local Qwen3.5-0.8B text-only Transformers generation
  qwen35_generate.py # optional local Qwen3.5-0.8B multimodal Transformers generation
  qwen35_compare.py # compare project Qwen3.5 LLM backend to direct Transformers
  sft.py           # supervised fine-tuning on text/messages JSONL
  dpo.py           # preference tuning with DPO
  import_hf.py     # import compatible HF Qwen checkpoints
  export_hf.py     # export a native checkpoint in HF-style format
examples/
  sft_tiny.jsonl
  preferences_tiny.jsonl
  prompts_tiny.jsonl
  multiple_choice_tiny.jsonl
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
  smoke.sh         # tiny CPU smoke training
  eval_smoke.sh
  sft_smoke.sh
  dpo_smoke.sh
  report_smoke.sh
  chat_web.sh
  hf_local_smoke.sh
  tiny.sh          # slightly larger local experiment
tests/
```

## Quickstart

Install the project in a virtual environment:

```bash
uv sync --extra dev
```

Text-only Qwen3/Qwen3.5 usage only needs the dev extra. For optional multimodal
experiments such as `scripts/qwen35_generate.py`, also install the vision extra:

```bash
uv sync --extra dev --extra vision
```

Run tests:

```bash
uv run pytest
```

Run the fast project check:

```bash
bash runs/check.sh
```

Run the fuller local smoke suite:

```bash
bash runs/check.sh --smoke
```

Include the downloaded Qwen3.5-0.8B Transformers path in the check:

```bash
bash runs/check.sh --smoke --qwen35
```

Include both downloaded Qwen3-0.6B and Qwen3.5-0.8B Transformers paths:

```bash
bash runs/check.sh --qwen
```

Run a tiny CPU smoke train:

```bash
bash runs/smoke.sh
```

Sample from the resulting checkpoint:

```bash
uv run python scripts/sample.py --checkpoint out/smoke --prompt "Qwen is"
```

Evaluate the checkpoint:

```bash
bash runs/eval_smoke.sh
```

Generate a checkpoint report:

```bash
bash runs/report_smoke.sh
```

Or write JSON:

```bash
uv run python scripts/report.py --checkpoint out/smoke --format json --out out/smoke/report.json
```

The eval script reports loss/perplexity by default and can also score prompt
completion rows or multiple-choice rows:

```bash
uv run python scripts/eval.py \
  --checkpoint out/smoke \
  --prompts examples/prompts_tiny.jsonl \
  --multiple-choice examples/multiple_choice_tiny.jsonl
```

Run a tiny assistant-masked SFT smoke:

```bash
bash runs/sft_smoke.sh
```

Run a tiny DPO preference-tuning smoke:

```bash
bash runs/dpo_smoke.sh
```

Serve the browser chat UI:

```bash
bash runs/chat_web.sh
```

Then open <http://127.0.0.1:8000>.

Check HF-style export/import locally:

```bash
bash runs/hf_local_smoke.sh
```

Check a real Hugging Face Qwen repo without downloading weights:

```bash
uv run python scripts/hf_smoke.py Qwen/Qwen3-0.6B
```

Pass `--weights` only when you intentionally want to download and import the
model weights.

Download Qwen/Qwen3-0.6B into `models/Qwen/Qwen3-0.6B`:

```bash
bash runs/download_qwen3_06b.sh
```

Verify its local tokenizer without loading weights:

```bash
bash runs/qwen3_dry_smoke.sh
```

Run a short local text-only generation through Transformers:

```bash
bash runs/qwen3_text_smoke.sh
```

Verify that the project Qwen3 LLM wrapper produces the same deterministic output
as a direct Transformers call:

```bash
bash runs/qwen3_compare_smoke.sh
```

Download Qwen/Qwen3.5-0.8B into `models/Qwen/Qwen3.5-0.8B`:

```bash
bash runs/download_qwen35_08b.sh
```

This checkpoint includes a multimodal wrapper, but nanoqwen uses it as a
text-only LLM through `AutoTokenizer` + `AutoModelForCausalLM` by default. The
native nanoqwen decoder remains a compact Qwen2/Qwen3-style text decoder; Qwen3.5
same-output mode uses the Transformers Qwen3.5 LLM backend because Qwen3.5 text
layers include hybrid linear-attention blocks.

Verify its local tokenizer without loading weights:

```bash
bash runs/qwen35_dry_smoke.sh
```

Run a short local text-only generation through Transformers:

```bash
bash runs/qwen35_text_smoke.sh
```

Verify that the project Qwen3.5 LLM backend produces the same deterministic output
as a direct Transformers call:

```bash
bash runs/qwen35_compare_smoke.sh
```

The family-selecting entrypoints are also available:

```bash
uv run python scripts/qwen_llm_generate.py --family qwen3 --prompt "你好"
uv run python scripts/qwen_llm_generate.py --family qwen35 --prompt "你好"
```

## Data Formats

Base training expects a plain UTF-8 text file:

```bash
uv run python scripts/train.py --data data.txt --out-dir out/base
```

SFT expects JSONL. A row can contain free text, which trains all tokens:

```json
{"text":"nanoqwen is a tiny Qwen-style model project."}
```

Or chat messages, where only assistant tokens are included in the loss:

```json
{"messages":[{"role":"user","content":"Say hello."},{"role":"assistant","content":"hello"}]}
```

Multiple-choice eval rows use mean log probability of each answer choice:

```json
{"question":"The capital of France is","choices":[" Paris"," Berlin"],"answer":0}
```

DPO expects preference JSONL. Rows can use a plain prompt:

```json
{"prompt":"user: Say hello.\nassistant: ","chosen":"hello","rejected":"goodbye"}
```

Or chat messages as the prompt context:

```json
{"messages":[{"role":"user","content":"What is this?"}],"chosen":"nanoqwen is a tiny Qwen-style model project.","rejected":"I cannot answer."}
```

## Current Scope

Implemented:

- Qwen-style decoder-only causal LM with RMSNorm, RoPE, GQA, SwiGLU MLP, optional
  Q/K head normalization, causal masking, next-token loss, and KV cache.
- Native checkpoint save/load.
- Byte-tokenized training and sampling scripts for local smoke tests.
- HF-style import/export helpers for compatible Qwen state dicts.
- Qwen2 and Qwen3 parity tests against `transformers`.
- Assistant-only SFT masking for chat JSONL.
- DPO preference tuning with a frozen reference model.
- Loss/perplexity evaluation and prompt exact-match evaluation.
- Multiple-choice scoring by answer-choice log probability.
- Local web chat UI.
- Checkpoint reporting in Markdown or JSON.
- Local Qwen3.5-0.8B text-only Transformers generation helper for the downloaded
  checkpoint.
- Deterministic Qwen3.5 LLM generation parity smoke against direct Transformers.

Future Extensions:

- broader task suites beyond local JSONL tasks;
- larger curated training/eval runs;
- richer preference/RL post-training experiments.
