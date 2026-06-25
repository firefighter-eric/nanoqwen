# Pretrain Architecture Compare

Karpathy's `autoresearch` keeps comparisons fair by fixing the protocol and
comparing runs with the same metric. This suite applies that idea to nanoqwen's
pretraining path: GPT and Qwen-like models use the same tokenizer, data path,
context length, batch size, optimizer settings, seed, and time/step budget.
The full suite aligns the non-model protocol with `../autoresearch`: it loads
`~/.cache/autoresearch/tokenizer/tokenizer.pkl`, reads climbmix parquet shards
directly, prepends BOS per document, uses autoresearch-style best-fit packing,
and evaluates BPB over `40*524288` validation tokens.

The default comparison keeps the shared protocol fixed, but does not force
parameter counts to match. The Qwen-like config is downscaled from the local
`Qwen/Qwen3-0.6B` text config:

- `hidden_size`: 1024 -> 512
- `intermediate_size / hidden_size`: 3072 / 1024 = 3, so 512 -> 1536
- `num_key_value_heads / num_attention_heads`: 8 / 16 = 1/2, so 4 heads -> 2 kv heads
- `head_dim=128`, `rope_theta=1000000`, `tie_word_embeddings=true`

| id | architecture | core shape | parameters |
| --- | --- | --- | ---: |
| `gpt_h512_l8_ctx2048` | GPT-2/nanoGPT-style | vocab=8192, hidden=512, layers=8, heads=4, head_dim=128, context=2048 | 30,462,976 |
| `qwen3_scaled_h512_l8_ctx2048` | Qwen3-proportional | vocab=8192, hidden=512, layers=8, heads=4, kv_heads=2, head_dim=128, intermediate=1536, context=2048 | 29,370,880 |

The Qwen-like model is about 96.4% of the GPT parameter count in this setup.
That difference is intentional: the Qwen side preserves the Qwen3-0.6B shape
ratios instead of compensating with an artificial MLP size.

Note: `run_smoke.sh` stays lightweight and uses the local text path so ordinary
checks do not require multi-GB climbmix shards or the autoresearch cache. The
full `experiments.json` is the autoresearch-aligned protocol.

Prepare the autoresearch tokenizer and data first:

```bash
cd ../autoresearch
uv run python prepare.py
cd ../nanoqwen
```

The older `prepare_tokenizer.sh` path creates a Hugging Face ByteLevel BPE
tokenizer for local experiments. It is not bit-exact with autoresearch and is
not used by the full architecture comparison.

Run the smoke suite:

```bash
bash autoresearch/pretrain_arch_compare/run_smoke.sh
```

Run the default comparison:

```bash
bash autoresearch/pretrain_arch_compare/run.sh
```

Results are written under `out/autoresearch/pretrain_arch_compare*/` as
`results.jsonl` and `results.csv`, with each experiment's checkpoint and log in
its own subdirectory.
