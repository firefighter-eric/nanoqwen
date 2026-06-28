# Pretrain Architecture Compare

This note records the June 2026 pretraining architecture comparison for the
small GPT, autoresearch-style NanoGPT, and Qwen-like models.

## Protocol

The comparison uses the project pretraining script with the autoresearch data
path and tokenizer:

- Dataset: `climbmix`
- Data format: `autoresearch`
- Tokenizer: `~/.cache/autoresearch/tokenizer`
- Train shards: first 10 train parquet shards
- Validation shard: pinned dataset validation split
- Context length: `2048`
- Vocab size: `8192`
- Micro batch: `8` sequences
- Effective batch: `262144` tokens per optimizer step
- Attention backend: `sdpa`
- Mixed precision: `auto`, resolved to `bfloat16` on CUDA
- Compile: `on`
- Time budget: `300` measured training seconds
- Time-budget warmup: first `10` optimizer steps excluded from the measured
  time budget, so compile/warmup overhead does not consume the 300 second
  window.
- Eval budget: `20971520` validation tokens
- Optimizer:
  - GPT/Qwen baseline: project default `AdamW`, `lr=0.0003`,
    `weight_decay=0.1`
  - NanoGPT autoresearch-standard reproduction: Muon+AdamW parameter groups
    from `../autoresearch/train.py`, `embedding_lr=0.6`,
    `unembedding_lr=0.006`, `matrix_lr=0.04`, `scalar_lr=0.5`,
    `weight_decay=0.2`, `warmdown_ratio=0.62`, `final_lr_frac=0.05`

The exact suite config is in
`autoresearch/pretrain_arch_compare/experiments.json`.

## Models

| model | id | layers | hidden | heads | kv heads | intermediate | params |
|---|---|---:|---:|---:|---:|---:|---:|
| NanoGPT | `nanogpt_ar_h384_l8_ctx2048` | 8 | 384 | 3 | 3 | 1536 | 33.03M |
| GPT | `gpt_h512_l8_ctx2048` | 8 | 512 | 4 | 4 | 2048 | 30.46M |
| Qwen-like | `qwen3_scaled_h512_l8_ctx2048` | 8 | 512 | 4 | 2 | 1536 | 29.37M |

NanoGPT uses the current `../autoresearch` RTX 5080 shape:
`hidden=384`, `layers=8`, `heads=3`, `head_dim=128`, full attention.

## Autoresearch-Standard NanoGPT Result

After adding the autoresearch optimizer and schedule, the NanoGPT target was
rerun with the same data/tokenizer/batch/eval protocol and the
`../autoresearch` RTX 5080 model shape.

Output root:
`out/autoresearch/pretrain_arch_compare_autoresearch_standard`

| model | id | params | step | tokens | tokens/sec | val_bpb | val_loss | wall sec |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| NanoGPT | `nanogpt_ar_h384_l8_ctx2048` | 33.03M | 474 | 124.3M | 414K | 1.111180 | 3.1119 | 324.2 |

This matches the intended autoresearch-level target: the local
`../autoresearch` RTX 5080 reference is about `val_bpb=1.101613` for the same
300 second measured training budget. The remaining difference is mostly
throughput: nanoqwen trained 474 optimizer steps in the measured window, while
the reference trained 560 steps.

## AdamW Architecture Baseline Results

Output root:
`out/autoresearch/pretrain_arch_compare_qwen3_scaled_sdpa_compile`

| model | id | params | step | tokens | tokens/sec | val_bpb | val_loss | wall sec |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| NanoGPT | `nanogpt_ar_h384_l8_ctx2048` | 33.03M | 451 | 118.2M | 394K | 1.797817 | 5.0356 | 318.9 |
| GPT | `gpt_h512_l8_ctx2048` | 30.46M | 359 | 94.1M | 313K | 1.842542 | 5.1605 | 316.6 |
| Qwen-like | `qwen3_scaled_h512_l8_ctx2048` | 29.37M | 335 | 87.8M | 292K | 1.656265 | 4.6385 | 322.0 |

Lower `val_bpb` is better. In this run, the Qwen-like model has the best
validation BPB, despite lower throughput. The autoresearch-style NanoGPT has
the highest throughput and trains the most tokens within the same measured
time budget.

## Interpretation

The AdamW baseline is a controlled comparison under one shared optimizer. The
autoresearch-standard NanoGPT result above is a reproduction check for the
Karpathy protocol and should not be mixed with the AdamW architecture table as a
like-for-like architecture comparison.

## Commands

The AdamW baseline runs and NanoGPT autoresearch-standard reproduction were
executed in separate tmux sessions, one at a time:

```bash
tmux new-session -d -s nanogpt_compile "PYTORCH_ALLOC_CONF=expandable_segments:True uv run python scripts/train.py --out-dir out/autoresearch/pretrain_arch_compare_qwen3_scaled_sdpa_compile/nanogpt_ar_h384_l8_ctx2048 --model nanogpt --dataset climbmix --data-format autoresearch --dataset-num-shards 10 --tokenizer /home/eric/.cache/autoresearch/tokenizer --steps 1000 --time-budget-sec 300 --time-budget-warmup-steps 10 --batch-size 8 --block-size 2048 --total-batch-tokens 262144 --lr 0.0003 --weight-decay 0.1 --eval-every 0 --eval-tokens 20971520 --save-every 0 --device cuda --mixed-precision auto --compile on --seed 2026 --vocab-size 8192 --hidden-size 384 --layers 8 --heads 3 --kv-heads 3 --window-pattern L --attn-implementation sdpa --use-dataset-val"

tmux new-session -d -s nanogpt_autoresearch_standard "PYTORCH_ALLOC_CONF=expandable_segments:True uv run python scripts/train.py --out-dir out/autoresearch/pretrain_arch_compare_autoresearch_standard/nanogpt_ar_h384_l8_ctx2048 --model nanogpt --dataset climbmix --data-format autoresearch --dataset-num-shards 10 --tokenizer /home/eric/.cache/autoresearch/tokenizer --steps 1000 --time-budget-sec 300 --time-budget-warmup-steps 10 --batch-size 8 --block-size 2048 --total-batch-tokens 262144 --weight-decay 0.2 --optimizer autoresearch --eval-every 0 --eval-tokens 20971520 --save-every 0 --device cuda --mixed-precision auto --compile on --seed 2026 --vocab-size 8192 --hidden-size 384 --layers 8 --heads 3 --kv-heads 3 --window-pattern L --attn-implementation sdpa --use-dataset-val"

tmux new-session -d -s gpt_compile "PYTORCH_ALLOC_CONF=expandable_segments:True uv run python scripts/train.py --out-dir out/autoresearch/pretrain_arch_compare_qwen3_scaled_sdpa_compile/gpt_h512_l8_ctx2048 --model gpt --dataset climbmix --data-format autoresearch --dataset-num-shards 10 --tokenizer /home/eric/.cache/autoresearch/tokenizer --steps 1000 --time-budget-sec 300 --time-budget-warmup-steps 10 --batch-size 8 --block-size 2048 --total-batch-tokens 262144 --lr 0.0003 --weight-decay 0.1 --eval-every 0 --eval-tokens 20971520 --save-every 0 --device cuda --mixed-precision auto --compile on --seed 2026 --vocab-size 8192 --hidden-size 512 --layers 8 --heads 4 --dropout 0.0 --attn-implementation sdpa --use-dataset-val"

tmux new-session -d -s qwen_compile "PYTORCH_ALLOC_CONF=expandable_segments:True uv run python scripts/train.py --out-dir out/autoresearch/pretrain_arch_compare_qwen3_scaled_sdpa_compile/qwen3_scaled_h512_l8_ctx2048 --model qwen --dataset climbmix --data-format autoresearch --dataset-num-shards 10 --tokenizer /home/eric/.cache/autoresearch/tokenizer --steps 1000 --time-budget-sec 300 --time-budget-warmup-steps 10 --batch-size 8 --block-size 2048 --total-batch-tokens 262144 --lr 0.0003 --weight-decay 0.1 --eval-every 0 --eval-tokens 20971520 --save-every 0 --device cuda --mixed-precision auto --compile on --seed 2026 --vocab-size 8192 --hidden-size 512 --intermediate-size 1536 --layers 8 --heads 4 --kv-heads 2 --rope-theta 1000000.0 --tie-word-embeddings --dropout 0.0 --attn-implementation sdpa --use-dataset-val"
```

## Verification

The code changes that enabled these runs were checked with:

```bash
uv run python -m compileall nanoqwen scripts tests autoresearch
uv run pytest -q tests/test_nanogpt_model.py tests/test_checkpoint.py tests/test_eval.py
bash autoresearch/pretrain_arch_compare/run_smoke.sh
uv run pytest -q
PYTORCH_ALLOC_CONF=expandable_segments:True uv run python scripts/train.py --out-dir out/tmp/autoresearch_optimizer_cuda_smoke --model nanogpt --steps 2 --batch-size 1 --block-size 32 --grad-accum-steps 1 --eval-every 0 --eval-iters 1 --save-every 0 --device cuda --mixed-precision auto --compile on --optimizer autoresearch --seed 2026 --vocab-size 257 --hidden-size 64 --layers 2 --heads 4 --kv-heads 4 --window-pattern L --attn-implementation sdpa --weight-decay 0.2
```
