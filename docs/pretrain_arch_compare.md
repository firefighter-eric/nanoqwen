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

## NanoGPT Optimizer Ablations

To isolate which autoresearch training choices matter most, each ablation below
removes one design from the NanoGPT autoresearch-standard run while keeping the
same model shape, tokenizer, data, batch size, eval budget, seed, and 300 second
measured training budget.

Baseline for deltas:
`nanogpt_ar_h384_l8_ctx2048`, `val_bpb=1.111180`, `step=474`.

Output root:
`out/autoresearch/nanogpt_optimizer_ablations`

| rank | ablation | removed design | step | tokens/sec | val_bpb | delta vs baseline |
|---:|---|---|---:|---:|---:|---:|
| 1 | `ablate_matrix_muon` | Matrix params use AdamW instead of Muon | 477 | 416K | 1.510847 | +0.399667 |
| 2 | `ablate_mixed_precision` | Mixed precision off, fp32 training | 175 | 153K | 1.256081 | +0.144902 |
| 3 | `ablate_unembedding_lr` | `lm_head` LR lowered from `0.006` to `0.0003` | 470 | 410K | 1.230440 | +0.119260 |
| 4 | `ablate_embedding_lr` | token/value embedding LR lowered from `0.6` to `0.0003` | 475 | 414K | 1.198628 | +0.087449 |
| 5 | `ablate_compile` | `torch.compile` off | 231 | 201K | 1.198613 | +0.087434 |
| 6 | `ablate_warmdown` | LR warmdown disabled | 475 | 415K | 1.143108 | +0.031928 |
| 7 | `ablate_scalar_lr` | scalar LR lowered from `0.5` to `0.0003` | 473 | 413K | 1.117443 | +0.006263 |
| 8 | `ablate_momentum_ramp` | Muon momentum starts at `0.95` instead of ramping `0.85 -> 0.95` | 475 | 415K | 1.113607 | +0.002427 |
| 9 | `ablate_weight_decay` | Matrix weight decay set to `0` | 475 | 415K | 1.112837 | +0.001657 |
| 10 | `ablate_weight_decay_schedule` | Matrix weight decay kept constant instead of linearly decaying | 475 | 414K | 1.110230 | -0.000950 |

Main readout:

- The largest optimization-design contributor is Muon for transformer matrix
  parameters. Removing it regresses BPB by about `+0.400`, far larger than any
  other single optimizer ablation.
- The next largest optimizer contributors are the parameter-group learning
  rates: high `lm_head` LR (`+0.119` when removed) and high token/value
  embedding LR (`+0.087` when removed).
- LR warmdown helps, but less than Muon and the high embedding/unembedding
  rates (`+0.032` when removed).
- Muon momentum ramp, scalar LR, weight decay, and weight-decay scheduling are
  small in this single-run 300 second setting.
- Compile and mixed precision are runtime controls rather than optimizer
  semantics. They matter because the benchmark is time-budgeted: fp32 only
  trained 175 steps and compile-off trained 231 steps, versus roughly 474-477
  steps for the compiled bf16 runs.

## Commands

The AdamW baseline runs and NanoGPT autoresearch-standard reproduction were
executed in separate tmux sessions, one at a time:

```bash
tmux new-session -d -s nanogpt_compile "PYTORCH_ALLOC_CONF=expandable_segments:True uv run python scripts/train.py --out-dir out/autoresearch/pretrain_arch_compare_qwen3_scaled_sdpa_compile/nanogpt_ar_h384_l8_ctx2048 --model nanogpt --dataset climbmix --data-format autoresearch --dataset-num-shards 10 --tokenizer /home/eric/.cache/autoresearch/tokenizer --steps 1000 --time-budget-sec 300 --time-budget-warmup-steps 10 --batch-size 8 --block-size 2048 --total-batch-tokens 262144 --lr 0.0003 --weight-decay 0.1 --eval-every 0 --eval-tokens 20971520 --save-every 0 --device cuda --mixed-precision auto --compile on --seed 2026 --vocab-size 8192 --hidden-size 384 --layers 8 --heads 3 --kv-heads 3 --window-pattern L --attn-implementation sdpa --use-dataset-val"

tmux new-session -d -s nanogpt_autoresearch_standard "PYTORCH_ALLOC_CONF=expandable_segments:True uv run python scripts/train.py --out-dir out/autoresearch/pretrain_arch_compare_autoresearch_standard/nanogpt_ar_h384_l8_ctx2048 --model nanogpt --dataset climbmix --data-format autoresearch --dataset-num-shards 10 --tokenizer /home/eric/.cache/autoresearch/tokenizer --steps 1000 --time-budget-sec 300 --time-budget-warmup-steps 10 --batch-size 8 --block-size 2048 --total-batch-tokens 262144 --weight-decay 0.2 --optimizer autoresearch --eval-every 0 --eval-tokens 20971520 --save-every 0 --device cuda --mixed-precision auto --compile on --seed 2026 --vocab-size 8192 --hidden-size 384 --layers 8 --heads 3 --kv-heads 3 --window-pattern L --attn-implementation sdpa --use-dataset-val"

tmux new-session -d -s gpt_compile "PYTORCH_ALLOC_CONF=expandable_segments:True uv run python scripts/train.py --out-dir out/autoresearch/pretrain_arch_compare_qwen3_scaled_sdpa_compile/gpt_h512_l8_ctx2048 --model gpt --dataset climbmix --data-format autoresearch --dataset-num-shards 10 --tokenizer /home/eric/.cache/autoresearch/tokenizer --steps 1000 --time-budget-sec 300 --time-budget-warmup-steps 10 --batch-size 8 --block-size 2048 --total-batch-tokens 262144 --lr 0.0003 --weight-decay 0.1 --eval-every 0 --eval-tokens 20971520 --save-every 0 --device cuda --mixed-precision auto --compile on --seed 2026 --vocab-size 8192 --hidden-size 512 --layers 8 --heads 4 --dropout 0.0 --attn-implementation sdpa --use-dataset-val"

tmux new-session -d -s qwen_compile "PYTORCH_ALLOC_CONF=expandable_segments:True uv run python scripts/train.py --out-dir out/autoresearch/pretrain_arch_compare_qwen3_scaled_sdpa_compile/qwen3_scaled_h512_l8_ctx2048 --model qwen --dataset climbmix --data-format autoresearch --dataset-num-shards 10 --tokenizer /home/eric/.cache/autoresearch/tokenizer --steps 1000 --time-budget-sec 300 --time-budget-warmup-steps 10 --batch-size 8 --block-size 2048 --total-batch-tokens 262144 --lr 0.0003 --weight-decay 0.1 --eval-every 0 --eval-tokens 20971520 --save-every 0 --device cuda --mixed-precision auto --compile on --seed 2026 --vocab-size 8192 --hidden-size 512 --intermediate-size 1536 --layers 8 --heads 4 --kv-heads 2 --rope-theta 1000000.0 --tie-word-embeddings --dropout 0.0 --attn-implementation sdpa --use-dataset-val"

tmux new-session -d -s nanogpt_optimizer_ablations "PYTORCH_ALLOC_CONF=expandable_segments:True uv run python -m autoresearch.pretrain_arch_compare run-suite --config autoresearch/pretrain_arch_compare/optimizer_ablations.json"
```

## Verification

The code changes that enabled these runs were checked with:

```bash
uv run python -m compileall nanoqwen scripts tests autoresearch
uv run pytest -q tests/test_nanogpt_model.py tests/test_train_schedule.py
uv run pytest -q tests/test_checkpoint.py tests/test_eval.py
bash autoresearch/pretrain_arch_compare/run_smoke.sh
uv run pytest -q
PYTORCH_ALLOC_CONF=expandable_segments:True uv run python scripts/train.py --out-dir out/tmp/autoresearch_optimizer_cuda_smoke --model nanogpt --steps 2 --batch-size 1 --block-size 32 --grad-accum-steps 1 --eval-every 0 --eval-iters 1 --save-every 0 --device cuda --mixed-precision auto --compile on --optimizer autoresearch --seed 2026 --vocab-size 257 --hidden-size 64 --layers 2 --heads 4 --kv-heads 4 --window-pattern L --attn-implementation sdpa --weight-decay 0.2
```
