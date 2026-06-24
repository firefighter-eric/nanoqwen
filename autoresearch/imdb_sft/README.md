# IMDb SFT

Run the five sequential Qwen3-0.6B IMDb SFT experiments:

```bash
bash autoresearch/imdb_sft/run.sh
```

Equivalent Python entrypoint:

```bash
uv run python -m autoresearch.imdb_sft run-suite \
  --config autoresearch/imdb_sft/experiments.json
```

The default suite starts from the requested baseline:

- `epochs=3`
- `lr=1e-5`
- effective `batch_size=16`

It then runs four nearby variations: lower LR, higher LR, longer training, and a
larger effective batch. Each epoch is saved as its own checkpoint under
`out/autoresearch/imdb_sft/<experiment-id>/epoch_XXX/`, and each saved epoch is
evaluated on the IMDb test split. The same directory also records
`training_params.json`, `eval_result.json`, and `result.json` so the checkpoint,
training parameters, and evaluation result can be inspected together. Aggregate
metrics are written to
`out/autoresearch/imdb_sft/results.jsonl` and
`out/autoresearch/imdb_sft/results.csv`.

The checked-in config uses a bounded IMDb slice so the full five-run loop can be
iterated locally. Increase `train_examples` and `eval_examples`, or set them to
`null`, for larger runs.

## Recorded Results

These results were produced with the checked-in bounded IMDb slice:
`train_examples=256`, `eval_examples=256`. Each cell is `eval_acc / eval_loss`
for the checkpoint saved at that epoch.

| Experiment | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 | Epoch 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `exp01_baseline_e3_lr1e-5_b16` | 82.42% / 0.2048 | 88.67% / 0.1680 | 86.33% / 0.2162 | - | - |
| `exp02_low_lr_e3_lr5e-6_b16` | 85.94% / 0.1636 | 87.50% / 0.1557 | 88.28% / 0.1532 | - | - |
| `exp03_high_lr_e3_lr2e-5_b16` | 86.33% / 0.1653 | 88.28% / 0.1740 | 87.50% / 0.6442 | - | - |
| `exp04_longer_e5_lr1e-5_b16` | 85.94% / 0.1585 | 88.28% / 0.1767 | 87.89% / 0.2475 | 87.11% / 0.4174 | 87.11% / 0.5151 |
| `exp05_bigger_batch_e3_lr1e-5_b32` | 66.41% / 0.3154 | 87.50% / 0.1490 | 88.67% / 0.1371 | - | - |

Best accuracy is tied between `exp01_baseline_e3_lr1e-5_b16` epoch 2 and
`exp05_bigger_batch_e3_lr1e-5_b32` epoch 3 at 88.67%. The lower eval loss is
from `exp05_bigger_batch_e3_lr1e-5_b32` epoch 3.

## Public IMDb References

The public IMDb sentiment benchmark is normally reported on the full Large Movie
Review Dataset split: 25,000 labeled reviews for training and 25,000 labeled
reviews for testing. The bounded 256-example results above are useful for local
iteration, but are not directly comparable to full-benchmark scores.

Useful reference points:

| Result | Accuracy | Notes | Source |
| --- | ---: | --- | --- |
| `RoBERTa-large with LlamBERT` | 96.68% | RoBERTa-large combined with extra LLM-labeled IMDb data, evaluated on the full IMDb test set. | [LlamBERT arXiv](https://arxiv.org/html/2403.15938v1) |
| `CFA ensemble` | 97.072% | RoBERTa plus classical models fused with combinatorial fusion analysis; newer preprint claim. | [CFA sentiment arXiv](https://arxiv.org/html/2510.27014v1) |

The original dataset description is available from the
[Stanford Large Movie Review Dataset page](https://ai.stanford.edu/~amaas/data/sentiment/).
