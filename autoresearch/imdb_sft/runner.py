from __future__ import annotations

import argparse
import csv
import gc
import html
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nanoqwen.config import NanoqwenConfig
from nanoqwen.hf_text import load_tokenizer, render_chat_prompt
from nanoqwen.manual_text import DTYPE_NAMES
from nanoqwen.qwen3_model import DEFAULT_MODEL_PATH, Qwen3ForCausalLM, require_downloaded


IMDB_RAW_SPLITS = {
    "train": "train-00000-of-00001.parquet",
    "test": "test-00000-of-00001.parquet",
}
LABEL_TEXT = {0: "negative", 1: "positive"}
SYSTEM_PROMPT = (
    "You classify movie-review sentiment. Respond with exactly one word: "
    "positive or negative."
)


@dataclass(frozen=True)
class ExperimentConfig:
    id: str
    epochs: int
    lr: float
    batch_size: int


@dataclass(frozen=True)
class SuiteConfig:
    model_path: str
    output_root: str
    train_split: str
    eval_split: str
    train_examples: int | None
    eval_examples: int | None
    max_length: int
    micro_batch_size: int
    attn_implementation: str
    dtype: str
    seed: int
    experiments: list[ExperimentConfig]
    gradient_checkpointing: bool = False


@dataclass
class EncodedExample:
    prompt_ids: list[int]
    input_ids: list[int]
    labels: list[int]
    target: int


@dataclass
class EvalResult:
    accuracy: float
    correct: int
    total: int
    loss: float
    examples_per_second: float


def load_suite_config(path: str | Path) -> SuiteConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    experiments = [ExperimentConfig(**item) for item in data.pop("experiments")]
    return SuiteConfig(experiments=experiments, **data)


def imdb_split_path(split: str, data_dir: str | Path = "data/imdb/raw/plain_text") -> Path:
    if split not in IMDB_RAW_SPLITS:
        raise ValueError(f"Unsupported IMDb split {split!r}; expected one of {sorted(IMDB_RAW_SPLITS)}")
    return Path(data_dir) / IMDB_RAW_SPLITS[split]


def clean_review(text: str) -> str:
    text = html.unescape(text.replace("<br /><br />", "\n").replace("<br />", "\n"))
    return " ".join(text.split())


def load_imdb_rows(
    split: str,
    *,
    max_examples: int | None,
    seed: int,
    data_dir: str | Path = "data/imdb/raw/plain_text",
) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("Install data dependencies with `uv sync --extra data --extra dev`.") from exc

    path = imdb_split_path(split, data_dir=data_dir)
    if not path.is_file():
        raise FileNotFoundError(
            f"IMDb raw parquet not found at {path}. Run: uv run python dataset/imdb/prepare.py"
        )
    rows = pq.read_table(path, columns=["text", "label"]).to_pylist()
    rows = [
        {"text": clean_review(str(row["text"])), "label": int(row["label"])}
        for row in rows
        if int(row["label"]) in LABEL_TEXT
    ]

    rng = random.Random(seed)
    if max_examples is None:
        rng.shuffle(rows)
        return rows

    by_label = {
        label: [row for row in rows if row["label"] == label]
        for label in sorted(LABEL_TEXT)
    }
    for label_rows in by_label.values():
        rng.shuffle(label_rows)

    per_label = max_examples // len(by_label)
    selected: list[dict[str, Any]] = []
    for label in sorted(by_label):
        selected.extend(by_label[label][:per_label])

    remaining = max_examples - len(selected)
    if remaining > 0:
        leftovers = [
            row
            for label in sorted(by_label)
            for row in by_label[label][per_label:]
        ]
        rng.shuffle(leftovers)
        selected.extend(leftovers[:remaining])

    rng.shuffle(selected)
    return selected


def build_prompt(review: str) -> str:
    return f"Review:\n{review}\n\nSentiment:"


def answer_ids(tokenizer: Any, label: int) -> list[int]:
    ids = tokenizer.encode(LABEL_TEXT[label], add_special_tokens=False)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        ids.append(int(eos_token_id))
    return [int(token) for token in ids]


def encode_for_sft(
    tokenizer: Any,
    review: str,
    label: int,
    *,
    max_length: int,
) -> EncodedExample:
    target_ids = answer_ids(tokenizer, label)
    max_prompt_length = max_length - len(target_ids)
    if max_prompt_length < 8:
        raise ValueError("max_length is too small for the target answer tokens")

    truncated_review = review
    while True:
        rendered = render_chat_prompt(
            tokenizer,
            build_prompt(truncated_review),
            system=SYSTEM_PROMPT,
            enable_thinking=False,
        )
        prompt_ids = [int(token) for token in tokenizer.encode(rendered, add_special_tokens=False)]
        if len(prompt_ids) <= max_prompt_length:
            break
        if len(truncated_review) <= 200:
            prompt_ids = prompt_ids[-max_prompt_length:]
            break
        keep_chars = max(200, int(len(truncated_review) * 0.8))
        truncated_review = truncated_review[:keep_chars].rstrip()

    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    return EncodedExample(
        prompt_ids=prompt_ids,
        input_ids=input_ids,
        labels=labels,
        target=label,
    )


class EncodedSftDataset(Dataset):
    def __init__(self, examples: Iterable[EncodedExample]) -> None:
        self.examples = list(examples)
        if not self.examples:
            raise ValueError("No encoded examples")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> EncodedExample:
        return self.examples[idx]


def encode_rows(tokenizer: Any, rows: list[dict[str, Any]], max_length: int) -> EncodedSftDataset:
    encoded = [
        encode_for_sft(tokenizer, row["text"], row["label"], max_length=max_length)
        for row in tqdm(rows, desc="encode", leave=False, mininterval=2.0)
    ]
    return EncodedSftDataset(encoded)


def collate_sft(batch: list[EncodedExample], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(item.input_ids) for item in batch)
    input_ids = []
    attention_mask = []
    labels = []
    targets = []
    for item in batch:
        pad_len = max_len - len(item.input_ids)
        input_ids.append(item.input_ids + [pad_token_id] * pad_len)
        attention_mask.append([1] * len(item.input_ids) + [0] * pad_len)
        labels.append(item.labels + [-100] * pad_len)
        targets.append(item.target)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "targets": torch.tensor(targets, dtype=torch.long),
    }


def collate_eval_pairs(batch: list[EncodedExample], pad_token_id: int, tokenizer: Any) -> dict[str, torch.Tensor]:
    pair_input_ids: list[list[int]] = []
    pair_attention_mask: list[list[int]] = []
    pair_labels: list[list[int]] = []
    targets: list[int] = []
    candidate_labels: list[int] = []

    for item in batch:
        for candidate in sorted(LABEL_TEXT):
            candidate_ids = answer_ids(tokenizer, candidate)
            ids = item.prompt_ids + candidate_ids
            labels = [-100] * len(item.prompt_ids) + candidate_ids
            pair_input_ids.append(ids)
            pair_attention_mask.append([1] * len(ids))
            pair_labels.append(labels)
            targets.append(item.target)
            candidate_labels.append(candidate)

    max_len = max(len(ids) for ids in pair_input_ids)
    for ids, mask, labels in zip(pair_input_ids, pair_attention_mask, pair_labels, strict=True):
        pad_len = max_len - len(ids)
        ids.extend([pad_token_id] * pad_len)
        mask.extend([0] * pad_len)
        labels.extend([-100] * pad_len)

    return {
        "input_ids": torch.tensor(pair_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(pair_attention_mask, dtype=torch.long),
        "labels": torch.tensor(pair_labels, dtype=torch.long),
        "targets": torch.tensor(targets, dtype=torch.long),
        "candidate_labels": torch.tensor(candidate_labels, dtype=torch.long),
    }


def move_batch(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def sequence_logprobs(
    model: Qwen3ForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    shift_logits = outputs.logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    mask = shift_labels != -100
    safe_labels = shift_labels.masked_fill(~mask, 0)
    logprobs = F.log_softmax(shift_logits.float(), dim=-1)
    token_logprobs = logprobs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logprobs = token_logprobs.masked_fill(~mask, 0.0)
    lengths = mask.sum(dim=-1).clamp_min(1)
    return token_logprobs.sum(dim=-1), lengths


@torch.no_grad()
def evaluate(
    model: Qwen3ForCausalLM,
    dataset: EncodedSftDataset,
    tokenizer: Any,
    *,
    batch_size: int,
    device: str,
    pad_token_id: int,
) -> EvalResult:
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=lambda batch: collate_eval_pairs(batch, pad_token_id, tokenizer),
    )

    start_time = time.time()
    correct = 0
    total = 0
    loss_sum = 0.0
    supervised_tokens = 0

    for batch in tqdm(loader, desc="eval", leave=False, mininterval=2.0):
        batch = move_batch(batch, device)
        total_logprobs, lengths = sequence_logprobs(
            model,
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        mean_logprobs = total_logprobs / lengths

        scores = mean_logprobs.view(-1, len(LABEL_TEXT))
        candidate_labels = batch["candidate_labels"].view(-1, len(LABEL_TEXT))
        targets = batch["targets"].view(-1, len(LABEL_TEXT))[:, 0]
        predicted = candidate_labels.gather(1, scores.argmax(dim=-1, keepdim=True)).squeeze(1)
        correct += int((predicted == targets).sum().item())
        total += int(targets.numel())

        correct_mask = batch["candidate_labels"] == batch["targets"]
        loss_sum += float((-total_logprobs[correct_mask]).sum().item())
        supervised_tokens += int(lengths[correct_mask].sum().item())

    elapsed = max(time.time() - start_time, 1e-9)
    model.train()
    return EvalResult(
        accuracy=correct / max(1, total),
        correct=correct,
        total=total,
        loss=loss_sum / max(1, supervised_tokens),
        examples_per_second=total / elapsed,
    )


def checkpoint_state_dict(model: Qwen3ForCausalLM) -> dict[str, torch.Tensor]:
    state = {}
    for key, value in model.state_dict().items():
        if key == "lm_head.weight" and model.config.tie_word_embeddings:
            continue
        state[key] = value.detach().cpu()
    return state


def save_epoch_checkpoint(
    model: Qwen3ForCausalLM,
    path: str | Path,
    *,
    metadata: dict[str, Any],
) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.config.to_json_file(path / "config.json")
    torch.save(
        {
            "model": checkpoint_state_dict(model),
            "metadata": metadata,
        },
        path / "checkpoint.pt",
    )
    with (path / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def training_params(
    suite: SuiteConfig,
    experiment: ExperimentConfig,
    *,
    device: str,
    pad_token_id: int,
    grad_accum_steps: int,
) -> dict[str, Any]:
    return {
        "experiment": asdict(experiment),
        "data": {
            "dataset": "imdb",
            "train_split": suite.train_split,
            "eval_split": suite.eval_split,
            "train_examples": suite.train_examples,
            "eval_examples": suite.eval_examples,
            "max_length": suite.max_length,
        },
        "model": {
            "family": "qwen3",
            "model_path": suite.model_path,
            "dtype": suite.dtype,
            "attn_implementation": suite.attn_implementation,
            "gradient_checkpointing": suite.gradient_checkpointing,
            "pad_token_id": pad_token_id,
        },
        "optimization": {
            "optimizer": "AdamW",
            "lr": experiment.lr,
            "weight_decay": 0.0,
            "max_grad_norm": 1.0,
            "micro_batch_size": suite.micro_batch_size,
            "effective_batch_size": experiment.batch_size,
            "grad_accum_steps": grad_accum_steps,
            "device": device,
        },
        "seed": suite.seed,
    }


def load_epoch_checkpoint(path: str | Path, *, device: str, dtype: str) -> Qwen3ForCausalLM:
    path = Path(path)
    config = NanoqwenConfig.from_json_file(path / "config.json")
    model = Qwen3ForCausalLM(config)
    model.to(dtype=DTYPE_NAMES[dtype])
    payload = torch.load(path / "checkpoint.pt", map_location="cpu")
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    allowed_missing = {"lm_head.weight"} if config.tie_word_embeddings else set()
    missing = [key for key in missing if key not in allowed_missing]
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch at {path}: missing={missing}, unexpected={unexpected}")
    return model.to(device)


def load_base_model(config: SuiteConfig, device: str) -> Qwen3ForCausalLM:
    require_downloaded(config.model_path)
    model = Qwen3ForCausalLM.from_pretrained(
        config.model_path,
        dtype=config.dtype,
        attn_implementation=config.attn_implementation,
    )
    model.config.use_cache = False
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model.to(device)


def train_one_experiment(
    suite: SuiteConfig,
    experiment: ExperimentConfig,
    train_dataset: EncodedSftDataset,
    eval_dataset: EncodedSftDataset,
    tokenizer: Any,
    *,
    device: str,
    pad_token_id: int,
) -> list[dict[str, Any]]:
    output_dir = Path(suite.output_root) / experiment.id
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(experiment), handle, indent=2, sort_keys=True)
        handle.write("\n")

    model = load_base_model(suite, device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=experiment.lr, weight_decay=0.0)
    accum_steps = max(1, math.ceil(experiment.batch_size / suite.micro_batch_size))
    params = training_params(
        suite,
        experiment,
        device=device,
        pad_token_id=pad_token_id,
        grad_accum_steps=accum_steps,
    )
    write_json(output_dir / "training_params.json", params)
    train_loader = DataLoader(
        train_dataset,
        batch_size=suite.micro_batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=lambda batch: collate_sft(batch, pad_token_id),
    )

    rows: list[dict[str, Any]] = []
    global_step = 0
    for epoch in range(1, experiment.epochs + 1):
        epoch_loss = 0.0
        epoch_tokens = 0
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(train_loader, desc=f"{experiment.id} epoch {epoch}", leave=True, mininterval=2.0)
        for micro_step, batch in enumerate(progress, start=1):
            batch = move_batch(batch, device)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                use_cache=False,
            )
            loss = outputs.loss
            if loss is None:
                raise RuntimeError("Model did not return a training loss")
            supervised = int((batch["labels"][:, 1:] != -100).sum().item())
            epoch_loss += float(loss.detach().item()) * supervised
            epoch_tokens += supervised

            (loss / accum_steps).backward()
            if micro_step % accum_steps == 0 or micro_step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            if micro_step == 1 or micro_step % 16 == 0 or micro_step == len(train_loader):
                progress.set_postfix(loss=f"{loss.detach().item():.4f}", refresh=False)

        train_loss = epoch_loss / max(1, epoch_tokens)
        eval_result = evaluate(
            model,
            eval_dataset,
            tokenizer,
            batch_size=max(1, suite.micro_batch_size),
            device=device,
            pad_token_id=pad_token_id,
        )
        checkpoint_dir = output_dir / f"epoch_{epoch:03d}"
        row = {
            "experiment_id": experiment.id,
            "epoch": epoch,
            "lr": experiment.lr,
            "batch_size": experiment.batch_size,
            "micro_batch_size": suite.micro_batch_size,
            "train_examples": len(train_dataset),
            "eval_examples": len(eval_dataset),
            "train_loss": train_loss,
            "eval_loss": eval_result.loss,
            "eval_accuracy": eval_result.accuracy,
            "eval_correct": eval_result.correct,
            "eval_total": eval_result.total,
            "checkpoint": str(checkpoint_dir),
            "training_params_file": str(checkpoint_dir / "training_params.json"),
            "eval_result_file": str(checkpoint_dir / "eval_result.json"),
        }
        eval_payload = asdict(eval_result)
        metadata = {
            "experiment_id": experiment.id,
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "train_tokens": epoch_tokens,
            "training_params": params,
            "eval_result": eval_payload,
        }
        save_epoch_checkpoint(model, checkpoint_dir, metadata=metadata)
        write_json(checkpoint_dir / "training_params.json", params)
        write_json(checkpoint_dir / "eval_result.json", eval_payload)
        write_json(checkpoint_dir / "result.json", row)
        rows.append(row)
        append_result(suite.output_root, row)
        print(
            f"{experiment.id} epoch {epoch}: "
            f"train_loss={train_loss:.4f} "
            f"eval_loss={eval_result.loss:.4f} "
            f"eval_acc={eval_result.accuracy:.3f} "
            f"({eval_result.correct}/{eval_result.total})"
        )

    del optimizer
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def write_results(output_root: str | Path, rows: list[dict[str, Any]]) -> None:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_root / "results.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    if rows:
        csv_path = output_root / "results.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_result(output_root: str | Path, row: dict[str, Any]) -> None:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_root / "results.jsonl"
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")

    rows = read_jsonl(jsonl_path)
    if rows:
        csv_path = output_root / "results.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def reset_result_files(output_root: str | Path) -> None:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for name in ("results.jsonl", "results.csv"):
        path = output_root / name
        if path.exists():
            path.unlink()


def run_suite(args: argparse.Namespace) -> None:
    suite = load_suite_config(args.config)
    torch.manual_seed(suite.seed)
    random.seed(suite.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(suite.seed)
        torch.set_float32_matmul_precision("high")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(suite.model_path)
    pad_token_id = (
        getattr(tokenizer, "pad_token_id", None)
        or getattr(tokenizer, "eos_token_id", None)
        or 0
    )

    train_rows = load_imdb_rows(
        suite.train_split,
        max_examples=suite.train_examples,
        seed=suite.seed,
    )
    eval_rows = load_imdb_rows(
        suite.eval_split,
        max_examples=suite.eval_examples,
        seed=suite.seed + 1,
    )
    print(
        f"loaded IMDb rows: train={len(train_rows)} eval={len(eval_rows)} "
        f"max_length={suite.max_length}"
    )
    train_dataset = encode_rows(tokenizer, train_rows, suite.max_length)
    eval_dataset = encode_rows(tokenizer, eval_rows, suite.max_length)

    reset_result_files(suite.output_root)
    all_rows: list[dict[str, Any]] = []
    for index, experiment in enumerate(suite.experiments, start=1):
        print(f"starting experiment {index}/{len(suite.experiments)}: {experiment.id}")
        rows = train_one_experiment(
            suite,
            experiment,
            train_dataset,
            eval_dataset,
            tokenizer,
            device=device,
            pad_token_id=int(pad_token_id),
        )
        all_rows.extend(rows)
        write_results(suite.output_root, all_rows)

    print(f"wrote results to {Path(suite.output_root).resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autoresearch IMDb SFT experiments for Qwen3-0.6B.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    suite_parser = subparsers.add_parser("run-suite", help="Run the configured experiment suite.")
    suite_parser.add_argument("--config", default="autoresearch/imdb_sft/experiments.json")
    suite_parser.add_argument("--device", default=None)
    suite_parser.set_defaults(func=run_suite)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
