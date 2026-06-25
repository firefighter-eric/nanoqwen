from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = [
        "id",
        "model",
        "parameters",
        "parameter_ratio_to_gpt",
        "step",
        "elapsed_sec",
        "train_loss",
        "val_loss",
        "val_ppl",
        "val_bpb",
        "output_dir",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def append_arg(cmd: list[str], flag: str, value: Any) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def experiment_value(suite: dict[str, Any], experiment: dict[str, Any], key: str, default: Any = None) -> Any:
    return experiment.get(key, suite.get(key, default))


def train_command(
    suite: dict[str, Any],
    experiment: dict[str, Any],
    output_dir: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/train.py",
        "--out-dir",
        str(output_dir),
        "--model",
        experiment["model"],
    ]

    for key, flag in [
        ("data", "--data"),
        ("dataset", "--dataset"),
        ("data_dir", "--data-dir"),
        ("val_data", "--val-data"),
        ("data_format", "--data-format"),
        ("dataset_num_shards", "--dataset-num-shards"),
        ("download_workers", "--download-workers"),
        ("tokenizer", "--tokenizer"),
        ("max_data_chars", "--max-data-chars"),
        ("max_val_chars", "--max-val-chars"),
        ("steps", "--steps"),
        ("time_budget_sec", "--time-budget-sec"),
        ("batch_size", "--batch-size"),
        ("block_size", "--block-size"),
        ("lr", "--lr"),
        ("weight_decay", "--weight-decay"),
        ("eval_every", "--eval-every"),
        ("eval_iters", "--eval-iters"),
        ("eval_tokens", "--eval-tokens"),
        ("save_every", "--save-every"),
        ("device", "--device"),
        ("seed", "--seed"),
        ("vocab_size", "--vocab-size"),
        ("hidden_size", "--hidden-size"),
        ("intermediate_size", "--intermediate-size"),
        ("layers", "--layers"),
        ("heads", "--heads"),
        ("kv_heads", "--kv-heads"),
        ("dropout", "--dropout"),
        ("rope_theta", "--rope-theta"),
        ("window_pattern", "--window-pattern"),
        ("attn_implementation", "--attn-implementation"),
    ]:
        append_arg(cmd, flag, experiment_value(suite, experiment, key))

    if experiment_value(suite, experiment, "download", False):
        cmd.append("--download")
    if experiment_value(suite, experiment, "use_dataset_val", False):
        cmd.append("--use-dataset-val")
    if experiment_value(suite, experiment, "no_gpt_bias", False):
        cmd.append("--no-gpt-bias")
    if experiment_value(suite, experiment, "tie_word_embeddings", False):
        cmd.append("--tie-word-embeddings")
    if experiment_value(suite, experiment, "no_qk_norm", False):
        cmd.append("--no-qk-norm")
    return cmd


def run_suite(config_path: str | Path) -> None:
    suite = read_json(config_path)
    output_root = Path(suite["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    gpt_parameters: int | None = None
    for experiment in suite["experiments"]:
        exp_dir = output_root / experiment["id"]
        exp_dir.mkdir(parents=True, exist_ok=True)
        log_path = exp_dir / "train.log"
        cmd = train_command(suite, experiment, exp_dir)
        print(f"[pretrain-arch] {experiment['id']}: {' '.join(cmd)}")
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)

        result = read_json(exp_dir / "result.json")
        params = int(result["parameters"])
        if experiment["model"] == "gpt" and gpt_parameters is None:
            gpt_parameters = params
        row = {
            "id": experiment["id"],
            "model": experiment["model"],
            "parameters": params,
            "parameter_ratio_to_gpt": (params / gpt_parameters) if gpt_parameters else None,
            "step": result["step"],
            "elapsed_sec": result["elapsed_sec"],
            "train_loss": result["train_loss"],
            "val_loss": result["val_loss"],
            "val_ppl": result["val_ppl"],
            "val_bpb": result.get("val_bpb"),
            "output_dir": str(exp_dir),
            "log_file": str(log_path),
        }
        rows.append(row)
        write_jsonl(output_root / "results.jsonl", rows)
        write_csv(output_root / "results.csv", rows)
        print(
            f"[pretrain-arch] {experiment['id']}: "
            f"params={params:,} val_loss={row['val_loss']:.4f} "
            f"val_bpb={row['val_bpb']} "
            f"elapsed_sec={row['elapsed_sec']:.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GPT, autoresearch NanoGPT, and Qwen-like pretraining suites.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    suite_parser = subparsers.add_parser("run-suite")
    suite_parser.add_argument("--config", default="autoresearch/pretrain_arch_compare/experiments.json")
    args = parser.parse_args()

    if args.command == "run-suite":
        run_suite(args.config)


if __name__ == "__main__":
    main()
