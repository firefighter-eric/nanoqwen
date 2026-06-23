from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .checkpoint import load_checkpoint
from .config import NanoqwenConfig
from .data import built_in_tiny_dataset, load_text_dataset
from .eval import LossEvalResult, evaluate_lm_loss


@dataclass
class CheckpointReport:
    checkpoint: str
    step: int
    parameters: int
    trainable_parameters: int
    checkpoint_bytes: int
    config: dict[str, Any]
    extra: dict[str, Any]
    eval: dict[str, Any] | None = None


def file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def parameter_counts(model) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def checkpoint_report(
    checkpoint: str | Path,
    data: str | Path | None = None,
    block_size: int = 128,
    batch_size: int = 16,
    max_batches: int | None = None,
    device: str = "cpu",
) -> CheckpointReport:
    checkpoint = Path(checkpoint)
    model, metadata = load_checkpoint(checkpoint, map_location=device)
    total, trainable = parameter_counts(model)

    eval_result: LossEvalResult | None = None
    if data is not None:
        dataset = load_text_dataset(data, block_size=block_size)
        eval_result = evaluate_lm_loss(
            model,
            dataset,
            batch_size=batch_size,
            device=device,
            max_batches=max_batches,
        )
    elif max_batches is not None:
        dataset = built_in_tiny_dataset(block_size=block_size)
        eval_result = evaluate_lm_loss(
            model,
            dataset,
            batch_size=batch_size,
            device=device,
            max_batches=max_batches,
        )

    return CheckpointReport(
        checkpoint=str(checkpoint),
        step=int(metadata.get("step", 0)),
        parameters=total,
        trainable_parameters=trainable,
        checkpoint_bytes=file_size(checkpoint / "checkpoint.pt"),
        config=model.config.to_dict(),
        extra=metadata.get("extra", {}),
        eval=asdict(eval_result) if eval_result else None,
    )


def report_to_dict(report: CheckpointReport) -> dict[str, Any]:
    return asdict(report)


def report_to_json(report: CheckpointReport) -> str:
    return json.dumps(report_to_dict(report), indent=2, sort_keys=True) + "\n"


def compact_config(config: dict[str, Any]) -> dict[str, Any]:
    fields = [
        "model_type",
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "max_position_embeddings",
        "use_qk_norm",
        "attention_bias",
        "attention_output_bias",
    ]
    return {field: config.get(field) for field in fields if field in config}


def report_to_markdown(report: CheckpointReport) -> str:
    lines = [
        "# nanoqwen Checkpoint Report",
        "",
        f"- checkpoint: `{report.checkpoint}`",
        f"- step: `{report.step}`",
        f"- parameters: `{report.parameters:,}`",
        f"- trainable parameters: `{report.trainable_parameters:,}`",
        f"- checkpoint size: `{report.checkpoint_bytes:,}` bytes",
        "",
        "## Config",
        "",
    ]
    for key, value in compact_config(report.config).items():
        lines.append(f"- `{key}`: `{value}`")

    if report.extra:
        lines.extend(["", "## Extra", ""])
        for key, value in report.extra.items():
            lines.append(f"- `{key}`: `{value}`")

    if report.eval:
        lines.extend(["", "## Eval", ""])
        for key, value in report.eval.items():
            if isinstance(value, float):
                lines.append(f"- `{key}`: `{value:.6g}`")
            else:
                lines.append(f"- `{key}`: `{value}`")

    return "\n".join(lines) + "\n"

