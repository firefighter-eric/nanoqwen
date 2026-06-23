from __future__ import annotations

from nanoqwen import qwen3_model, qwen35_model


def test_qwen_model_files_have_separate_defaults() -> None:
    assert qwen3_model.REPO_ID == "Qwen/Qwen3-0.6B"
    assert qwen35_model.REPO_ID == "Qwen/Qwen3.5-0.8B"
    assert qwen3_model.DEFAULT_MODEL_PATH != qwen35_model.DEFAULT_MODEL_PATH


def test_qwen3_missing_files_reports_required_files(tmp_path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")

    assert qwen3_model.missing_files(str(tmp_path)) == [
        "tokenizer_config.json",
        "model.safetensors",
    ]


def test_qwen35_missing_files_reports_required_files(tmp_path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    assert qwen35_model.missing_files(str(tmp_path)) == [
        "model.safetensors-00001-of-00001.safetensors",
    ]
