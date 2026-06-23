from __future__ import annotations

import json

from nanoqwen.checkpoint import save_checkpoint
from nanoqwen.config import NanoqwenConfig
from nanoqwen.model import NanoqwenForCausalLM
from nanoqwen.report import checkpoint_report, report_to_json, report_to_markdown


def test_checkpoint_report_roundtrip(tmp_path) -> None:
    model = NanoqwenForCausalLM(NanoqwenConfig.tiny(vocab_size=32))
    save_checkpoint(model, tmp_path, step=7, extra={"stage": "unit"})

    report = checkpoint_report(tmp_path)
    markdown = report_to_markdown(report)
    payload = json.loads(report_to_json(report))

    assert report.step == 7
    assert report.parameters > 0
    assert report.trainable_parameters == report.parameters
    assert report.checkpoint_bytes > 0
    assert report.extra == {"stage": "unit"}
    assert "# nanoqwen Checkpoint Report" in markdown
    assert payload["step"] == 7
    assert payload["config"]["vocab_size"] == 32

