from __future__ import annotations

import math

from scripts.train import muon_momentum, scheduled_weight_decay


def test_muon_momentum_can_disable_ramp() -> None:
    assert muon_momentum(step=0, start=0.95, end=0.95, ramp_steps=0) == 0.95
    assert muon_momentum(step=200, start=0.95, end=0.95, ramp_steps=0) == 0.95


def test_muon_momentum_ramps_to_end_value() -> None:
    assert muon_momentum(step=0, start=0.85, end=0.95, ramp_steps=300) == 0.85
    assert muon_momentum(step=300, start=0.85, end=0.95, ramp_steps=300) == 0.95


def test_scheduled_weight_decay_supports_linear_and_constant() -> None:
    assert math.isclose(scheduled_weight_decay(0.2, progress=0.25, schedule="linear"), 0.15)
    assert scheduled_weight_decay(0.2, progress=0.25, schedule="constant") == 0.2
