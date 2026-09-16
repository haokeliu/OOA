import math

import pytest

from edcr.gate_targets import (
    bce_with_logits,
    hard_gate_regret,
    loss_difference,
    oracle_gate,
    update_dual,
)


@pytest.mark.parametrize("logit", [-1000.0, -10.0, 0.0, 10.0, 1000.0])
@pytest.mark.parametrize("target", [0, 1])
def test_bce_extreme_logits_are_finite(logit, target):
    assert math.isfinite(bce_with_logits(logit, target))


@pytest.mark.parametrize("base", [-2.0, 0.0, 2.0])
@pytest.mark.parametrize("residual", [-1.0, 1.0])
@pytest.mark.parametrize("target", [0, 1])
@pytest.mark.parametrize("gate", [0, 1])
def test_hard_gate_regret_identity(base, residual, target, gate):
    oracle = oracle_gate(base, residual, target)
    expected = abs(loss_difference(base, residual, target)) * int(gate != oracle)
    assert hard_gate_regret(base, residual, target, gate) == pytest.approx(expected)


def test_positive_residual_gain_target_degenerates_to_label():
    for base in [-20.0, -1.0, 0.0, 1.0, 20.0]:
        assert oracle_gate(base, 0.5, 0) == 0
        assert oracle_gate(base, 0.5, 1) == 1


def test_dual_projection_direction():
    assert update_dual(0.2, 1.0, 0.01) > 0.2
    assert update_dual(0.2, -1.0, 0.01) < 0.2
    assert update_dual(0.0, -1.0, 0.01) == 0.0
