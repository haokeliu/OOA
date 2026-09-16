from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from run_v10_matched_baseline_calibration import (
    compute_gains,
    fit_static_action,
    fit_temperature_with_bounds,
    modal_grid_value,
)


def test_static_action_improves_over_endpoints_for_symmetric_logits() -> None:
    logits_a = np.asarray([[3.0, 0.0], [0.0, 1.0]])
    logits_b = np.asarray([[1.0, 0.0], [0.0, 3.0]])
    labels = np.asarray([0, 1])
    q = fit_static_action(logits_a, logits_b, labels)
    assert 0.0 < q < 1.0


def test_temperature_wide_bound_recovers_known_scale() -> None:
    logits = np.asarray([[4.0, 0.0], [0.0, 4.0], [2.0, 0.0], [0.0, 2.0]])
    labels = np.asarray([0, 1, 0, 1])
    temperature, loss = fit_temperature_with_bounds(logits, labels, (-8.0, 3.0))
    assert 0.0 < temperature < 1.0
    assert np.isfinite(loss)


def test_modal_grid_value_prefers_smaller_value_on_tie() -> None:
    assert modal_grid_value([0.1, 1.0], (0.1, 1.0, 10.0)) == 0.1


def test_compute_gains_zeroes_identical_candidates() -> None:
    logits = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    labels = np.asarray([0, 1])
    q = np.asarray([0.2, 0.8])
    actions = {
        "label_budget_matched_static": q,
        "tree_hard": np.asarray([0.0, 1.0]),
        "linear_gain_hard": np.asarray([0.0, 1.0]),
        "same_score_projected_hard": np.asarray([0.0, 1.0]),
        "direct_soft": q,
    }
    gains = compute_gains(logits, logits, labels, actions, 0.0, 0.5)
    for values in gains.values():
        np.testing.assert_allclose(values, 0.0)
