from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from run_v8_calibration_baselines import (
    CALIBRATION_METHODS,
    fit_calibrator,
    fixed_bin_ece,
    probability_nll,
)


def test_all_calibrators_return_finite_probabilities() -> None:
    calibration_probability = np.asarray([0.05, 0.2, 0.4, 0.7, 0.85, 0.95])
    labels = np.asarray([0, 0, 1, 0, 1, 1], dtype=np.float64)
    evaluation_probability = np.asarray([0.0, 0.1, 0.5, 0.9, 1.0])
    for method in CALIBRATION_METHODS:
        values = fit_calibrator(method, calibration_probability, labels)(
            evaluation_probability
        )
        assert values.shape == evaluation_probability.shape
        assert np.isfinite(values).all()
        assert np.all((values > 0) & (values < 1))


def test_temperature_scaling_can_reduce_overconfidence_nll() -> None:
    probability = np.asarray([0.001, 0.01, 0.1, 0.9, 0.99, 0.999])
    labels = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.float64)
    calibrated = fit_calibrator("temperature", probability, labels)(probability)
    assert probability_nll(calibrated, labels) < probability_nll(probability, labels)


def test_fixed_bin_ece_is_zero_for_balanced_constant_forecast() -> None:
    probability = np.full(10, 0.5)
    labels = np.asarray([0, 1] * 5, dtype=np.float64)
    assert fixed_bin_ece(probability, labels) == 0.0
