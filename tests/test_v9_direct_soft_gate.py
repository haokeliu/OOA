from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from run_v9a_direct_soft_gate import (
    direct_soft_objective,
    fit_direct_soft_gate,
    predict_direct_soft,
)


def test_direct_soft_gradient_matches_finite_difference() -> None:
    generator = np.random.default_rng(9)
    features = generator.normal(size=(12, 3))
    base = generator.normal(size=12)
    residual = generator.normal(size=12)
    labels = generator.integers(0, 2, size=12).astype(np.float64)
    parameters = generator.normal(scale=0.2, size=4)
    _, gradient = direct_soft_objective(
        parameters, features, base, residual, labels, 1e-2
    )
    epsilon = 1e-6
    numerical = np.empty_like(parameters)
    for index in range(len(parameters)):
        high = parameters.copy()
        low = parameters.copy()
        high[index] += epsilon
        low[index] -= epsilon
        numerical[index] = (
            direct_soft_objective(high, features, base, residual, labels, 1e-2)[0]
            - direct_soft_objective(low, features, base, residual, labels, 1e-2)[0]
        ) / (2 * epsilon)
    np.testing.assert_allclose(gradient, numerical, atol=1e-6, rtol=1e-5)


def test_direct_soft_predictions_are_strictly_interior() -> None:
    features = np.asarray([[-1.0], [0.0], [1.0], [2.0]])
    base = np.asarray([-1.0, -0.5, 0.5, 1.0])
    residual = np.asarray([1.0, 1.0, -1.0, -1.0])
    labels = np.asarray([0.0, 1.0, 1.0, 0.0])
    parameters = fit_direct_soft_gate(features, base, residual, labels, 1e-2)
    action = predict_direct_soft(parameters, features)
    assert np.all((action > 0) & (action < 1))
