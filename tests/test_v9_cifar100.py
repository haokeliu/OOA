from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from evaluate_v9b_cifar100_sealed import candidate_logits, cluster_summary
from run_v9b0_cifar100_feasibility import (
    deterministic_router_folds,
    direct_soft_objective,
    router_features,
)


def test_multiclass_direct_soft_gradient_matches_finite_difference() -> None:
    generator = np.random.default_rng(12)
    features = generator.normal(size=(9, 4))
    logits_a = generator.normal(size=(9, 5))
    logits_b = generator.normal(size=(9, 5))
    labels = generator.integers(0, 5, size=9)
    parameters = generator.normal(scale=0.1, size=5)
    _, gradient = direct_soft_objective(parameters, features, logits_a, logits_b, labels, 1e-2)
    epsilon = 1e-6
    numerical = np.empty_like(parameters)
    for index in range(len(parameters)):
        high = parameters.copy()
        low = parameters.copy()
        high[index] += epsilon
        low[index] -= epsilon
        numerical[index] = (
            direct_soft_objective(high, features, logits_a, logits_b, labels, 1e-2)[0]
            - direct_soft_objective(low, features, logits_a, logits_b, labels, 1e-2)[0]
        ) / (2 * epsilon)
    np.testing.assert_allclose(gradient, numerical, atol=1e-6, rtol=1e-5)


def test_router_features_are_finite_and_label_free() -> None:
    generator = np.random.default_rng(4)
    matrix = router_features(generator.normal(size=(7, 100)), generator.normal(size=(7, 100)))
    assert matrix.shape == (7, 10)
    assert np.isfinite(matrix).all()


def test_router_folds_are_balanced_within_class() -> None:
    indices = np.arange(1000)
    labels = np.repeat(np.arange(100), 10)
    folds = deterministic_router_folds(indices, labels)
    for class_index in range(100):
        counts = np.bincount(folds[labels == class_index], minlength=5)
        np.testing.assert_array_equal(counts, np.full(5, 2))


def test_candidate_logits_apply_head_and_temperature() -> None:
    features = np.asarray([[1.0, 2.0], [-1.0, 0.5]])
    coefficient = np.asarray([[2.0, -1.0], [0.5, 3.0]])
    intercept = np.asarray([0.25, -0.5])
    expected = (features @ coefficient.T + intercept) / 2.0
    np.testing.assert_allclose(candidate_logits(features, coefficient, intercept, 2.0), expected)


def test_cluster_summary_uses_twenty_balanced_blocks() -> None:
    clusters = np.repeat(np.arange(20), 500)
    values = np.repeat(np.arange(20, dtype=np.float64), 500)
    summary = cluster_summary(values, clusters, np.random.default_rng(3))
    assert summary["sample_count"] == 10_000
    assert summary["cluster_count"] == 20
    assert summary["mean"] == 9.5
