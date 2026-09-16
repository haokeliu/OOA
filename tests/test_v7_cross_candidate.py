from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from run_v7_cross_candidate_development import (
    bce_gain,
    feature_sets,
    soft_action,
    stratified_folds,
)


def test_stratified_folds_are_deterministic_and_balanced() -> None:
    image_ids = np.asarray([f"image-{index}" for index in range(47)])
    labels = np.asarray([0] * 22 + [1] * 25, dtype=np.float64)
    first = stratified_folds(image_ids, labels, "source_to_target")
    second = stratified_folds(image_ids, labels, "source_to_target")
    np.testing.assert_array_equal(first, second)
    for label in (0, 1):
        counts = [int(np.sum((first == fold) & (labels == label))) for fold in range(5)]
        assert max(counts) - min(counts) <= 1


def test_soft_action_hits_conditional_logit_when_inside_segment() -> None:
    base = np.asarray([-2.0, 2.0, -1.0])
    residual = np.asarray([4.0, -4.0, 0.0])
    probability = np.asarray([0.5, 0.5, 0.8])
    action = soft_action(probability, base, residual)
    np.testing.assert_allclose(action[:2], 0.5)
    assert action[2] == 0.0


def test_soft_gain_never_exceeds_revealed_endpoint_oracle() -> None:
    base = np.linspace(-3.0, 3.0, 31)
    residual = np.linspace(4.0, -4.0, 31)
    probability = np.linspace(0.01, 0.99, 31)
    action = soft_action(probability, base, residual)
    for label_value in (0.0, 1.0):
        labels = np.full_like(base, label_value)
        soft_gain = bce_gain(base, residual, labels, action)
        endpoint_gain = bce_gain(base, residual, labels, 1.0)
        assert np.max(soft_gain - np.maximum(endpoint_gain, 0.0)) <= 1e-12


def test_cross_candidate_feature_dimensions() -> None:
    global_logits = np.arange(24, dtype=np.float64).reshape(3, 8)
    crop_logits = global_logits + 0.25
    crop_view_logits = np.repeat(crop_logits[:, None, :], 4, axis=1)
    matrices = feature_sets(global_logits, crop_logits, crop_view_logits, 2, 5)
    assert matrices["endpoint_confidence"].shape == (3, 2)
    assert matrices["summary"].shape == (3, 6)
    assert matrices["view_logits"].shape == (3, 14)
    assert all(np.isfinite(matrix).all() for matrix in matrices.values())
