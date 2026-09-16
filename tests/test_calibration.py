import numpy as np
import pytest

from edcr.calibration import threshold_at_recall


def test_highest_threshold_at_recall_and_ties():
    labels = np.ones(5)
    scores = np.array([0.9, 0.8, 0.8, 0.2, 0.1])
    threshold = threshold_at_recall(labels, scores, 0.6, 5)
    assert threshold == 0.8
    assert np.mean(scores >= threshold) == 0.6


def test_insufficient_positives_returns_none():
    labels = np.array([1, 0, 0])
    scores = np.array([0.9, 0.8, 0.7])
    assert threshold_at_recall(labels, scores, 0.8, 2) is None


def test_invalid_target_recall_rejected():
    with pytest.raises(ValueError):
        threshold_at_recall(np.ones(2), np.ones(2), 0.0, 1)
