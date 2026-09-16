"""Calibration-only threshold selection with deterministic tie behavior."""

from __future__ import annotations

import math

import numpy as np


def threshold_at_recall(
    labels: np.ndarray,
    scores: np.ndarray,
    target_recall: float,
    min_positives: int,
) -> float | None:
    """Return the highest threshold whose positive recall is at least the target.

    Predictions use score >= threshold. Ties can therefore make achieved recall
    exceed the requested value. Unsupported classes return None.
    """
    if labels.shape != scores.shape or labels.ndim != 1:
        raise ValueError("labels and scores must be equal one-dimensional arrays")
    if not 0.0 < target_recall <= 1.0:
        raise ValueError("target_recall must be in (0, 1]")
    positive_scores = scores[labels == 1]
    if len(positive_scores) < min_positives:
        return None
    required = math.ceil(target_recall * len(positive_scores))
    ordered = np.sort(positive_scores)[::-1]
    return float(ordered[required - 1])


def calibrate_classes(
    labels: np.ndarray,
    scores: np.ndarray,
    target_recall: float,
    min_positives: int,
) -> list[float | None]:
    if labels.shape != scores.shape or labels.ndim != 2:
        raise ValueError("labels and scores must have equal [samples, classes] shape")
    return [
        threshold_at_recall(labels[:, index], scores[:, index], target_recall, min_positives)
        for index in range(labels.shape[1])
    ]
