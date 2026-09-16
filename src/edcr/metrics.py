"""Leakage-safe EDCR evaluation metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


def per_class_average_precision(labels: np.ndarray, scores: np.ndarray) -> list[float | None]:
    if labels.shape != scores.shape or labels.ndim != 2:
        raise ValueError("labels and scores must have equal [samples, classes] shape")
    output: list[float | None] = []
    for class_index in range(labels.shape[1]):
        target = labels[:, class_index]
        output.append(
            None
            if target.sum() == 0
            else float(average_precision_score(target, scores[:, class_index]))
        )
    return output


def macro_average_precision(
    labels: np.ndarray, scores: np.ndarray, class_indices: list[int] | None = None
) -> float | None:
    values = per_class_average_precision(labels, scores)
    selected = values if class_indices is None else [values[index] for index in class_indices]
    supported = [value for value in selected if value is not None]
    return float(np.mean(supported)) if supported else None


def context_false_positive_rate(
    target_labels: np.ndarray,
    source_labels: np.ndarray,
    target_predictions: np.ndarray,
) -> float | None:
    if not (
        target_labels.shape == source_labels.shape == target_predictions.shape
        and target_labels.ndim == 1
    ):
        raise ValueError("cFPR inputs must be equal one-dimensional arrays")
    support = (target_labels == 0) & (source_labels == 1)
    if not support.any():
        return None
    return float(np.mean(target_predictions[support] == 1))


def candidate_opportunity(
    labels: np.ndarray,
    base_logits: np.ndarray,
    context_logits: np.ndarray,
    target_indices: list[int],
    epsilon: float,
) -> dict[str, object]:
    """Summarize per-sample BCE preference between two frozen candidates."""
    if not (labels.shape == base_logits.shape == context_logits.shape and labels.ndim == 2):
        raise ValueError("candidate arrays must have equal [samples, classes] shape")
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    selected_labels = labels[:, target_indices]
    base = base_logits[:, target_indices]
    context = context_logits[:, target_indices]
    base_loss = np.logaddexp(0.0, base) - selected_labels * base
    context_loss = np.logaddexp(0.0, context) - selected_labels * context
    difference = base_loss - context_loss

    def fractions(values: np.ndarray) -> dict[str, float]:
        return {
            "beneficial_fraction": float(np.mean(values > epsilon)),
            "harmful_fraction": float(np.mean(values < -epsilon)),
            "ignored_fraction": float(np.mean(np.abs(values) <= epsilon)),
        }

    overall = fractions(difference)
    return {
        **overall,
        "passes_preregistered_support": bool(
            overall["beneficial_fraction"] >= 0.10 and overall["harmful_fraction"] >= 0.10
        ),
        "per_target": [fractions(difference[:, index]) for index in range(len(target_indices))],
    }
