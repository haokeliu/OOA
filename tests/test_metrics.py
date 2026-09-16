import numpy as np
import pytest

from edcr.metrics import candidate_opportunity, context_false_positive_rate, macro_average_precision


def test_context_fpr_uses_only_target_negative_source_positive():
    target = np.array([0, 0, 1, 0])
    source = np.array([1, 0, 1, 1])
    prediction = np.array([1, 1, 1, 0])
    assert context_false_positive_rate(target, source, prediction) == 0.5


def test_context_fpr_without_support_is_null():
    zeros = np.zeros(3)
    assert context_false_positive_rate(zeros, zeros, zeros) is None


def test_macro_ap_skips_unsupported_class():
    labels = np.array([[1, 0], [0, 0]])
    scores = np.array([[0.9, 0.1], [0.1, 0.9]])
    assert macro_average_precision(labels, scores) == 1.0


def test_candidate_opportunity_counts_beneficial_harmful_and_ignored():
    labels = np.array([[1.0], [0.0], [1.0]])
    base = np.zeros((3, 1))
    context = np.array([[1.0], [1.0], [0.0]])
    result = candidate_opportunity(labels, base, context, [0], 1e-4)
    assert result["beneficial_fraction"] == 1 / 3
    assert result["harmful_fraction"] == 1 / 3
    assert result["ignored_fraction"] == 1 / 3
    assert result["passes_preregistered_support"] is True


def test_candidate_opportunity_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="equal"):
        candidate_opportunity(np.zeros((2, 1)), np.zeros((1, 1)), np.zeros((2, 1)), [0], 0.0)
