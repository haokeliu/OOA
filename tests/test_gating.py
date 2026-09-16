import pytest
import torch
from torch import nn

from edcr.gating import (
    ConfidenceGate,
    ConstantGate,
    build_gate_tensors,
    gate_values,
    natural_group_masks,
    predict_gate,
    target_balanced_bce,
)
from edcr.models import CandidateModel, GateNetwork, MultiViewHead, RelationIndex


def make_candidate() -> CandidateModel:
    base = MultiViewHead(3, 3)
    full = nn.Linear(3, 3)
    return CandidateModel(
        base,
        full,
        [RelationIndex(source=0, target=2)],
        context_signal="standardized_logit",
    )


def test_gate_numeric_features_do_not_depend_on_labels():
    model = make_candidate()
    features = torch.randn(4, 5, 3)
    zeros = build_gate_tensors(model, features, torch.zeros(4, 3))
    ones = build_gate_tensors(model, features, torch.ones(4, 3))
    assert torch.equal(zeros.numeric, ones.numeric)
    assert not torch.equal(zeros.target_labels, ones.target_labels)


def test_natural_group_masks_follow_relation_labels_and_weakness():
    model = make_candidate()
    labels = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    tensors = build_gate_tensors(model, torch.randn(2, 5, 3), labels)
    negative, weak = natural_group_masks(tensors, torch.tensor([[False], [True]]))
    assert negative[:, 0].tolist() == [True, False]
    assert weak[:, 0].tolist() == [False, True]


def test_target_balanced_bce_weights_relations_equally():
    logits = torch.zeros(3, 2)
    labels = torch.tensor([[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]])
    mask = torch.tensor([[True, True], [False, True], [False, False]])
    assert target_balanced_bce(logits, labels, mask).item() == pytest.approx(0.69314718)


def test_target_balanced_bce_requires_each_relation():
    with pytest.raises(ValueError, match="no group support"):
        target_balanced_bce(
            torch.zeros(2, 2),
            torch.zeros(2, 2),
            torch.tensor([[True, False], [True, False]]),
        )


@pytest.mark.parametrize("method", ["B3", "B4", "B4C", "B4G", "B5", "M2"])
def test_predict_gate_matches_flat_gate_values(method):
    model = make_candidate()
    tensors = build_gate_tensors(model, torch.randn(4, 5, 3), torch.zeros(4, 3))
    if method == "B3":
        gate = ConstantGate(1)
    elif method in {"B4", "B4C", "B4G"}:
        gate = ConfidenceGate(4)
    else:
        gate = GateNetwork(tensors.numeric.shape[-1], tensors.source_embeddings.shape[-1], 4)
    expected = gate_values(method, gate, tensors, torch.arange(tensors.base.numel())).reshape_as(
        tensors.base
    )
    actual = predict_gate(method, gate, tensors, batch_size=3)
    assert torch.allclose(actual, expected)
