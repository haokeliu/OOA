import pytest
import torch
from torch import nn

from edcr.models import (
    CandidateModel,
    MultiViewHead,
    RelationIndex,
    apply_gate,
    smooth_max_pool,
)


def test_smooth_max_pool_constant_and_shape():
    logits = torch.full((2, 5, 3), 4.0)
    pooled = smooth_max_pool(logits, 1.0)
    assert pooled.shape == (2, 3)
    assert torch.allclose(pooled, torch.full((2, 3), 4.0))


def test_pooling_rejects_per_view_label_shape():
    with pytest.raises(ValueError):
        smooth_max_pool(torch.zeros(2, 3))


def test_candidate_only_modifies_selected_target_and_is_bounded():
    base_head = MultiViewHead(4, 3)
    full_head = nn.Linear(4, 3)
    model = CandidateModel(base_head, full_head, [RelationIndex(source=0, target=2)], 2.0)
    with torch.no_grad():
        model.context_raw.fill_(100.0)
        model.full_head.weight.zero_()
        model.full_head.bias.fill_(100.0)
    outputs = model(torch.zeros(2, 5, 4))
    assert torch.allclose(outputs["residual"][:, :2], torch.zeros(2, 2))
    assert torch.all(outputs["residual"][:, 2] <= 2.0)
    assert torch.all(outputs["residual"][:, 2] > 1.99)


def test_candidate_can_exclude_full_view_from_evidence_baseline():
    base_head = MultiViewHead(1, 1)
    full_head = nn.Linear(1, 1)
    model = CandidateModel(
        base_head,
        full_head,
        [],
        base_view_indices=(1, 2, 3, 4),
    )
    with torch.no_grad():
        base_head.linear.weight.fill_(1.0)
        base_head.linear.bias.zero_()
    features = torch.tensor([[[100.0], [1.0], [1.0], [1.0], [1.0]]])
    assert torch.allclose(model(features)["base"], torch.tensor([[1.0]]))


def test_centered_context_correction_changes_sign_around_center():
    base_head = MultiViewHead(1, 2)
    full_head = nn.Linear(1, 2)
    model = CandidateModel(
        base_head,
        full_head,
        [RelationIndex(source=0, target=1)],
        context_centers=torch.tensor([0.5]),
    )
    with torch.no_grad():
        model.context_raw.fill_(1.0)
        full_head.weight.zero_()
        full_head.bias[0] = -2.0
    low = model(torch.zeros(1, 5, 1))["residual"][0, 1]
    with torch.no_grad():
        full_head.bias[0] = 2.0
    high = model(torch.zeros(1, 5, 1))["residual"][0, 1]
    assert low < 0 < high


def test_standardized_logit_context_signal_is_bounded():
    base_head = MultiViewHead(1, 2)
    full_head = nn.Linear(1, 2)
    model = CandidateModel(
        base_head,
        full_head,
        [RelationIndex(source=0, target=1)],
        residual_bound=2.0,
        context_centers=torch.tensor([0.0]),
        context_scales=torch.tensor([0.5]),
        context_signal="standardized_logit",
    )
    with torch.no_grad():
        model.context_raw.fill_(100.0)
        full_head.weight.zero_()
        full_head.bias[0] = 100.0
    residual = model(torch.zeros(1, 5, 1))["residual"][0, 1]
    assert 1.99 < residual <= 2.0


def test_gate_degeneracies():
    base = torch.tensor([[1.0, 2.0]])
    residual = torch.tensor([[0.5, -0.5]])
    assert torch.equal(apply_gate(base, residual, torch.tensor([0.0])), base)
    assert torch.equal(apply_gate(base, residual, torch.tensor([1.0])), base + residual)
    assert torch.equal(apply_gate(base, torch.zeros_like(base), torch.tensor([0.7])), base)
